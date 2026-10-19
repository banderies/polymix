//! Agent access: a Model Context Protocol (MCP) server that lets LLM clients (Claude Code,
//! Claude Desktop, Cursor, …) read the running app.
//!
//! The server runs on its own thread (tokio + the official `rmcp` SDK) and owns no app state:
//! each tool call becomes a [`Call`] that the app takes from [`Server::try_recv`] on its UI
//! thread, between frames, and answers with a [`Reply`]. A reply can be [`Reply::Later`]: work
//! the UI thread snapshotted but should not wait on (statistics of a big mesh, PNG encoding),
//! which then runs on a worker thread here.
//!
//! It listens on 127.0.0.1 only and every request needs the bearer token. Requests from web
//! pages (anything carrying an `Origin` header) and with a non-local `Host` are refused, so a
//! site open in a browser cannot reach it (DNS rebinding).

use base64::Engine;
use rmcp::model::{CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation, ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool, ToolAnnotations};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData, RoleServer, ServerHandler};
use serde_json::Value;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tokio_util::sync::CancellationToken;

/// Port the server listens on unless told otherwise.
pub const DEFAULT_PORT: u16 = 47390;
/// Path of the MCP endpoint.
pub const PATH: &str = "/mcp";
/// Calls remembered for [`Server::recent`].
const RECENT: usize = 50;

/// One tool as agents see it.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments (an object schema).
    pub input_schema: Value,
    /// Changes nothing (lets clients run it without asking).
    pub read_only: bool,
    /// May change meshes, the scene or files (clients should ask first).
    pub destructive: bool,
}

/// A tool's answer.
pub enum Reply {
    /// Structured data (a JSON object), also sent as text for clients that only read text.
    Json(Value),
    /// A PNG image (screenshots).
    Png(Vec<u8>),
    /// The tool ran and could not do what was asked: the agent sees the message.
    Error(String),
    /// Work to finish on a worker thread, off the app's UI thread.
    Later(Box<dyn FnOnce() -> Reply + Send>),
}

/// A tool call waiting for the app's answer.
pub struct Call {
    pub tool: String,
    /// The arguments (an object; `{}` when none were given).
    pub args: Value,
    reply: tokio::sync::oneshot::Sender<Reply>,
}

impl Call {
    pub fn answer(self, reply: Reply) {
        // The agent may have given up waiting: nobody to tell.
        let _ = self.reply.send(reply);
    }
}

/// How a call went, for the app's activity list.
#[derive(Clone, Debug)]
pub struct CallRecord {
    pub tool: String,
    pub at: SystemTime,
    pub duration: Duration,
    /// The error the agent was told, if it failed.
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// 0 picks a free port (tests).
    pub port: u16,
    /// Bearer token every request must carry.
    pub token: String,
    /// How long a call waits for the UI thread before the agent is told the app is busy.
    pub timeout: Duration,
    /// Shown to agents when they connect: what the server is and how to use it.
    pub instructions: String,
}

/// The running server. Dropping it stops it.
pub struct Server {
    addr: SocketAddr,
    calls: mpsc::Receiver<Call>,
    recent: Arc<Mutex<VecDeque<CallRecord>>>,
    cancel: CancellationToken,
}

impl Server {
    /// Listen on 127.0.0.1:`config.port`. `wake` is called after every new call (it should
    /// make the UI thread run soon, e.g. `egui::Context::request_repaint`).
    pub fn start(config: Config, tools: Vec<ToolSpec>, wake: impl Fn() + Send + Sync + 'static) -> anyhow::Result<Server> {
        let (call_tx, calls) = mpsc::channel();
        let recent = Arc::new(Mutex::new(VecDeque::new()));
        let cancel = CancellationToken::new();
        let handler = Handler {
            tools: Arc::new(tools.iter().map(tool).collect()),
            calls: call_tx,
            wake: Arc::new(wake),
            timeout: config.timeout,
            recent: recent.clone(),
            instructions: Arc::new(config.instructions.clone()),
        };
        let (ready_tx, ready) = mpsc::channel();
        let token = Arc::new(config.token.clone());
        let port = config.port;
        let ct = cancel.clone();
        std::thread::Builder::new().name("polymix-mcp".into()).spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(2).thread_name("polymix-mcp-worker").enable_all().build() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow::anyhow!("cannot start the server runtime: {e}")));
                    return;
                }
            };
            rt.block_on(async move {
                let listener = match tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await {
                    Ok(l) => l,
                    Err(e) => {
                        let what = if e.kind() == std::io::ErrorKind::AddrInUse { format!("port {port} is in use") } else { e.to_string() };
                        let _ = ready_tx.send(Err(anyhow::anyhow!("cannot listen on 127.0.0.1:{port}: {what}")));
                        return;
                    }
                };
                let _ = ready_tx.send(listener.local_addr().map_err(Into::into));
                // Stateless: every request stands alone (the 2026-07-28 protocol has no
                // sessions, and older clients are served the same way), answered as plain JSON.
                // Any `Origin` is refused; `Host` must be a loopback name (the default).
                let mcp_config = StreamableHttpServerConfig::default().with_legacy_session_mode(false).with_json_response(true).enforce_origin_validation().with_cancellation_token(ct.child_token());
                let service: StreamableHttpService<Handler, LocalSessionManager> = StreamableHttpService::new(move || Ok(handler.clone()), Default::default(), mcp_config);
                let app = axum::Router::new().nest_service(PATH, service).layer(axum::middleware::from_fn_with_state(token, require_token));
                tokio::select! {
                    r = axum::serve(listener, app) => {
                        if let Err(e) = r {
                            log::error!("agent server stopped: {e}");
                        }
                    }
                    _ = ct.cancelled() => {}
                }
            });
            // Calls still in flight are dropped: their agents are told the app went away.
            rt.shutdown_timeout(Duration::from_millis(500));
        })?;
        let addr = ready.recv().map_err(|_| anyhow::anyhow!("the server thread exited"))??;
        Ok(Server { addr, calls, recent, cancel })
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The endpoint URL clients connect to.
    pub fn url(&self) -> String {
        format!("http://{}{PATH}", self.addr)
    }

    /// The next call waiting for an answer.
    pub fn try_recv(&self) -> Option<Call> {
        self.calls.try_recv().ok()
    }

    /// Recent calls, newest first.
    pub fn recent(&self) -> Vec<CallRecord> {
        self.recent.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// A new random bearer token (64 hex digits).
pub fn new_token() -> anyhow::Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| anyhow::anyhow!("no random numbers: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn tool(spec: &ToolSpec) -> Tool {
    let schema = match &spec.input_schema {
        Value::Object(o) => o.clone(),
        _ => serde_json::Map::from_iter([("type".to_string(), Value::from("object"))]),
    };
    let annotations = if spec.read_only { ToolAnnotations::new().read_only(true).destructive(false).idempotent(true).open_world(false) } else { ToolAnnotations::new().read_only(false).destructive(spec.destructive).open_world(false) };
    Tool::new(spec.name.clone(), spec.description.clone(), schema).with_annotations(annotations)
}

async fn require_token(axum::extract::State(token): axum::extract::State<Arc<String>>, req: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    let given = req.headers().get(axum::http::header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).map(str::trim);
    if given.is_some_and(|g| same(g.as_bytes(), token.as_bytes())) {
        return next.run(req).await;
    }
    axum::response::Response::builder()
        .status(axum::http::StatusCode::UNAUTHORIZED)
        .header(axum::http::header::WWW_AUTHENTICATE, "Bearer realm=\"polymix\"")
        .body(axum::body::Body::from("Polymix needs its agent token: Authorization: Bearer <token> (see the Agents window in Polymix)"))
        .expect("valid response")
}

/// Constant-time comparison (the time taken does not reveal how much of a guess was right).
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

#[derive(Clone)]
struct Handler {
    tools: Arc<Vec<Tool>>,
    calls: mpsc::Sender<Call>,
    wake: Arc<dyn Fn() + Send + Sync>,
    timeout: Duration,
    recent: Arc<Mutex<VecDeque<CallRecord>>>,
    instructions: Arc<String>,
}

impl Handler {
    /// Hand the call to the UI thread and wait for its answer (and any work it left for later).
    async fn run(&self, tool: String, args: Value) -> Reply {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.calls.send(Call { tool, args, reply: tx }).is_err() {
            return Reply::Error("Polymix is closing.".into());
        }
        (self.wake)();
        let reply = match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(r)) => r,
            Ok(Err(_)) => Reply::Error("Polymix dropped the call (the workspace may have stopped).".into()),
            Err(_) => return Reply::Error(format!("Polymix did not answer within {} s: a dialog may be open, or the window may be minimized.", self.timeout.as_secs())),
        };
        let mut reply = reply;
        // Work left for later may itself leave more (e.g. a screenshot's encoding).
        while let Reply::Later(work) = reply {
            reply = tokio::task::spawn_blocking(work).await.unwrap_or_else(|e| Reply::Error(format!("internal error: {e}")));
        }
        reply
    }

    fn record(&self, tool: &str, start: Instant, error: Option<String>) {
        let mut recent = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        recent.push_front(CallRecord { tool: tool.to_string(), at: SystemTime::now(), duration: start.elapsed(), error });
        recent.truncate(RECENT);
    }
}

impl ServerHandler for Handler {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(Implementation::new("polymix", env!("CARGO_PKG_VERSION")).with_title("Polymix")).with_instructions(self.instructions.as_str())
    }

    async fn list_tools(&self, _request: Option<PaginatedRequestParams>, _context: RequestContext<RoleServer>) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(self.tools.to_vec()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.tools.iter().find(|t| t.name == name).cloned()
    }

    async fn call_tool(&self, request: CallToolRequestParams, _context: RequestContext<RoleServer>) -> Result<CallToolResponse, ErrorData> {
        let name = request.name.to_string();
        if !self.tools.iter().any(|t| t.name == name) {
            return Err(ErrorData::invalid_params(format!("unknown tool {name}"), None));
        }
        let start = Instant::now();
        let args = request.arguments.map(Value::Object).unwrap_or_else(|| Value::Object(Default::default()));
        let result = match self.run(name.clone(), args).await {
            Reply::Json(v) => {
                let text = serde_json::to_string_pretty(&v).unwrap_or_default();
                self.record(&name, start, None);
                let mut r = CallToolResult::success(vec![ContentBlock::text(text)]);
                r.structured_content = Some(v);
                r
            }
            Reply::Png(png) => {
                self.record(&name, start, None);
                CallToolResult::success(vec![ContentBlock::image(base64::engine::general_purpose::STANDARD.encode(png), "image/png")])
            }
            Reply::Error(msg) => {
                self.record(&name, start, Some(msg.clone()));
                CallToolResult::error(vec![ContentBlock::text(msg)])
            }
            Reply::Later(_) => unreachable!("run() finishes work left for later"),
        };
        Ok(result.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_compare_whole() {
        assert!(same(b"abc", b"abc"));
        assert!(!same(b"abc", b"abd"));
        assert!(!same(b"abc", b"abcd"));
        let t = new_token().unwrap();
        assert_eq!(t.len(), 64);
        assert_ne!(t, new_token().unwrap());
    }
}

//! The server over real HTTP: the token and origin checks, the handshake, listing tools, and
//! calls answered by a stand-in for the app's UI thread.

use polymix_mcp::{Config, Reply, Server, ToolSpec};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::time::Duration;

const TOKEN: &str = "test-token";

fn start() -> Server {
    let tools = ["echo", "later", "fails", "ignored"].map(|n| ToolSpec { name: n.into(), description: format!("{n} tool"), input_schema: json!({"type": "object", "properties": {"x": {"type": "integer"}}}), read_only: true, destructive: false });
    let config = Config { port: 0, token: TOKEN.into(), timeout: Duration::from_millis(500), instructions: "test server".into() };
    Server::start(config, tools.to_vec(), || {}).expect("server starts")
}

/// Answer calls the way the app does, until the server is dropped.
fn serve(server: &Server, calls: usize) {
    let mut answered = 0;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while answered < calls && std::time::Instant::now() < deadline {
        let Some(call) = server.try_recv() else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };
        answered += 1;
        let x = call.args.get("x").and_then(Value::as_i64).unwrap_or(0);
        match call.tool.as_str() {
            "echo" => call.answer(Reply::Json(json!({"x": x}))),
            "later" => call.answer(Reply::Later(Box::new(move || Reply::Json(json!({"doubled": x * 2}))))),
            "fails" => call.answer(Reply::Error("nothing selected".into())),
            // Never answered: the agent hears about the timeout.
            _ => std::mem::forget(call),
        }
    }
}

/// POST a JSON-RPC message; returns the status and the body.
fn post(addr: SocketAddr, body: &Value, headers: &[(&str, &str)]) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let body = body.to_string();
    let mut req = format!("POST /mcp HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n", addr, body.len());
    for (k, v) in headers {
        req += &format!("{k}: {v}\r\n");
    }
    req += "\r\n";
    req += &body;
    s.write_all(req.as_bytes()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let status = out.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let body = out.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    (status, dechunk(&body))
}

/// Undo `Transfer-Encoding: chunked` (if the body is chunked).
fn dechunk(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    loop {
        let Some((size, after)) = rest.split_once("\r\n") else { return body.to_string() };
        let Ok(n) = usize::from_str_radix(size.trim(), 16) else { return body.to_string() };
        if n == 0 {
            return out;
        }
        if after.len() < n {
            return body.to_string();
        }
        out += &after[..n];
        rest = after[n..].trim_start_matches("\r\n");
    }
}

fn auth() -> (&'static str, String) {
    ("Authorization", format!("Bearer {TOKEN}"))
}

fn rpc(addr: SocketAddr, id: u64, method: &str, params: Value) -> Value {
    let (k, v) = auth();
    let (status, body) = post(addr, &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}), &[(k, &v), ("MCP-Protocol-Version", "2025-11-25")]);
    assert_eq!(status, 200, "{method}: {body}");
    serde_json::from_str(&body).unwrap_or_else(|e| panic!("{method}: {e}: {body}"))
}

fn call(addr: SocketAddr, id: u64, tool: &str, args: Value) -> Value {
    rpc(addr, id, "tools/call", json!({"name": tool, "arguments": args}))["result"].clone()
}

#[test]
fn needs_the_token() {
    let server = start();
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}});
    assert_eq!(post(server.addr(), &init, &[]).0, 401);
    assert_eq!(post(server.addr(), &init, &[("Authorization", "Bearer wrong")]).0, 401);
    let (k, v) = auth();
    assert_eq!(post(server.addr(), &init, &[(k, &v)]).0, 200);
}

#[test]
fn refuses_web_pages() {
    let server = start();
    let (k, v) = auth();
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}});
    let (status, _) = post(server.addr(), &init, &[(k, &v), ("Origin", "https://evil.example")]);
    assert_eq!(status, 403);
}

#[test]
fn handshake_list_and_calls() {
    let server = start();
    let init = rpc(server.addr(), 1, "initialize", json!({"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}));
    assert_eq!(init["result"]["serverInfo"]["name"], "polymix");
    assert_eq!(init["result"]["instructions"], "test server");

    let tools = rpc(server.addr(), 2, "tools/list", json!({}));
    let echo = tools["result"]["tools"].as_array().unwrap().iter().find(|t| t["name"] == "echo").unwrap().clone();
    assert_eq!(echo["annotations"]["readOnlyHint"], true);
    assert_eq!(echo["inputSchema"]["properties"]["x"]["type"], "integer");

    // The agent on another thread; this one answers, as the app's UI thread does.
    let addr = server.addr();
    let agent = std::thread::spawn(move || {
        let r = call(addr, 3, "echo", json!({"x": 7}));
        assert_eq!(r["structuredContent"], json!({"x": 7}));
        assert!(r["content"][0]["text"].as_str().unwrap().contains("7"));
        assert_eq!(call(addr, 4, "later", json!({"x": 21}))["structuredContent"], json!({"doubled": 42}));
        let r = call(addr, 5, "fails", json!({}));
        assert_eq!(r["isError"], true);
        assert_eq!(r["content"][0]["text"], "nothing selected");
        let r = call(addr, 6, "ignored", json!({}));
        assert_eq!(r["isError"], true);
        assert!(r["content"][0]["text"].as_str().unwrap().contains("did not answer"), "{r}");
    });
    serve(&server, 4);
    agent.join().unwrap();

    let names: Vec<String> = server.recent().iter().map(|c| c.tool.clone()).collect();
    assert_eq!(names, ["ignored", "fails", "later", "echo"]);
    assert!(server.recent()[0].error.is_some());
}

#[test]
fn unknown_tools_are_protocol_errors() {
    let server = start();
    let r = rpc(server.addr(), 1, "tools/call", json!({"name": "nope", "arguments": {}}));
    assert!(r["error"]["message"].as_str().unwrap().contains("unknown tool"), "{r}");
}

#[test]
fn stateless_protocol() {
    // 2026-07-28: no handshake; every request carries its protocol version.
    let server = start();
    let addr = server.addr();
    let agent = std::thread::spawn(move || {
        let (k, v) = auth();
        let meta = json!({"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"}, "io.modelcontextprotocol/clientCapabilities": {}});
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"x": 3}, "_meta": meta}});
        let (status, body) = post(addr, &body, &[(k, &v), ("MCP-Protocol-Version", "2026-07-28"), ("Mcp-Method", "tools/call"), ("Mcp-Name", "echo")]);
        assert_eq!(status, 200, "{body}");
        let r: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(r["result"]["structuredContent"], json!({"x": 3}), "{r}");
    });
    serve(&server, 1);
    agent.join().unwrap();
}

#[test]
fn port_in_use_is_reported() {
    let a = start();
    let config = Config { port: a.addr().port(), token: TOKEN.into(), timeout: Duration::from_secs(1), instructions: String::new() };
    let err = Server::start(config, vec![], || {}).err().expect("second server on the same port fails");
    assert!(err.to_string().contains("in use"), "{err}");
}

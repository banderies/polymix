//! Background computations with progress reporting and cancellation.
//!
//! Each job runs its work in a thread pool of its own while it runs. Its parallel loops then
//! queue there, not in rayon's global pool, which the UI thread uses for its own short
//! parallel work (mesh uploads, BVH builds, slice resampling): a frame never waits for a
//! job's loop. Nor does a job wait for another's. In one pool shared by all jobs, a job that
//! starts while another's long loop runs would only get a thread once that loop ran out of
//! tasks to hand out, and a thread waiting inside a loop could take up another job's whole
//! work on top of its own. A job's pool is kept for the next job when it ends (a couple at
//! most), so short jobs do not start and stop threads each time.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// Progress callback handed to geometry operations: report a 0..1 fraction, returns
/// `false` when the user cancelled.
pub type ProgressFn<'a> = &'a dyn Fn(f32) -> bool;

/// The start of the job pools' thread names. The crash guard logs panics on threads named
/// polymix-job…
const POOL_THREAD: &str = "polymix-job-pool-";

/// Threads in a job's pool: one per core but one (the UI thread's).
fn pool_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(1).max(1)))
}

/// The job pools: those no job is using wait here for the next jobs.
struct Pools {
    idle: Mutex<Vec<rayon::ThreadPool>>,
    /// How many idle pools are kept; more are dropped (their threads end on their own).
    keep: usize,
}

impl Pools {
    const fn new(keep: usize) -> Self {
        Self { idle: Mutex::new(Vec::new()), keep }
    }

    /// Run `f` in a pool nobody else uses meanwhile: an idle one, or one started for it (well
    /// under a millisecond). A pool whose work panicked is dropped rather than kept.
    fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        let idle = self.idle.lock().unwrap_or_else(PoisonError::into_inner).pop();
        let pool = match idle {
            Some(pool) => pool,
            None => match rayon::ThreadPoolBuilder::new().num_threads(pool_threads()).thread_name(|i| format!("{POOL_THREAD}{i}")).build() {
                Ok(pool) => pool,
                Err(e) => {
                    log::warn!("cannot start a job's threads: {e}");
                    return f();
                }
            },
        };
        let r = pool.install(f);
        let mut idle = self.idle.lock().unwrap_or_else(PoisonError::into_inner);
        if idle.len() < self.keep {
            idle.push(pool);
        }
        r
    }
}

/// Two kept pools cover a job finishing while the next starts, and a preview's slice loads
/// queued one after another, without many idle threads.
static POOLS: Pools = Pools::new(2);

/// Run `f` in a thread pool of its own, so that its parallel loops stay out of the UI
/// thread's way and out of other jobs'. For background work started outside [`Job::spawn`]
/// (e.g. a worker thread of its own); called from a job's work, `f` just runs in that job's
/// pool. Blocks until `f` returns, and passes on its panic. If no pool can be started, `f`
/// runs on the calling thread as it did before pools.
pub fn in_job_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let in_a_job_pool = rayon::current_thread_index().is_some() && std::thread::current().name().is_some_and(|n| n.starts_with(POOL_THREAD));
    if in_a_job_pool {
        return f();
    }
    POOLS.run(f)
}

pub struct Job<T> {
    pub label: String,
    progress: Arc<AtomicU32>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<anyhow::Result<T>>,
}

impl<T: Send + 'static> Job<T> {
    pub fn spawn(label: impl Into<String>, ctx: egui::Context, f: impl FnOnce(ProgressFn) -> anyhow::Result<T> + Send + 'static) -> Self {
        let progress = Arc::new(AtomicU32::new(0f32.to_bits()));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let (p2, c2) = (progress.clone(), cancel.clone());
        std::thread::Builder::new()
            .name("polymix-job".into())
            .spawn(move || {
                let cb = |f: f32| {
                    p2.store(f.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
                    !c2.load(Ordering::Relaxed)
                };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| in_job_pool(|| f(&cb))))
                    .unwrap_or_else(|p| {
                        let msg = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "operation panicked".into());
                        Err(anyhow::anyhow!("internal error: {msg}"))
                    });
                let _ = tx.send(r);
                ctx.request_repaint();
            })
            .expect("spawn job thread");
        Self { label: label.into(), progress, cancel, rx }
    }

    pub fn progress(&self) -> f32 {
        f32::from_bits(self.progress.load(Ordering::Relaxed))
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// `Some(result)` once finished.
    pub fn poll(&self) -> Option<anyhow::Result<T>> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(anyhow::anyhow!("job thread exited"))),
        }
    }
}

impl<T> Drop for Job<T> {
    /// Abandoned jobs (e.g. a tool closed mid-computation) are asked to stop early.
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::time::{Duration, Instant};

    /// Far longer than any of these jobs takes, even on a busy machine.
    const LIMIT: Duration = Duration::from_secs(20);

    /// The job's result, or `None` if it is still running after `limit`.
    fn wait<T: Send + 'static>(job: &Job<T>, limit: Duration) -> Option<anyhow::Result<T>> {
        let start = Instant::now();
        loop {
            if let Some(r) = job.poll() {
                return Some(r);
            }
            if start.elapsed() > limit {
                return None;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn thread_name() -> Option<String> {
        std::thread::current().name().map(str::to_string)
    }

    #[test]
    fn a_job_runs_in_a_pool_of_its_own() {
        let job = Job::spawn("probe", egui::Context::default(), |progress| {
            assert!(progress(0.5));
            let sum: u64 = (0..10_000u64).into_par_iter().sum();
            Ok((thread_name(), rayon::current_thread_index().is_some(), sum, in_job_pool(thread_name)))
        });
        let (name, in_rayon, sum, nested) = wait(&job, LIMIT).expect("finished").expect("no error");
        assert!(name.as_deref().is_some_and(|n| n.starts_with("polymix-job-pool-")), "{name:?}");
        assert!(in_rayon);
        assert_eq!(sum, 49_995_000);
        assert_eq!(nested, name, "in_job_pool in a job stays in the job's pool");
        assert_eq!(job.progress(), 0.5);

        // Outside a job.
        let (name, sum) = in_job_pool(|| (thread_name(), (0..100u32).into_par_iter().map(|i| i * 2).sum::<u32>()));
        assert!(name.as_deref().is_some_and(|n| n.starts_with("polymix-job-pool-")), "{name:?}");
        assert_eq!(sum, 9_900);
        assert!(std::panic::catch_unwind(|| in_job_pool(|| -> u32 { panic!("passed on") })).is_err());
    }

    #[test]
    fn idle_pools_are_used_again() {
        let pools = Pools::new(1);
        let threads = || rayon::broadcast(|_| std::thread::current().id()).into_iter().collect::<std::collections::HashSet<_>>();
        let first = pools.run(threads);
        assert_eq!(first.len(), pool_threads());
        assert_eq!(pools.run(threads), first, "the same pool's threads");
        // Two at once: the second gets a pool of its own, and only one is kept afterwards.
        let ((started_tx, started), (go, go_rx)) = (channel(), channel::<()>());
        let (other, kept) = std::thread::scope(|s| {
            let pools = &pools;
            let busy = s.spawn(move || {
                pools.run(move || {
                    let t = threads();
                    started_tx.send(()).expect("test running");
                    go_rx.recv().expect("test running");
                    t
                })
            });
            started.recv_timeout(LIMIT).expect("the first run started");
            let other = pools.run(threads);
            go.send(()).expect("first run waiting");
            (other, busy.join().expect("no panic"))
        });
        assert_eq!(kept, first, "the idle pool went to the first run");
        assert!(other.is_disjoint(&first), "the other run got new threads");
        assert_eq!(pools.idle.lock().unwrap().len(), 1);
        // A pool whose work panicked is not kept.
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pools.run(|| -> u32 { panic!("dropped") }))).is_err());
        assert!(pools.idle.lock().unwrap().is_empty());
        assert!(pools.run(threads).is_disjoint(&first));
        // None kept: a new pool each time.
        let none = Pools::new(0);
        assert!(none.run(threads).is_disjoint(&none.run(threads)));
    }

    #[test]
    fn panics_come_back_as_errors() {
        let ctx = egui::Context::default();
        let body = Job::spawn("panics", ctx.clone(), |_| -> anyhow::Result<()> { panic!("in the body") });
        let in_loop = Job::spawn("panics in a loop", ctx, |_| {
            (0..1000u32).into_par_iter().for_each(|i| assert!(i != 777, "in a loop at {i}"));
            Ok(())
        });
        let e = wait(&body, LIMIT).expect("finished").expect_err("panicked").to_string();
        assert_eq!(e, "internal error: in the body");
        let e = wait(&in_loop, LIMIT).expect("finished").expect_err("panicked").to_string();
        assert_eq!(e, "internal error: in a loop at 777");
    }

    #[test]
    fn a_job_does_not_wait_for_another_jobs_loop() {
        let ctx = egui::Context::default();
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        // Every thread of its pool stays inside this loop until told to stop.
        let long = Job::spawn("long", ctx.clone(), move |_| {
            (0..256u32).into_par_iter().for_each(|_| {
                while !s.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            Ok(())
        });
        std::thread::sleep(Duration::from_millis(50));
        let short = Job::spawn("short", ctx, |_| Ok((0..1000u64).into_par_iter().sum::<u64>()));
        let short = wait(&short, LIMIT);
        // Nor do the UI thread's loops, in rayon's global pool (on a thread of their own here
        // so that a failure cannot hang the test).
        let (tx, rx) = channel();
        std::thread::spawn(move || tx.send((0..1000u64).into_par_iter().sum::<u64>()));
        let ui = rx.recv_timeout(LIMIT);
        let long_ran_on = long.poll().is_none();
        stop.store(true, Ordering::Relaxed);
        assert!(matches!(short, Some(Ok(499_500))), "the short job finished");
        assert_eq!(ui, Ok(499_500), "the global pool's loop finished");
        assert!(long_ran_on, "while the long one was still running");
        assert!(wait(&long, LIMIT).is_some_and(|r| r.is_ok()));
    }
}

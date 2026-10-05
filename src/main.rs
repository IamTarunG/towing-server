mod api;
mod backend;
mod config;
mod metrics;
mod scheduler;
mod state;
mod types;

use axum::{
    Router,
    routing::{get, post},
};

use backend::candle::{CandleBackend, ComputeDevice};

use metrics::Metrics;

use state::AppState;

use types::QueuedRequest;

use std::sync::{Arc, mpsc};

#[tokio::main]
async fn main() {
    /*
     * Logging
     */
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    print_banner();

    // ========================================================
    // COMPUTE DEVICE
    // ========================================================

    let compute_device = ComputeDevice::from_env().expect("invalid TOWING_DEVICE");

    let device_name = compute_device.name();

    println!("Requested compute device: {}", device_name);

    println!();

    // ========================================================
    // MODEL BACKEND
    // ========================================================

    /*
     * Exactly one model instance is loaded.
     *
     * CPU:
     *
     * TOWING_DEVICE=cpu
     *
     * CUDA:
     *
     * TOWING_DEVICE=cuda
     *
     * CUDA binary must be compiled using:
     *
     * cargo build --release --features cuda
     */
    let backend =
        CandleBackend::new(compute_device).expect("failed to initialize inference backend");

    // ========================================================
    // METRICS
    // ========================================================

    let metrics = Arc::new(Metrics::new());

    // ========================================================
    // ADMISSION QUEUE
    // ========================================================

    /*
     * Bounded synchronous queue.
     *
     * The Axum layer uses try_send(),
     * therefore HTTP runtime workers never
     * block waiting for queue capacity.
     */
    let (queue_tx, queue_rx) = mpsc::sync_channel::<QueuedRequest>(config::QUEUE_CAPACITY);

    // ========================================================
    // APPLICATION STATE
    // ========================================================

    let state = AppState {
        queue_tx,

        metrics: metrics.clone(),

        device: device_name,
    };

    // ========================================================
    // INFERENCE WORKER
    // ========================================================

    /*
     * Candle model execution is synchronous
     * and compute-heavy.
     *
     * It therefore runs on a dedicated OS
     * thread rather than a Tokio worker.
     */
    let worker_metrics = metrics.clone();

    std::thread::Builder::new()
        .name("towing-inference".to_string())
        .spawn(move || {
            scheduler::run_inference_worker(queue_rx, worker_metrics, backend);
        })
        .expect("failed to start inference worker");

    // ========================================================
    // HTTP ROUTES
    // ========================================================

    let app = Router::new()
        /*
         * Health information.
         */
        .route("/health", get(api::health))
        /*
         * Internal engine metrics.
         */
        .route("/metrics", get(api::metrics))
        /*
         * Minimal serving-plane endpoint.
         *
         * Used to measure HTTP/runtime
         * overhead independently from
         * model inference.
         */
        .route("/bench", get(api::bench))
        /*
         * Standard JSON inference.
         */
        .route("/v1/inference", post(api::inference))
        /*
         * Progressive text generation.
         */
        .route("/v1/inference/stream", post(api::inference_stream))
        .with_state(state);

    // ========================================================
    // HTTP LISTENER
    // ========================================================

    let listener = tokio::net::TcpListener::bind(config::HOST)
        .await
        .expect("failed to bind server address");

    println!("Server ready.");

    println!("http://{}", config::HOST);

    println!();

    println!("POST /v1/inference");

    println!("POST /v1/inference/stream");

    println!("GET  /health");

    println!("GET  /metrics");

    println!("GET  /bench");

    println!();

    println!("Queue capacity: {}", config::QUEUE_CAPACITY);

    println!("Inference worker: dedicated OS thread");

    println!("Compute device: {}", device_name);

    println!("Model execution: sequential");

    println!();

    // ========================================================
    // START SERVER
    // ========================================================

    axum::serve(listener, app)
        .await
        .expect("HTTP server failed");
}

// ============================================================
// STARTUP BANNER
// ============================================================

fn print_banner() {
    println!();

    println!("======================================");

    println!("          TOWING SERVER");

    println!("       LLM Inference Engine");

    println!("             v{}", env!("CARGO_PKG_VERSION"));

    println!("======================================");

    println!();
}

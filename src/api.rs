use crate::{
    config,
    state::AppState,
    types::{GenerationParameters, InferenceRequest, QueuedRequest, ResponseChannel},
};

use axum::{
    Json,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

use std::{convert::Infallible, sync::mpsc::TrySendError, time::Instant};

use tokio::sync::{mpsc, oneshot};

use tokio_stream::{StreamExt, wrappers::ReceiverStream};

// ============================================================
// STANDARD INFERENCE
// ============================================================

pub async fn inference(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> Response {
    state.metrics.inc_received();

    if let Err(error) = validate_request(&request) {
        return json_error(StatusCode::BAD_REQUEST, &error);
    }

    let parameters = GenerationParameters::from(&request);

    let (response_tx, response_rx) = oneshot::channel();

    let queued = QueuedRequest {
        request_id: request.request_id,
        prompt: request.prompt,
        parameters,
        queued_at: Instant::now(),
        response: ResponseChannel::Standard(response_tx),
    };

    if let Err(response) = enqueue(&state, queued) {
        return response;
    }

    match response_rx.await {
        Ok(Ok(response)) => (StatusCode::OK, Json(response)).into_response(),

        Ok(Err(error)) => json_error(StatusCode::INTERNAL_SERVER_ERROR, &error),

        Err(_) => json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "inference worker unavailable",
        ),
    }
}

// ============================================================
// STREAMING INFERENCE
// ============================================================

pub async fn inference_stream(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> Response {
    state.metrics.inc_received();

    if let Err(error) = validate_request(&request) {
        return json_error(StatusCode::BAD_REQUEST, &error);
    }

    let parameters = GenerationParameters::from(&request);

    let (stream_tx, stream_rx) = mpsc::channel::<String>(config::STREAM_CAPACITY);

    let queued = QueuedRequest {
        request_id: request.request_id,
        prompt: request.prompt,
        parameters,
        queued_at: Instant::now(),
        response: ResponseChannel::Stream(stream_tx),
    };

    if let Err(response) = enqueue(&state, queued) {
        return response;
    }

    let stream = ReceiverStream::new(stream_rx).map(|chunk| Ok::<_, Infallible>(chunk));

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-cache, no-transform")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .expect("failed to construct streaming response")
}

// ============================================================
// ADMISSION CONTROL
// ============================================================

fn enqueue(state: &AppState, request: QueuedRequest) -> Result<(), Response> {
    match state.queue_tx.try_send(request) {
        Ok(_) => {
            state.metrics.inc_accepted();
            Ok(())
        }

        Err(TrySendError::Full(_)) => {
            state.metrics.inc_rejected();

            Err(json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "inference queue full",
            ))
        }

        Err(TrySendError::Disconnected(_)) => {
            state.metrics.inc_rejected();

            Err(json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "inference worker unavailable",
            ))
        }
    }
}

// ============================================================
// VALIDATION
// ============================================================

fn validate_request(request: &InferenceRequest) -> Result<(), String> {
    if request.prompt.trim().is_empty() {
        return Err("prompt cannot be empty".to_string());
    }

    if request.prompt.len() > config::MAX_PROMPT_CHARS {
        return Err(format!(
            "prompt exceeds {} characters",
            config::MAX_PROMPT_CHARS
        ));
    }

    if request.max_tokens == 0 || request.max_tokens > config::MAX_OUTPUT_TOKENS {
        return Err(format!(
            "max_tokens must be between 1 and {}",
            config::MAX_OUTPUT_TOKENS
        ));
    }

    if !request.temperature.is_finite() || request.temperature < 0.0 || request.temperature > 5.0 {
        return Err("temperature must be between 0 and 5".to_string());
    }

    if let Some(top_p) = request.top_p {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err("top_p must be > 0 and <= 1".to_string());
        }
    }

    if let Some(top_k) = request.top_k {
        if top_k == 0 {
            return Err("top_k must be greater than 0".to_string());
        }
    }

    if !request.repeat_penalty.is_finite() || request.repeat_penalty <= 0.0 {
        return Err("repeat_penalty must be greater than 0".to_string());
    }

    Ok(())
}

// ============================================================
// HEALTH
// ============================================================

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "towing-server",
        "version": env!("CARGO_PKG_VERSION"),
        "model": config::MODEL_ID,
        "device": state.device
    }))
}

// ============================================================
// METRICS
// ============================================================

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.metrics.snapshot())
}

// ============================================================
// SERVING-PLANE BENCHMARK
// ============================================================

pub async fn bench() -> &'static str {
    "OK"
}

// ============================================================
// ERROR RESPONSE
// ============================================================

fn json_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": message
        })),
    )
        .into_response()
}

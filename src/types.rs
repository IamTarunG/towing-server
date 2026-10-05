use crate::config;

use serde::{Deserialize, Serialize};

use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub request_id: u64,

    pub prompt: String,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    #[serde(default)]
    pub temperature: f64,

    pub top_p: Option<f64>,

    pub top_k: Option<usize>,

    #[serde(default = "default_seed")]
    pub seed: u64,

    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,

    #[serde(default = "default_repeat_last_n")]
    pub repeat_last_n: usize,
}

fn default_max_tokens() -> u32 {
    config::DEFAULT_MAX_TOKENS
}

fn default_seed() -> u64 {
    config::DEFAULT_SEED
}

fn default_repeat_penalty() -> f32 {
    config::DEFAULT_REPEAT_PENALTY
}

fn default_repeat_last_n() -> usize {
    config::DEFAULT_REPEAT_LAST_N
}

#[derive(Debug, Clone)]
pub struct GenerationParameters {
    pub max_tokens: usize,

    pub temperature: f64,

    pub top_p: Option<f64>,

    pub top_k: Option<usize>,

    pub seed: u64,

    pub repeat_penalty: f32,

    pub repeat_last_n: usize,
}

impl From<&InferenceRequest> for GenerationParameters {
    fn from(request: &InferenceRequest) -> Self {
        Self {
            max_tokens: request.max_tokens as usize,

            temperature: request.temperature,

            top_p: request.top_p,

            top_k: request.top_k,

            seed: request.seed,

            repeat_penalty: request.repeat_penalty,

            repeat_last_n: request.repeat_last_n,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,

    pub generated_tokens: usize,

    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct Timing {
    pub queue_ms: f64,

    pub model_ttft_ms: f64,

    pub end_to_end_ttft_ms: f64,

    pub tpot_ms: f64,

    pub tokens_per_second: f64,

    pub inference_ms: f64,

    pub total_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub request_id: u64,

    pub text: String,

    pub finish_reason: String,

    pub usage: Usage,

    pub timing: Timing,
}

#[derive(Debug)]
pub enum ResponseChannel {
    Standard(oneshot::Sender<Result<InferenceResponse, String>>),

    Stream(mpsc::Sender<String>),
}

#[derive(Debug)]
pub struct QueuedRequest {
    pub request_id: u64,

    pub prompt: String,

    pub parameters: GenerationParameters,

    pub queued_at: Instant,

    pub response: ResponseChannel,
}

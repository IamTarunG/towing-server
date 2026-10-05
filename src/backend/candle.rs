use crate::{config, types::GenerationParameters};

use anyhow::{Context, Result};

use candle_core::{DType, Device, Tensor};

use candle_nn::VarBuilder;

use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks},
    utils::apply_repeat_penalty,
};

use hf_hub::{Repo, RepoType, api::sync::Api};

use std::time::{Duration, Instant};

use tokenizers::Tokenizer;

// ============================================================
// COMPUTE DEVICE
// ============================================================

#[derive(Debug, Clone, Copy)]
pub enum ComputeDevice {
    Cpu,
    Cuda,
}

impl ComputeDevice {
    pub fn from_env() -> Result<Self> {
        let value = std::env::var("TOWING_DEVICE").unwrap_or_else(|_| "cpu".to_string());

        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),

            "cuda" | "gpu" => Ok(Self::Cuda),

            other => {
                anyhow::bail!(
                    "unsupported TOWING_DEVICE '{}'. \
                     Use 'cpu' or 'cuda'.",
                    other
                )
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

fn create_device(compute_device: ComputeDevice) -> Result<Device> {
    match compute_device {
        ComputeDevice::Cpu => Ok(Device::Cpu),

        ComputeDevice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let device = Device::new_cuda(0).context("failed to initialize CUDA device 0")?;

                Ok(device)
            }

            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!(
                    "CUDA was requested, but this binary \
                     was compiled without CUDA support. \
                     Rebuild using: \
                     cargo build --release --features cuda"
                )
            }
        }
    }
}

// ============================================================
// GENERATION TYPES
// ============================================================

pub struct GenerationOutput {
    pub text: String,

    pub prompt_tokens: usize,

    pub generated_tokens: usize,

    pub model_ttft: Duration,

    pub decode_time: Duration,

    pub inference_time: Duration,

    pub finish_reason: String,
}

pub enum StreamControl {
    Continue,
    Cancel,
}

// ============================================================
// CANDLE BACKEND
// ============================================================

pub struct CandleBackend {
    model: Llama,

    tokenizer: Tokenizer,

    config: Config,

    device: Device,

    dtype: DType,

    eos_tokens: Option<LlamaEosToks>,
}

impl CandleBackend {
    pub fn new(compute_device: ComputeDevice) -> Result<Self> {
        println!("Initializing Candle inference backend...");

        println!("Model: {}", config::MODEL_ID);

        println!("Device: {}", compute_device.name());

        /*
         * First CUDA benchmark intentionally
         * remains F32.
         *
         * This allows a controlled comparison:
         *
         * CPU F32
         * vs
         * CUDA F32
         *
         * F16/BF16 can be benchmarked after
         * CUDA correctness is established.
         */
        let dtype = DType::F32;

        let device =
            create_device(compute_device).context("failed to initialize compute device")?;

        let api = Api::new().context("failed to initialize Hugging Face Hub")?;

        let repo = api.repo(Repo::new(config::MODEL_ID.to_string(), RepoType::Model));

        println!("Locating model files...");

        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("failed to obtain tokenizer.json")?;

        let config_path = repo
            .get("config.json")
            .context("failed to obtain config.json")?;

        let model_path = repo
            .get("model.safetensors")
            .context("failed to obtain model.safetensors")?;

        println!("Loading tokenizer...");

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| anyhow::anyhow!("failed to load tokenizer: {}", error))?;

        println!("Loading model configuration...");

        let config_bytes = std::fs::read(config_path).context("failed to read model config")?;

        let llama_config: LlamaConfig =
            serde_json::from_slice(&config_bytes).context("failed to parse Llama configuration")?;

        let config = llama_config.into_config(false);

        /*
         * Prefer the EOS token supplied by
         * the model configuration.
         *
         * Fall back to common SmolLM chat
         * special tokens when necessary.
         */
        let eos_tokens = config
            .eos_token_id
            .clone()
            .or_else(|| {
                tokenizer
                    .token_to_id("<|im_end|>")
                    .map(LlamaEosToks::Single)
            })
            .or_else(|| {
                tokenizer
                    .token_to_id("<|endoftext|>")
                    .map(LlamaEosToks::Single)
            });

        println!("Loading model weights...");

        /*
         * The safetensor file is memory mapped.
         *
         * VarBuilder moves tensors to the
         * selected Candle device.
         */
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };

        let model = Llama::load(vb, &config).context("failed to construct Llama model")?;

        println!("Model loaded successfully.");

        println!("Precision: F32");

        println!();

        Ok(Self {
            model,

            tokenizer,

            config,

            device,

            dtype,

            eos_tokens,
        })
    }

    // ========================================================
    // STANDARD GENERATION
    // ========================================================

    pub fn generate(
        &mut self,

        prompt: &str,

        parameters: &GenerationParameters,
    ) -> Result<GenerationOutput> {
        self.generate_internal(prompt, parameters, |_| StreamControl::Continue)
    }

    // ========================================================
    // STREAMING GENERATION
    // ========================================================

    pub fn generate_stream<F>(
        &mut self,

        prompt: &str,

        parameters: &GenerationParameters,

        callback: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(String) -> StreamControl,
    {
        self.generate_internal(prompt, parameters, callback)
    }

    // ========================================================
    // GENERATION ENGINE
    // ========================================================

    fn generate_internal<F>(
        &mut self,

        prompt: &str,

        parameters: &GenerationParameters,

        mut callback: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(String) -> StreamControl,
    {
        let inference_start = Instant::now();

        /*
         * SmolLM2 instruct prompt.
         */
        let formatted_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );

        let encoding = self
            .tokenizer
            .encode(formatted_prompt, true)
            .map_err(|error| anyhow::anyhow!("tokenization failed: {}", error))?;

        let mut tokens = encoding.get_ids().to_vec();

        let prompt_tokens = tokens.len();

        if prompt_tokens == 0 {
            anyhow::bail!("prompt produced zero tokens");
        }

        /*
         * Each independent request receives
         * an independent KV cache.
         */
        let mut cache = Cache::new(true, self.dtype, &self.config, &self.device)
            .context("failed to create KV cache")?;

        let sampling = build_sampling(parameters);

        let mut logits_processor = LogitsProcessor::from_sampling(parameters.seed, sampling);

        let mut generated_ids: Vec<u32> = Vec::with_capacity(parameters.max_tokens);

        let mut previous_text = String::new();

        /*
         * Position of the next input token
         * in the KV cache.
         */
        let mut index_pos = 0usize;

        let mut first_token_time: Option<Duration> = None;

        let mut decode_start: Option<Instant> = None;

        let mut finish_reason = "length".to_string();

        for generation_index in 0..parameters.max_tokens {
            /*
             * PREFILL:
             *
             * First iteration sends the
             * entire prompt.
             *
             * DECODE:
             *
             * Subsequent iterations send
             * only the newest token because
             * previous tokens are already
             * represented in the KV cache.
             */
            let (context_size, context_index) = if cache.use_kv_cache && generation_index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let context = &tokens[tokens.len().saturating_sub(context_size)..];

            let input = Tensor::new(context, &self.device)?.unsqueeze(0)?;

            let logits = self
                .model
                .forward(&input, context_index, &mut cache)
                .context("model forward pass failed")?;

            /*
             * Candle Llama forward returns
             * logits for the final position.
             */
            let mut logits = logits.squeeze(0)?;

            /*
             * Optional repetition penalty.
             */
            if parameters.repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(parameters.repeat_last_n);

                logits =
                    apply_repeat_penalty(&logits, parameters.repeat_penalty, &tokens[start_at..])?;
            }

            /*
             * Advance KV position by the
             * number of tokens consumed by
             * this forward pass.
             */
            index_pos += context.len();

            let next_token = logits_processor
                .sample(&logits)
                .context("token sampling failed")?;

            /*
             * TTFT is measured after prefill
             * and first-token sampling.
             */
            if first_token_time.is_none() {
                first_token_time = Some(inference_start.elapsed());

                decode_start = Some(Instant::now());
            }

            /*
             * Stop before appending EOS to
             * visible generated output.
             */
            if is_eos(next_token, self.eos_tokens.as_ref()) {
                finish_reason = "stop".to_string();

                break;
            }

            tokens.push(next_token);

            generated_ids.push(next_token);

            /*
             * Decode the generated sequence.
             *
             * V1 emits only the newly created
             * suffix.
             *
             * A dedicated incremental token
             * output stream can replace this
             * later without changing the
             * inference architecture.
             */
            let current_text = self
                .tokenizer
                .decode(&generated_ids, true)
                .map_err(|error| anyhow::anyhow!("decode failed: {}", error))?;

            if current_text.starts_with(&previous_text) {
                let suffix = &current_text[previous_text.len()..];

                if !suffix.is_empty() {
                    match callback(suffix.to_string()) {
                        StreamControl::Continue => {}

                        StreamControl::Cancel => {
                            finish_reason = "cancelled".to_string();

                            break;
                        }
                    }
                }

                previous_text = current_text;
            } else {
                /*
                 * Tokenizer boundary fallback.
                 *
                 * Avoid emitting duplicated or
                 * invalid text if decoding does
                 * not preserve the previous
                 * string as a strict prefix.
                 */
                previous_text = current_text;
            }
        }

        let inference_time = inference_start.elapsed();

        let model_ttft = first_token_time.unwrap_or(inference_time);

        let decode_time = decode_start
            .map(|start| start.elapsed())
            .unwrap_or_default();

        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|error| anyhow::anyhow!("final decode failed: {}", error))?;

        Ok(GenerationOutput {
            text,

            prompt_tokens,

            generated_tokens: generated_ids.len(),

            model_ttft,

            decode_time,

            inference_time,

            finish_reason,
        })
    }
}

// ============================================================
// SAMPLING
// ============================================================

fn build_sampling(parameters: &GenerationParameters) -> Sampling {
    /*
     * Temperature <= 0 means deterministic
     * greedy decoding.
     */
    if parameters.temperature <= 0.0 {
        return Sampling::ArgMax;
    }

    match (parameters.top_k, parameters.top_p) {
        (None, None) => Sampling::All {
            temperature: parameters.temperature,
        },

        (Some(k), None) => Sampling::TopK {
            k,

            temperature: parameters.temperature,
        },

        (None, Some(p)) => Sampling::TopP {
            p,

            temperature: parameters.temperature,
        },

        (Some(k), Some(p)) => Sampling::TopKThenTopP {
            k,

            p,

            temperature: parameters.temperature,
        },
    }
}

// ============================================================
// EOS
// ============================================================

fn is_eos(token: u32, eos: Option<&LlamaEosToks>) -> bool {
    match eos {
        Some(LlamaEosToks::Single(eos_token)) => token == *eos_token,

        Some(LlamaEosToks::Multiple(tokens)) => tokens.contains(&token),

        None => false,
    }
}

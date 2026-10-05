use crate::{
    backend::candle::{CandleBackend, StreamControl},
    metrics::Metrics,
    types::{InferenceResponse, QueuedRequest, ResponseChannel, Timing, Usage},
};

use std::{
    sync::{Arc, mpsc::Receiver},
    time::Instant,
};

pub fn run_inference_worker(
    queue_rx: Receiver<QueuedRequest>,

    metrics: Arc<Metrics>,

    mut backend: CandleBackend,
) {
    println!("Inference worker started.");

    println!("Execution mode: single-model sequential scheduler");

    while let Ok(request) = queue_rx.recv() {
        process_request(request, &metrics, &mut backend);
    }

    println!("Inference worker stopped.");
}

fn process_request(request: QueuedRequest, metrics: &Arc<Metrics>, backend: &mut CandleBackend) {
    let request_start = request.queued_at;

    let inference_start = Instant::now();

    let queue_time = inference_start.saturating_duration_since(request_start);

    let request_id = request.request_id;

    match request.response {
        ResponseChannel::Standard(response_tx) => {
            let result = backend.generate(&request.prompt, &request.parameters);

            match result {
                Ok(output) => {
                    let total_time = request_start.elapsed();

                    let decode_tokens = output.generated_tokens.saturating_sub(1);

                    let tpot_ms = calculate_tpot(output.decode_time.as_secs_f64(), decode_tokens);

                    let tokens_per_second =
                        calculate_rate(output.decode_time.as_secs_f64(), decode_tokens);

                    let end_to_end_ttft = queue_time + output.model_ttft;

                    metrics.add_tokens(output.prompt_tokens, output.generated_tokens);

                    metrics.inc_completed();

                    let response = InferenceResponse {
                        request_id,

                        text: output.text,

                        finish_reason: output.finish_reason,

                        usage: Usage {
                            prompt_tokens: output.prompt_tokens,

                            generated_tokens: output.generated_tokens,

                            total_tokens: output.prompt_tokens + output.generated_tokens,
                        },

                        timing: Timing {
                            queue_ms: ms(queue_time),

                            model_ttft_ms: ms(output.model_ttft),

                            end_to_end_ttft_ms: ms(end_to_end_ttft),

                            tpot_ms,

                            tokens_per_second,

                            inference_ms: ms(output.inference_time),

                            total_ms: ms(total_time),
                        },
                    };

                    let _ = response_tx.send(Ok(response));
                }

                Err(error) => {
                    metrics.inc_failed();

                    let _ = response_tx.send(Err(error.to_string()));
                }
            }
        }

        ResponseChannel::Stream(stream_tx) => {
            let sender = stream_tx.clone();

            let result =
                backend.generate_stream(&request.prompt, &request.parameters, move |text| {
                    match sender.blocking_send(text) {
                        Ok(_) => StreamControl::Continue,

                        Err(_) => StreamControl::Cancel,
                    }
                });

            match result {
                Ok(output) => {
                    metrics.add_tokens(output.prompt_tokens, output.generated_tokens);

                    if output.finish_reason == "cancelled" {
                        metrics.inc_cancelled();
                    } else {
                        metrics.inc_completed();
                    }

                    let total = request_start.elapsed();

                    let decode_tokens = output.generated_tokens.saturating_sub(1);

                    println!(
                        concat!(
                            "request={} ",
                            "prompt_tokens={} ",
                            "generated_tokens={} ",
                            "queue_ms={:.2} ",
                            "model_ttft_ms={:.2} ",
                            "e2e_ttft_ms={:.2} ",
                            "tok_s={:.2} ",
                            "total_ms={:.2} ",
                            "finish={}"
                        ),
                        request_id,
                        output.prompt_tokens,
                        output.generated_tokens,
                        ms(queue_time,),
                        ms(output.model_ttft,),
                        ms(queue_time + output.model_ttft,),
                        calculate_rate(output.decode_time.as_secs_f64(), decode_tokens,),
                        ms(total),
                        output.finish_reason,
                    );
                }

                Err(error) => {
                    metrics.inc_failed();

                    eprintln!("request={} generation_error={}", request_id, error);
                }
            }

            drop(stream_tx);
        }
    }
}

fn calculate_tpot(seconds: f64, tokens: usize) -> f64 {
    if tokens == 0 || seconds <= 0.0 {
        0.0
    } else {
        seconds * 1000.0 / tokens as f64
    }
}

fn calculate_rate(seconds: f64, tokens: usize) -> f64 {
    if tokens == 0 || seconds <= 0.0 {
        0.0
    } else {
        tokens as f64 / seconds
    }
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

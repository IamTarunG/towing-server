# Towing Server

Towing Server is a lightweight **GPU-accelerated LLM inference server written in Rust**.

The project explores the systems side of LLM serving, including request handling, scheduling, backpressure, KV-cached generation, streaming responses, and inference performance measurement.

## Architecture

```text id="h17dq5"
Client
  ↓
Axum HTTP API
  ↓
Validation + Admission Control
  ↓
Bounded Request Queue
  ↓
Inference Scheduler
  ↓
Candle LLM Backend
  ↓
CUDA GPU
```

## Features

- Rust-based HTTP inference API
- NVIDIA CUDA inference
- Hugging Face model loading
- KV-cached autoregressive generation
- Bounded request queue and backpressure
- Token streaming
- Request cancellation
- Temperature, Top-K and Top-P sampling
- TTFT, TPOT and tokens/sec measurement
- Health and metrics endpoints

## Current Model

```text id="b6f27u"
HuggingFaceTB/SmolLM2-360M-Instruct
```

The model is executed using the **Candle** framework.

## GPU Benchmark

Tested on:

```text id="y8e3cc"
GPU: NVIDIA RTX A400
VRAM: 4 GB
CUDA Toolkit: 12.6
Precision: FP32
```

Measured warm inference performance:

```text id="mupk7x"
Model TTFT:        ~264 ms
Decode throughput: ~46.4 tokens/s
TPOT:              ~21.5 ms/token
Total latency:     ~0.82 s
```

## API

### Standard inference

```http id="o6p5tq"
POST /v1/inference
```

Example:

```json id="34m1eo"
{
  "request_id": 1,
  "prompt": "Explain machine learning in one sentence.",
  "max_tokens": 32,
  "temperature": 0
}
```

### Streaming inference

```http id="vwmmyg"
POST /v1/inference/stream
```

Additional endpoints:

```text id="h3evsl"
GET /health
GET /metrics
GET /bench
```

## Build

```bash id="rlozaf"
cargo build --release --features cuda
```

Run with CUDA:

```bash id="nrwbyo"
set TOWING_DEVICE=cuda
target\release\towing-server.exe
```

## Project Structure

```text id="y0t1uj"
src/
├── main.rs
├── api.rs
├── scheduler.rs
├── metrics.rs
├── state.rs
├── types.rs
├── config.rs
└── backend/
    └── candle.rs
```

## Current Focus

The next stage of the project focuses on:

- sequence-aware scheduling
- concurrent active requests
- dynamic GPU batching
- improved KV-cache management
- FP16 inference
- larger open-weight models

## Tech Stack

Rust · Axum · Tokio · Candle · CUDA · Hugging Face · Safetensors

## Purpose

This project is intended as a hands-on exploration of **ML systems and LLM inference serving**, with emphasis on scheduling, latency, throughput, streaming, and GPU execution.

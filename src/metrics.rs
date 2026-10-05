use serde::Serialize;

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct Metrics {
    received: AtomicU64,

    accepted: AtomicU64,

    rejected: AtomicU64,

    completed: AtomicU64,

    failed: AtomicU64,

    cancelled: AtomicU64,

    prompt_tokens: AtomicU64,

    generated_tokens: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            received: AtomicU64::new(0),

            accepted: AtomicU64::new(0),

            rejected: AtomicU64::new(0),

            completed: AtomicU64::new(0),

            failed: AtomicU64::new(0),

            cancelled: AtomicU64::new(0),

            prompt_tokens: AtomicU64::new(0),

            generated_tokens: AtomicU64::new(0),
        }
    }

    pub fn inc_received(&self) {
        self.received.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_accepted(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_rejected(&self) {
        self.rejected.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_completed(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_cancelled(&self) {
        self.cancelled.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_tokens(&self, prompt: usize, generated: usize) {
        self.prompt_tokens
            .fetch_add(prompt as u64, Ordering::Relaxed);

        self.generated_tokens
            .fetch_add(generated as u64, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            received: self.received.load(Ordering::Relaxed),

            accepted: self.accepted.load(Ordering::Relaxed),

            rejected: self.rejected.load(Ordering::Relaxed),

            completed: self.completed.load(Ordering::Relaxed),

            failed: self.failed.load(Ordering::Relaxed),

            cancelled: self.cancelled.load(Ordering::Relaxed),

            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),

            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MetricsSnapshot {
    pub received: u64,

    pub accepted: u64,

    pub rejected: u64,

    pub completed: u64,

    pub failed: u64,

    pub cancelled: u64,

    pub prompt_tokens: u64,

    pub generated_tokens: u64,
}

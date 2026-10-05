use crate::{metrics::Metrics, types::QueuedRequest};

use std::sync::{Arc, mpsc::SyncSender};

#[derive(Clone)]
pub struct AppState {
    pub queue_tx: SyncSender<QueuedRequest>,

    pub metrics: Arc<Metrics>,

    pub device: &'static str,
}

//! Tracing and Observability Module
//!
//! This module provides comprehensive tracing capabilities for the siumai library,
//! enabling debugging, monitoring, and performance analysis of LLM interactions.
//!
//! ## Features
//!
//! - **HTTP Request/Response Tracing**: Complete HTTP lifecycle tracking
//! - **LLM Interaction Tracing**: Chat messages, tool calls, streaming events
//! - **Performance Monitoring**: Latency, throughput, and resource usage
//! - **Error Tracking**: Detailed error classification and retry analysis
//! - **Structured Logging**: JSON-formatted logs for easy parsing
//!
//! ## Usage
//!
//! ```rust
//! use siumai::tracing::{TracingConfig, init_tracing, OutputFormat};
//!
//! // Initialize tracing with default configuration
//! init_tracing(TracingConfig::default())?;
//!
//! // Or with custom configuration
//! let config = TracingConfig::builder()
//!     .enable_http_tracing(true)
//!     .enable_performance_monitoring(true)
//!     .log_level_str("debug")?
//!     .output_format(OutputFormat::Json)
//!     .build();
//!
//! init_tracing(config)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod config;
pub mod events;
pub mod http;
pub mod llm;
pub mod performance;
pub mod subscriber;

// Re-export main types
pub use config::{OutputFormat, TracingConfig, TracingConfigBuilder};
pub use events::{
    ChatEvent, ErrorEvent, HttpEvent, LlmEvent, PerformanceEvent, StreamEvent, ToolEvent,
    TracingEvent,
};
pub use http::{HttpTracer, RequestContext, ResponseContext};
pub use llm::{ChatTracer, LlmTracer, StreamTracer, ToolTracer};
pub use performance::{PerformanceTracer, TimingContext};
pub use subscriber::{
    init_debug_tracing, init_default_tracing, init_performance_tracing, init_production_tracing,
    init_tracing, init_tracing_from_env,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};
use tracing::Span;
use uuid::Uuid;

/// Unique identifier for a tracing session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceId(pub Uuid);

impl TraceId {
    /// Generate a new trace ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Span identifier for hierarchical tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(pub Uuid);

impl SpanId {
    /// Generate a new span ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Context information for tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingContext {
    /// Unique trace identifier
    pub trace_id: TraceId,
    /// Current span identifier
    pub span_id: SpanId,
    /// Parent span identifier (if any)
    pub parent_span_id: Option<SpanId>,
    /// Provider name (e.g., "openai", "anthropic")
    pub provider: String,
    /// Model name
    pub model: Option<String>,
    /// User-defined tags
    pub tags: HashMap<String, String>,
    /// Session start time
    pub session_start: SystemTime,
}

impl TracingContext {
    /// Create a new tracing context
    pub fn new(provider: String) -> Self {
        Self {
            trace_id: TraceId::new(),
            span_id: SpanId::new(),
            parent_span_id: None,
            provider,
            model: None,
            tags: HashMap::new(),
            session_start: SystemTime::now(),
        }
    }

    /// Create a child context with a new span
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id,
            span_id: SpanId::new(),
            parent_span_id: Some(self.span_id),
            provider: self.provider.clone(),
            model: self.model.clone(),
            tags: self.tags.clone(),
            session_start: self.session_start,
        }
    }

    /// Set the model name
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Get session duration
    pub fn session_duration(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.session_start)
            .unwrap_or_default()
    }
}

/// Tracing utilities
pub struct TracingUtils;

impl TracingUtils {
    /// Create a tracing span with context
    pub fn create_span(_name: &'static str, _context: &TracingContext) -> Span {
        // For now, return a simple span - this would need proper implementation
        tracing::info_span!("siumai_operation")
    }

    /// Extract trace context from current span
    pub fn current_context() -> Option<TracingContext> {
        // This would extract context from the current tracing span
        // Implementation depends on how we store context in spans
        None
    }

    /// Format duration for human readability
    pub fn format_duration(duration: Duration) -> String {
        if duration.as_secs() > 0 {
            format!("{:.2}s", duration.as_secs_f64())
        } else if duration.as_millis() > 0 {
            format!("{}ms", duration.as_millis())
        } else {
            format!("{}μs", duration.as_micros())
        }
    }

    /// Format bytes for human readability
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
}

/// Macro for creating traced HTTP requests
#[macro_export]
macro_rules! traced_http_request {
    ($tracer:expr, $method:expr, $url:expr, $body:expr) => {{
        let context = $tracer.start_request($method, $url);
        // Request execution would go here
        $tracer.end_request(context /* response data */);
    }};
}

/// Macro for creating traced LLM interactions
#[macro_export]
macro_rules! traced_llm_chat {
    ($tracer:expr, $messages:expr, $tools:expr) => {{
        let context = $tracer.start_chat($messages, $tools);
        // Chat execution would go here
        $tracer.end_chat(context /* response */);
    }};
}

/// Global flag for pretty JSON formatting in tracing
static PRETTY_JSON: AtomicBool = AtomicBool::new(false);

/// Global flag for masking sensitive values in tracing
static MASK_SENSITIVE_VALUES: AtomicBool = AtomicBool::new(true);

/// Set the global pretty JSON flag
pub fn set_pretty_json(pretty: bool) {
    PRETTY_JSON.store(pretty, Ordering::Relaxed);
}

/// Get the global pretty JSON flag
pub fn get_pretty_json() -> bool {
    PRETTY_JSON.load(Ordering::Relaxed)
}

/// Set the global mask sensitive values flag
pub fn set_mask_sensitive_values(mask: bool) {
    MASK_SENSITIVE_VALUES.store(mask, Ordering::Relaxed);
}

/// Get the global mask sensitive values flag
pub fn get_mask_sensitive_values() -> bool {
    MASK_SENSITIVE_VALUES.load(Ordering::Relaxed)
}

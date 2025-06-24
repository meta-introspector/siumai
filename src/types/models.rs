//! Model information and capabilities

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Model owner/organization
    pub owned_by: String,
    /// Creation timestamp
    pub created: Option<u64>,
    /// Model capabilities
    pub capabilities: Vec<String>,
    /// Context window size
    pub context_window: Option<u32>,
    /// Maximum output tokens
    pub max_output_tokens: Option<u32>,
    /// Input cost per token
    pub input_cost_per_token: Option<f64>,
    /// Output cost per token
    pub output_cost_per_token: Option<f64>,
}

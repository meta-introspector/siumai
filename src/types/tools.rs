//! Tool calling and function definition types

use serde::{Deserialize, Serialize};

/// Tool calling types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<FunctionCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type (usually "function")
    pub r#type: String,
    /// Function definition
    pub function: ToolFunction,
}

impl Tool {
    /// Create a new function tool
    pub fn function(name: String, description: String, parameters: serde_json::Value) -> Self {
        Self {
            r#type: "function".to_string(),
            function: ToolFunction {
                name,
                description,
                parameters,
            },
        }
    }
}

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    /// Function name
    pub name: String,
    /// Function description
    pub description: String,
    /// JSON schema for function parameters
    pub parameters: serde_json::Value,
}

/// Tool type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
    #[serde(rename = "file_search")]
    FileSearch,
    #[serde(rename = "web_search")]
    WebSearch,
}

/// OpenAI-specific built-in tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpenAiBuiltInTool {
    /// Web search tool
    WebSearch,
    /// File search tool
    FileSearch {
        /// Vector store IDs to search
        vector_store_ids: Option<Vec<String>>,
    },
    /// Computer use tool
    ComputerUse {
        /// Display width
        display_width: u32,
        /// Display height
        display_height: u32,
        /// Environment type
        environment: String,
    },
}

impl OpenAiBuiltInTool {
    /// Convert to JSON for API requests
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::WebSearch => serde_json::json!({
                "type": "web_search_preview"
            }),
            Self::FileSearch { vector_store_ids } => {
                let mut json = serde_json::json!({
                    "type": "file_search"
                });
                if let Some(ids) = vector_store_ids {
                    json["vector_store_ids"] = serde_json::Value::Array(
                        ids.iter()
                            .map(|id| serde_json::Value::String(id.clone()))
                            .collect(),
                    );
                }
                json
            }
            Self::ComputerUse {
                display_width,
                display_height,
                environment,
            } => serde_json::json!({
                "type": "computer_use_preview",
                "display_width": display_width,
                "display_height": display_height,
                "environment": environment
            }),
        }
    }
}

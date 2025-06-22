//! `Groq` Models Capability Implementation
//!
//! Implements model listing and information capabilities for Groq.

use crate::error::LlmError;
use crate::types::HttpConfig;

// Model info type for Groq
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub model_type: ModelType,
    pub context_length: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
    pub created: Option<chrono::DateTime<chrono::Utc>>,
    pub owned_by: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Chat,
    Audio,
    Embedding,
    Image,
}

use super::types::*;
use super::utils::*;

/// `Groq` Models Capability Implementation
pub struct GroqModels {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
}

impl GroqModels {
    /// Create a new `Groq` models capability instance
    pub const fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
        }
    }

    /// Convert Groq model to our ModelInfo
    #[allow(dead_code)]
    fn convert_groq_model(&self, groq_model: GroqModel) -> ModelInfo {
        // Determine model type based on model ID
        let model_type = if groq_model.id.contains("whisper")
            || groq_model.id.contains("tts")
            || groq_model.id.contains("playai")
        {
            ModelType::Audio
        } else {
            ModelType::Chat
        };

        let model_id = groq_model.id.clone();
        ModelInfo {
            id: model_id.clone(),
            name: model_id,
            description: None, // Groq doesn't provide descriptions
            model_type,
            context_length: Some(groq_model.context_window as usize),
            max_output_tokens: groq_model.max_completion_tokens.map(|t| t as usize),
            input_cost_per_token: None, // Groq doesn't provide pricing info via API
            output_cost_per_token: None,
            created: Some(
                chrono::DateTime::from_timestamp(groq_model.created as i64, 0)
                    .unwrap_or_else(chrono::Utc::now),
            ),
            owned_by: groq_model.owned_by.clone(),
            capabilities: self.get_model_capabilities(&groq_model),
        }
    }

    /// Get capabilities for a specific model
    #[allow(dead_code)]
    fn get_model_capabilities(&self, model: &GroqModel) -> Vec<String> {
        let mut capabilities = Vec::new();

        if model.id.contains("whisper") {
            capabilities.push("transcription".to_string());
            if !model.id.contains("-en") {
                capabilities.push("translation".to_string());
            }
        } else if model.id.contains("tts") || model.id.contains("playai") {
            capabilities.push("speech_synthesis".to_string());
        } else {
            // Chat models
            capabilities.push("chat".to_string());
            capabilities.push("streaming".to_string());

            // Most Groq chat models support function calling
            if !model.id.contains("guard") {
                capabilities.push("function_calling".to_string());
            }

            // Some models support reasoning
            if model.id.contains("qwen") {
                capabilities.push("reasoning".to_string());
            }
        }

        capabilities
    }
}

#[allow(dead_code)]
impl GroqModels {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let url = format!("{}/models", self.base_url);
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq list models error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_response: GroqModelsResponse = response.json().await?;
        let models = groq_response
            .data
            .into_iter()
            .map(|m| self.convert_groq_model(m))
            .collect();

        Ok(models)
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        let url = format!("{}/models/{}", self.base_url, model_id);
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq get model error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_model: GroqModel = response.json().await?;
        Ok(self.convert_groq_model(groq_model))
    }

    fn supports_model_listing(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HttpConfig;

    fn create_test_models() -> GroqModels {
        GroqModels::new(
            "test-api-key".to_string(),
            "https://api.groq.com/openai/v1".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        )
    }

    #[test]
    fn test_convert_groq_model_chat() {
        let models = create_test_models();
        let groq_model = GroqModel {
            id: "llama-3.3-70b-versatile".to_string(),
            object: "model".to_string(),
            created: 1640995200,
            owned_by: "Meta".to_string(),
            active: true,
            context_window: 32768,
            public_apps: None,
            max_completion_tokens: Some(8192),
        };

        let model_info = models.convert_groq_model(groq_model);

        assert_eq!(model_info.id, "llama-3.3-70b-versatile");
        assert_eq!(model_info.model_type, ModelType::Chat);
        assert_eq!(model_info.context_length, Some(32768));
        assert_eq!(model_info.max_output_tokens, Some(8192));
        assert!(model_info.capabilities.contains(&"chat".to_string()));
        assert!(model_info.capabilities.contains(&"streaming".to_string()));
        assert!(
            model_info
                .capabilities
                .contains(&"function_calling".to_string())
        );
    }

    #[test]
    fn test_convert_groq_model_whisper() {
        let models = create_test_models();
        let groq_model = GroqModel {
            id: "whisper-large-v3".to_string(),
            object: "model".to_string(),
            created: 1640995200,
            owned_by: "OpenAI".to_string(),
            active: true,
            context_window: 448,
            public_apps: None,
            max_completion_tokens: None,
        };

        let model_info = models.convert_groq_model(groq_model);

        assert_eq!(model_info.id, "whisper-large-v3");
        assert_eq!(model_info.model_type, ModelType::Audio);
        assert!(
            model_info
                .capabilities
                .contains(&"transcription".to_string())
        );
        assert!(model_info.capabilities.contains(&"translation".to_string()));
    }

    #[test]
    fn test_convert_groq_model_qwen() {
        let models = create_test_models();
        let groq_model = GroqModel {
            id: "qwen3-8b-instruct".to_string(),
            object: "model".to_string(),
            created: 1640995200,
            owned_by: "Alibaba".to_string(),
            active: true,
            context_window: 8192,
            public_apps: None,
            max_completion_tokens: Some(4096),
        };

        let model_info = models.convert_groq_model(groq_model);

        assert_eq!(model_info.id, "qwen3-8b-instruct");
        assert_eq!(model_info.model_type, ModelType::Chat);
        assert!(model_info.capabilities.contains(&"reasoning".to_string()));
    }

    #[test]
    fn test_capability_support() {
        let models = create_test_models();
        assert!(models.supports_model_listing());
    }
}

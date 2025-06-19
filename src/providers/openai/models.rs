//! OpenAI Models API Implementation
//!
//! This module implements the OpenAI Models API for listing and retrieving
//! information about available models.
//!
//! API Reference: https://platform.openai.com/docs/api-reference/models

use async_trait::async_trait;
use reqwest::header::HeaderMap;

use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::{HttpConfig, ModelInfo};

use super::types::*;
use super::utils::build_headers;

/// OpenAI Models API client
pub struct OpenAiModels {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for OpenAI API
    pub base_url: String,
    /// HTTP client
    pub http_client: reqwest::Client,
    /// Organization ID (optional)
    pub organization: Option<String>,
    /// Project ID (optional)
    pub project: Option<String>,
    /// HTTP configuration
    pub http_config: HttpConfig,
}

impl OpenAiModels {
    /// Create a new OpenAI models client
    pub fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        organization: Option<String>,
        project: Option<String>,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            organization,
            project,
            http_config,
        }
    }

    /// Build headers for API requests
    fn build_request_headers(&self) -> Result<HeaderMap, LlmError> {
        build_headers(
            &self.api_key,
            self.organization.as_deref(),
            self.project.as_deref(),
            &self.http_config.headers,
        )
    }

    /// Get the models endpoint URL
    fn models_endpoint(&self) -> String {
        format!("{}/models", self.base_url)
    }

    /// Get a specific model endpoint URL
    fn model_endpoint(&self, model_id: &str) -> String {
        format!("{}/models/{}", self.base_url, model_id)
    }

    /// Convert OpenAI model response to ModelInfo
    fn convert_openai_model_to_model_info(&self, openai_model: OpenAiModel) -> ModelInfo {
        // Determine capabilities based on model ID
        let capabilities = determine_model_capabilities(&openai_model.id);

        // Estimate context window and costs based on model type
        let (context_window, max_output_tokens, input_cost, output_cost) =
            estimate_model_specs(&openai_model.id);

        ModelInfo {
            id: openai_model.id.clone(),
            name: Some(openai_model.id.clone()),
            description: Some(format!("OpenAI {} model", openai_model.id)),
            owned_by: openai_model.owned_by,
            created: openai_model.created,
            capabilities,
            context_window,
            max_output_tokens,
            input_cost_per_token: input_cost,
            output_cost_per_token: output_cost,
        }
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiModels {
    /// List all available models
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let headers = self.build_request_headers()?;
        let url = self.models_endpoint();

        let response = self
            .http_client
            .get(&url)
            .headers(headers)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI Models API error: {}", error_text),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let models_response: OpenAiModelsResponse = response.json().await?;
        
        let models = models_response
            .data
            .into_iter()
            .map(|model| self.convert_openai_model_to_model_info(model))
            .collect();

        Ok(models)
    }

    /// Get information about a specific model
    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        let headers = self.build_request_headers()?;
        let url = self.model_endpoint(&model_id);

        let response = self
            .http_client
            .get(&url)
            .headers(headers)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI Model API error: {}", error_text),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let openai_model: OpenAiModel = response.json().await?;
        Ok(self.convert_openai_model_to_model_info(openai_model))
    }
}

/// Determine model capabilities based on model ID
fn determine_model_capabilities(model_id: &str) -> Vec<String> {
    let mut capabilities = vec!["chat".to_string(), "text".to_string()];

    // GPT-4 models have vision capability
    if model_id.contains("gpt-4") && !model_id.contains("gpt-4-turbo-preview") {
        capabilities.push("vision".to_string());
    }

    // o1 models have reasoning capability
    if model_id.contains("o1") {
        capabilities.push("reasoning".to_string());
    }

    // All modern models support tools
    if !model_id.contains("gpt-3.5") || model_id.contains("gpt-3.5-turbo") {
        capabilities.push("tools".to_string());
    }

    // All models support streaming
    capabilities.push("streaming".to_string());

    capabilities
}

/// Estimate model specifications based on model ID
fn estimate_model_specs(model_id: &str) -> (Option<u32>, Option<u32>, Option<f64>, Option<f64>) {
    match model_id {
        // GPT-4o models
        id if id.contains("gpt-4o") => (Some(128000), Some(16384), Some(0.0000025), Some(0.00001)),
        // GPT-4 Turbo models
        id if id.contains("gpt-4-turbo") => (Some(128000), Some(4096), Some(0.00001), Some(0.00003)),
        // GPT-4 models
        id if id.contains("gpt-4") => (Some(8192), Some(4096), Some(0.00003), Some(0.00006)),
        // o1 models
        id if id.contains("o1-preview") => (Some(128000), Some(32768), Some(0.000015), Some(0.00006)),
        id if id.contains("o1-mini") => (Some(128000), Some(65536), Some(0.000003), Some(0.000012)),
        // GPT-3.5 Turbo
        id if id.contains("gpt-3.5-turbo") => (Some(16385), Some(4096), Some(0.0000005), Some(0.0000015)),
        // Default fallback
        _ => (Some(4096), Some(2048), Some(0.00001), Some(0.00003)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAiConfig;

    #[test]
    fn test_models_endpoint() {
        let config = OpenAiConfig::new("test-key");
        let models = OpenAiModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            reqwest::Client::new(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
        );

        assert_eq!(models.models_endpoint(), "https://api.openai.com/v1/models");
        assert_eq!(models.model_endpoint("gpt-4"), "https://api.openai.com/v1/models/gpt-4");
    }

    #[test]
    fn test_determine_model_capabilities() {
        let gpt4_caps = determine_model_capabilities("gpt-4");
        assert!(gpt4_caps.contains(&"vision".to_string()));
        assert!(gpt4_caps.contains(&"tools".to_string()));

        let o1_caps = determine_model_capabilities("o1-preview");
        assert!(o1_caps.contains(&"reasoning".to_string()));

        let gpt35_caps = determine_model_capabilities("gpt-3.5-turbo");
        assert!(gpt35_caps.contains(&"tools".to_string()));
    }

    #[test]
    fn test_estimate_model_specs() {
        let (context, max_output, input_cost, output_cost) = estimate_model_specs("gpt-4o");
        assert_eq!(context, Some(128000));
        assert_eq!(max_output, Some(16384));
        assert!(input_cost.is_some());
        assert!(output_cost.is_some());
    }
}

//! Request Factory Module
//!
//! Provides unified request building patterns across all providers.
//! This module standardizes how ChatRequest objects are constructed
//! and ensures consistent parameter handling.

use crate::error::LlmError;
use crate::types::{ChatMessage, ChatRequest, CommonParams, ProviderParams, Tool};

/// Configuration for request building behavior
#[derive(Debug, Clone, Default)]
pub struct RequestBuilderConfig {
    /// Whether to enable strict parameter validation
    pub strict_validation: bool,
    /// Whether to enable provider-specific parameter validation
    pub provider_validation: bool,
}

/// Trait for building standardized ChatRequest objects
///
/// This trait ensures all providers follow the same pattern for
/// constructing ChatRequest objects with proper parameter handling.
pub trait RequestBuilder {
    /// Build a ChatRequest with consistent parameter handling
    ///
    /// # Arguments
    /// * `messages` - The conversation messages
    /// * `tools` - Optional tools to include
    /// * `stream` - Whether this is a streaming request
    ///
    /// # Returns
    /// A properly constructed ChatRequest with provider-specific parameters included
    fn build_chat_request(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> Result<ChatRequest, LlmError>;

    /// Build a ChatRequest with custom configuration
    ///
    /// # Arguments
    /// * `messages` - The conversation messages
    /// * `tools` - Optional tools to include
    /// * `stream` - Whether this is a streaming request
    /// * `config` - Configuration for request building behavior
    ///
    /// # Returns
    /// A properly constructed ChatRequest with provider-specific parameters included
    fn build_chat_request_with_config(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        stream: bool,
        config: &RequestBuilderConfig,
    ) -> Result<ChatRequest, LlmError> {
        // Default implementation delegates to the standard method
        // Providers can override this for custom validation behavior
        let _ = config; // Suppress unused parameter warning
        self.build_chat_request(messages, tools, stream)
    }

    /// Get the common parameters for this provider instance
    fn get_common_params(&self) -> &CommonParams;

    /// Get the provider-specific parameters
    fn get_provider_params(&self) -> Option<ProviderParams>;

    /// Validate the request before sending
    fn validate_request(&self, request: &ChatRequest) -> Result<(), LlmError> {
        // Default validation - can be overridden by providers
        if request.messages.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Messages cannot be empty".to_string(),
            ));
        }

        if request.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        Ok(())
    }
}

/// Standard implementation helper for RequestBuilder
pub struct StandardRequestBuilder {
    common_params: CommonParams,
    provider_params: Option<ProviderParams>,
}

impl StandardRequestBuilder {
    /// Create a new standard request builder
    pub fn new(common_params: CommonParams, provider_params: Option<ProviderParams>) -> Self {
        Self {
            common_params,
            provider_params,
        }
    }

    /// Build a ChatRequest using the standard pattern
    pub fn build_standard_request(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: self.provider_params.clone(),
            http_config: None,
            web_search: None,
            stream,
        };

        // Validate the request
        self.validate_standard_request(&request)?;

        Ok(request)
    }

    /// Standard validation logic
    fn validate_standard_request(&self, request: &ChatRequest) -> Result<(), LlmError> {
        if request.messages.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Messages cannot be empty".to_string(),
            ));
        }

        if request.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        Ok(())
    }
}

impl RequestBuilder for StandardRequestBuilder {
    fn build_chat_request(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        self.build_standard_request(messages, tools, stream)
    }

    fn get_common_params(&self) -> &CommonParams {
        &self.common_params
    }

    fn get_provider_params(&self) -> Option<ProviderParams> {
        self.provider_params.clone()
    }

    fn validate_request(&self, request: &ChatRequest) -> Result<(), LlmError> {
        self.validate_standard_request(request)
    }
}

/// Factory for creating provider-specific request builders
pub struct RequestBuilderFactory;

impl RequestBuilderFactory {
    /// Create a request builder for the specified provider type
    pub fn create_builder(
        provider_type: &crate::types::ProviderType,
        common_params: CommonParams,
        provider_params: Option<ProviderParams>,
    ) -> Box<dyn RequestBuilder> {
        match provider_type {
            crate::types::ProviderType::OpenAi => {
                Box::new(StandardRequestBuilder::new(common_params, provider_params))
            }
            crate::types::ProviderType::Anthropic => {
                Box::new(StandardRequestBuilder::new(common_params, provider_params))
            }
            crate::types::ProviderType::Gemini => {
                Box::new(StandardRequestBuilder::new(common_params, provider_params))
            }
            crate::types::ProviderType::Ollama => {
                Box::new(StandardRequestBuilder::new(common_params, provider_params))
            }
            crate::types::ProviderType::XAI => {
                Box::new(StandardRequestBuilder::new(common_params, provider_params))
            }
            crate::types::ProviderType::Groq => {
                Box::new(StandardRequestBuilder::new(common_params, provider_params))
            }
            crate::types::ProviderType::Custom(_) => {
                Box::new(StandardRequestBuilder::new(common_params, provider_params))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageContent, MessageRole};

    #[test]
    fn test_standard_request_builder() {
        let common_params = CommonParams {
            model: "test-model".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let builder = StandardRequestBuilder::new(common_params, None);

        let messages = vec![crate::types::ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: Default::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = builder
            .build_chat_request(messages, None, false)
            .expect("Should build request successfully");

        assert_eq!(request.common_params.model, "test-model");
        assert!(!request.stream);
        assert!(request.provider_params.is_none());
    }

    #[test]
    fn test_request_validation() {
        let common_params = CommonParams {
            model: "".to_string(), // Empty model should fail validation
            ..Default::default()
        };

        let builder = StandardRequestBuilder::new(common_params, None);

        let messages = vec![crate::types::ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: Default::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let result = builder.build_chat_request(messages, None, false);
        assert!(result.is_err());
    }
}

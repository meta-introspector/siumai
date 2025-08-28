//! Request Factory Module - Parameter Management Layer
//!
//! ## 🎯 Core Responsibility: Parameter Management and Request Building
//!
//! This module is the **parameter management layer** of the LLM library architecture.
//! It is responsible for:
//!
//! ### ✅ What RequestBuilder Does:
//! - **Parameter Validation**: Validates common and provider-specific parameters
//! - **Parameter Mapping**: Maps unified parameters to provider-specific formats
//! - **Request Construction**: Builds standardized ChatRequest objects
//! - **Parameter Constraints**: Enforces parameter constraints (ranges, types, etc.)
//! - **Provider Abstraction**: Provides unified parameter interface across providers
//!
//! ### ❌ What RequestBuilder Does NOT Do:
//! - **Client Construction**: Does not create or configure HTTP clients
//! - **Authentication**: Does not handle API keys or authentication
//! - **Network Configuration**: Does not manage timeouts, retries, or HTTP settings
//! - **Provider Selection**: Does not decide which provider to use
//! - **Business Logic**: Does not implement chat logic or streaming
//!
//! ## 🏗️ Architecture Position
//!
//! ```text
//! User Code
//!     ↓
//! SiumaiBuilder (Client Configuration Layer)
//!     ↓
//! RequestBuilder (Parameter Management Layer) ← YOU ARE HERE
//!     ↓
//! Provider Clients (Implementation Layer)
//!     ↓
//! HTTP/Network Layer
//! ```
//!
//! ## 🔄 Relationship with LlmBuilder
//!
//! - **RequestBuilder**: Handles parameters, validation, and request building
//! - **LlmBuilder**: Handles client configuration, HTTP setup, and provider instantiation
//! - **Separation**: These are different architectural layers with distinct responsibilities
//!
//! This clear separation ensures maintainable code where parameter logic
//! is centralized and client configuration is handled separately.

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

/// Core trait for parameter management and request building
///
/// ## 🎯 Primary Responsibility: Parameter Management
///
/// This trait defines the **parameter management layer** interface.
/// It is responsible for taking raw parameters and converting them
/// into validated, properly formatted ChatRequest objects.
///
/// ### Key Responsibilities:
/// 1. **Parameter Validation**: Ensure parameters are within valid ranges
/// 2. **Parameter Mapping**: Convert unified parameters to provider formats
/// 3. **Request Building**: Construct ChatRequest objects with proper structure
/// 4. **Constraint Enforcement**: Apply provider-specific parameter constraints
///
/// ### Usage Pattern:
/// ```rust,no_run
/// use siumai::request_factory::RequestBuilderFactory;
/// use siumai::types::{CommonParams, ProviderParams, ProviderType, ChatMessage, Tool};
///
/// // 1. Create builder with parameters
/// let provider_type = ProviderType::OpenAi;
/// let common_params = CommonParams::default();
/// let provider_params = None;
/// let builder = RequestBuilderFactory::create_builder(
///     &provider_type,
///     common_params,
///     provider_params,
/// );
///
/// // 2. Validate configuration (used by SiumaiBuilder)
/// builder.validate_configuration();
///
/// // 3. Build requests (used by client implementations)
/// let messages = vec![ChatMessage::user("Hello").build()];
/// let tools: Option<Vec<Tool>> = None;
/// let stream = false;
/// let request = builder.build_chat_request(messages, tools, stream);
/// ```
///
/// ### Architecture Note:
/// This trait operates at the **parameter layer**, not the **client layer**.
/// It does not handle HTTP clients, authentication, or network configuration.
/// Those responsibilities belong to LlmBuilder and provider clients.
pub trait RequestBuilder: Send + Sync {
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

    /// Validate configuration without building a request
    ///
    /// This method validates the common and provider parameters
    /// without requiring messages or building an actual request.
    /// Useful for validating configuration during client construction.
    fn validate_configuration(&self) -> Result<(), LlmError> {
        let common_params = self.get_common_params();

        // Validate model
        if common_params.model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model must be specified".to_string(),
            ));
        }

        // Validate temperature range
        if let Some(temp) = common_params.temperature
            && !(0.0..=2.0).contains(&temp)
        {
            return Err(LlmError::ConfigurationError(format!(
                "Temperature must be between 0.0 and 2.0, got {}",
                temp
            )));
        }

        // Validate top_p range
        if let Some(top_p) = common_params.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(LlmError::ConfigurationError(format!(
                "top_p must be between 0.0 and 1.0, got {}",
                top_p
            )));
        }

        // Validate max_tokens
        if let Some(max_tokens) = common_params.max_tokens
            && max_tokens == 0
        {
            return Err(LlmError::ConfigurationError(
                "max_tokens must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get validated common parameters
    ///
    /// Returns the common parameters after validation.
    /// This ensures that any parameters returned have been validated.
    fn get_validated_common_params(&self) -> Result<&CommonParams, LlmError> {
        self.validate_configuration()?;
        Ok(self.get_common_params())
    }

    /// Get validated provider parameters
    ///
    /// Returns the provider parameters after validation.
    fn get_validated_provider_params(&self) -> Result<Option<ProviderParams>, LlmError> {
        self.validate_configuration()?;
        Ok(self.get_provider_params())
    }
}

/// Standard implementation helper for RequestBuilder
#[derive(Clone)]
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
///
/// ## 🎯 Core Responsibility: RequestBuilder Instantiation
///
/// This factory creates the appropriate RequestBuilder implementation
/// for each provider type, ensuring proper parameter management.
///
/// ### Key Functions:
/// - **Builder Creation**: Creates provider-specific RequestBuilder instances
/// - **Parameter Injection**: Injects common and provider parameters into builders
/// - **Validation Coordination**: Provides validation entry points for SiumaiBuilder
/// - **Provider Abstraction**: Hides provider-specific RequestBuilder details
///
/// ### Usage by SiumaiBuilder:
/// ```rust,no_run
/// use siumai::request_factory::RequestBuilderFactory;
/// use siumai::types::{CommonParams, ProviderParams, ProviderType};
///
/// // SiumaiBuilder uses this factory to validate parameters
/// let provider_type = ProviderType::OpenAi;
/// let common_params = CommonParams::default();
/// let provider_params = None;
/// let _builder = RequestBuilderFactory::create_and_validate_builder(
///     &provider_type,
///     common_params,
///     provider_params,
/// );
/// ```
///
/// This factory serves as the **bridge** between the unified interface (SiumaiBuilder)
/// and the parameter management layer (RequestBuilder implementations).
#[derive(Clone)]
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
            crate::types::ProviderType::Custom(name) => {
                // Handle OpenAI-compatible providers
                match name.as_str() {
                    "deepseek" | "openrouter" | "groq" | "xai" => {
                        // OpenAI-compatible providers use OpenAI request builder pattern
                        Box::new(StandardRequestBuilder::new(common_params, provider_params))
                    }
                    _ => {
                        // Other custom providers use standard builder
                        Box::new(StandardRequestBuilder::new(common_params, provider_params))
                    }
                }
            }
        }
    }

    /// Create and validate a request builder for the specified provider type
    ///
    /// This method creates a request builder and immediately validates its configuration.
    /// This is useful for SiumaiBuilder to ensure parameters are valid during client construction.
    pub fn create_and_validate_builder(
        provider_type: &crate::types::ProviderType,
        common_params: CommonParams,
        provider_params: Option<ProviderParams>,
    ) -> Result<Box<dyn RequestBuilder>, LlmError> {
        let builder = Self::create_builder(provider_type, common_params, provider_params);

        // Validate the configuration
        builder.validate_configuration()?;

        Ok(builder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageContent, MessageRole};

    // Test that RequestBuilder implementations satisfy Send + Sync constraints
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_request_builder_send_sync() {
        // Verify that StandardRequestBuilder implements Send + Sync
        assert_send_sync::<StandardRequestBuilder>();

        // Verify that Box<dyn RequestBuilder> implements Send + Sync
        assert_send_sync::<Box<dyn RequestBuilder>>();
    }

    #[test]
    fn test_provider_specific_request_builders_send_sync() {
        use crate::providers::anthropic::request::AnthropicRequestBuilder;
        use crate::providers::gemini::request::GeminiRequestBuilder;
        use crate::providers::openai::request::OpenAiRequestBuilder;

        // Verify that provider-specific RequestBuilder implementations support Send + Sync
        assert_send_sync::<OpenAiRequestBuilder>();
        assert_send_sync::<AnthropicRequestBuilder>();
        assert_send_sync::<GeminiRequestBuilder>();
    }

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

    #[test]
    fn test_configuration_validation() {
        // Test valid configuration
        let valid_params = CommonParams {
            model: "test-model".to_string(),
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(1000),
            ..Default::default()
        };

        let builder = StandardRequestBuilder::new(valid_params, None);
        assert!(builder.validate_configuration().is_ok());

        // Test invalid temperature
        let invalid_temp_params = CommonParams {
            model: "test-model".to_string(),
            temperature: Some(3.0), // Invalid: > 2.0
            ..Default::default()
        };

        let invalid_builder = StandardRequestBuilder::new(invalid_temp_params, None);
        assert!(invalid_builder.validate_configuration().is_err());

        // Test invalid top_p
        let invalid_top_p_params = CommonParams {
            model: "test-model".to_string(),
            top_p: Some(1.5), // Invalid: > 1.0
            ..Default::default()
        };

        let invalid_top_p_builder = StandardRequestBuilder::new(invalid_top_p_params, None);
        assert!(invalid_top_p_builder.validate_configuration().is_err());

        // Test empty model
        let empty_model_params = CommonParams {
            model: "".to_string(), // Invalid: empty
            ..Default::default()
        };

        let empty_model_builder = StandardRequestBuilder::new(empty_model_params, None);
        assert!(empty_model_builder.validate_configuration().is_err());
    }

    #[test]
    fn test_factory_validation() {
        use crate::types::ProviderType;

        // Test valid configuration
        let valid_params = CommonParams {
            model: "test-model".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let result = RequestBuilderFactory::create_and_validate_builder(
            &ProviderType::OpenAi,
            valid_params,
            None,
        );
        assert!(result.is_ok());

        // Test invalid configuration
        let invalid_params = CommonParams {
            model: "".to_string(), // Invalid: empty
            ..Default::default()
        };

        let invalid_result = RequestBuilderFactory::create_and_validate_builder(
            &ProviderType::OpenAi,
            invalid_params,
            None,
        );
        assert!(invalid_result.is_err());
    }
}

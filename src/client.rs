//! Client Module
//!
//! Defines a unified LLM client interface with dynamic dispatch support.

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;

/// Unified LLM client interface
pub trait LlmClient: ChatCapability + Send + Sync {
    /// Get the provider name
    fn provider_name(&self) -> &'static str;

    /// Get the list of supported models
    fn supported_models(&self) -> Vec<String>;

    /// Get capability information
    fn capabilities(&self) -> ProviderCapabilities;
}

/// Client Wrapper - used to unify clients from different providers
pub enum ClientWrapper {
    OpenAi(Box<dyn LlmClient>),
    Anthropic(Box<dyn LlmClient>),
    Gemini(Box<dyn LlmClient>),
    Custom(Box<dyn LlmClient>),
}

impl ClientWrapper {
    /// Creates an OpenAI client wrapper
    pub fn openai(client: Box<dyn LlmClient>) -> Self {
        Self::OpenAi(client)
    }

    /// Creates an Anthropic client wrapper
    pub fn anthropic(client: Box<dyn LlmClient>) -> Self {
        Self::Anthropic(client)
    }

    /// Creates a Gemini client wrapper
    pub fn gemini(client: Box<dyn LlmClient>) -> Self {
        Self::Gemini(client)
    }

    /// Creates a custom client wrapper
    pub fn custom(client: Box<dyn LlmClient>) -> Self {
        Self::Custom(client)
    }

    /// Gets a reference to the internal client
    pub fn client(&self) -> &dyn LlmClient {
        match self {
            Self::OpenAi(client) => client.as_ref(),
            Self::Anthropic(client) => client.as_ref(),
            Self::Gemini(client) => client.as_ref(),
            Self::Custom(client) => client.as_ref(),
        }
    }

    /// Gets the provider type
    pub fn provider_type(&self) -> ProviderType {
        match self {
            Self::OpenAi(_) => ProviderType::OpenAi,
            Self::Anthropic(_) => ProviderType::Anthropic,
            Self::Gemini(_) => ProviderType::Gemini,
            Self::Custom(_) => ProviderType::Custom("unknown".to_string()),
        }
    }

    /// Check if the client supports a specific capability
    pub fn supports_capability(&self, capability: &str) -> bool {
        self.client().capabilities().supports(capability)
    }

    /// Get all supported capabilities
    pub fn get_capabilities(&self) -> ProviderCapabilities {
        self.client().capabilities()
    }
}

#[async_trait::async_trait]
impl ChatCapability for ClientWrapper {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.client().chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.client().chat_stream(messages, tools).await
    }
}

/// Unified LLM client that provides dynamic dispatch to different providers
///
/// This is similar to llm_dart's unified interface approach, allowing you to
/// call different provider functionality through a single interface.
pub struct UnifiedLlmClient {
    inner: ClientWrapper,
}

impl UnifiedLlmClient {
    /// Create a new unified client from a provider-specific client
    pub fn new(client: ClientWrapper) -> Self {
        Self { inner: client }
    }

    /// Create from an OpenAI client
    pub fn from_openai(client: Box<dyn LlmClient>) -> Self {
        Self::new(ClientWrapper::openai(client))
    }

    /// Create from an Anthropic client
    pub fn from_anthropic(client: Box<dyn LlmClient>) -> Self {
        Self::new(ClientWrapper::anthropic(client))
    }

    /// Create from a Gemini client
    pub fn from_gemini(client: Box<dyn LlmClient>) -> Self {
        Self::new(ClientWrapper::gemini(client))
    }

    /// Create from a custom client
    pub fn from_custom(client: Box<dyn LlmClient>) -> Self {
        Self::new(ClientWrapper::custom(client))
    }

    /// Get the provider name
    pub fn provider_name(&self) -> &'static str {
        self.inner.client().provider_name()
    }

    /// Get the provider type
    pub fn provider_type(&self) -> ProviderType {
        self.inner.provider_type()
    }

    /// Check if a capability is supported
    pub fn supports(&self, capability: &str) -> bool {
        self.inner.supports_capability(capability)
    }

    /// Get all capabilities
    pub fn capabilities(&self) -> ProviderCapabilities {
        self.inner.get_capabilities()
    }

    /// Try to cast to a specific capability trait
    /// This enables dynamic dispatch to provider-specific features
    pub fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        if self.supports("audio") {
            // This would require some unsafe casting or a different approach
            // For now, we'll return None and suggest using the provider-specific client
            None
        } else {
            None
        }
    }

    /// Try to cast to embedding capability
    pub fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        if self.supports("embedding") {
            None // Similar to above
        } else {
            None
        }
    }

    /// Try to cast to vision capability
    pub fn as_vision_capability(&self) -> Option<&dyn VisionCapability> {
        if self.supports("vision") {
            None // Similar to above
        } else {
            None
        }
    }

    /// Try to cast to image generation capability
    pub fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        if self.supports("image_generation") {
            None // Similar to above
        } else {
            None
        }
    }
}

#[async_trait::async_trait]
impl ChatCapability for UnifiedLlmClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.inner.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.inner.chat_stream(messages, tools).await
    }
}

impl LlmClient for UnifiedLlmClient {
    fn provider_name(&self) -> &'static str {
        self.inner.client().provider_name()
    }

    fn supported_models(&self) -> Vec<String> {
        self.inner.client().supported_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.client().capabilities()
    }
}

impl LlmClient for ClientWrapper {
    fn provider_name(&self) -> &'static str {
        self.client().provider_name()
    }

    fn supported_models(&self) -> Vec<String> {
        self.client().supported_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.client().capabilities()
    }
}

/// Client Configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// API Key
    pub api_key: String,
    /// Base URL
    pub base_url: String,
    /// HTTP Configuration
    pub http_config: HttpConfig,
    /// Common Parameters
    pub common_params: CommonParams,
    /// Provider-specific Parameters
    pub provider_params: ProviderParams,
}

impl ClientConfig {
    /// Creates a new client configuration
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            http_config: HttpConfig::default(),
            common_params: CommonParams::default(),
            provider_params: ProviderParams::default(),
        }
    }

    /// Sets the HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Sets the common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Sets the provider parameters
    pub fn with_provider_params(mut self, params: ProviderParams) -> Self {
        self.provider_params = params;
        self
    }
}

/// Client Factory
pub struct ClientFactory;

impl ClientFactory {
    /// Creates a client based on the provider type
    pub async fn create_client(
        provider_type: ProviderType,
        _config: ClientConfig,
    ) -> Result<ClientWrapper, LlmError> {
        match provider_type {
            ProviderType::OpenAi => {
                // This will be filled in after the OpenAI client is implemented
                Err(LlmError::UnsupportedOperation(
                    "OpenAI client not yet implemented".to_string(),
                ))
            }
            ProviderType::Anthropic => {
                // This will be filled in after the Anthropic client is implemented
                Err(LlmError::UnsupportedOperation(
                    "Anthropic client not yet implemented".to_string(),
                ))
            }
            ProviderType::Gemini => Err(LlmError::UnsupportedOperation(
                "Gemini client not yet implemented".to_string(),
            )),
            ProviderType::XAI => Err(LlmError::UnsupportedOperation(
                "xAI client not yet implemented".to_string(),
            )),
            ProviderType::Ollama => Err(LlmError::UnsupportedOperation(
                "Ollama client not yet implemented in ClientWrapper".to_string(),
            )),
            ProviderType::Custom(_) => Err(LlmError::UnsupportedOperation(
                "Custom client not yet implemented".to_string(),
            )),
        }
    }
}

/// Client Manager - used to manage multiple client instances
pub struct ClientManager {
    clients: std::collections::HashMap<String, ClientWrapper>,
}

impl ClientManager {
    /// Creates a new client manager
    pub fn new() -> Self {
        Self {
            clients: std::collections::HashMap::new(),
        }
    }

    /// Adds a client
    pub fn add_client(&mut self, name: String, client: ClientWrapper) {
        self.clients.insert(name, client);
    }

    /// Gets a client
    pub fn get_client(&self, name: &str) -> Option<&ClientWrapper> {
        self.clients.get(name)
    }

    /// Removes a client
    pub fn remove_client(&mut self, name: &str) -> Option<ClientWrapper> {
        self.clients.remove(name)
    }

    /// Lists all client names
    pub fn list_clients(&self) -> Vec<&String> {
        self.clients.keys().collect()
    }

    /// Gets the default client (the first one added)
    pub fn default_client(&self) -> Option<&ClientWrapper> {
        self.clients.values().next()
    }
}

impl Default for ClientManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Client Pool - used for connection pool management
pub struct ClientPool {
    pool: std::sync::Arc<std::sync::Mutex<Vec<ClientWrapper>>>,
    max_size: usize,
}

impl ClientPool {
    /// Creates a new client pool
    pub fn new(max_size: usize) -> Self {
        Self {
            pool: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            max_size,
        }
    }

    /// Gets a client
    pub fn get_client(&self) -> Option<ClientWrapper> {
        let mut pool = self.pool.lock().unwrap();
        pool.pop()
    }

    /// Returns a client
    pub fn return_client(&self, client: ClientWrapper) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.max_size {
            pool.push(client);
        }
    }

    /// Gets the pool size
    pub fn size(&self) -> usize {
        let pool = self.pool.lock().unwrap();
        pool.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_manager() {
        let manager = ClientManager::new();
        assert_eq!(manager.list_clients().len(), 0);
        assert!(manager.default_client().is_none());
    }

    #[test]
    fn test_client_pool() {
        let pool = ClientPool::new(5);
        assert_eq!(pool.size(), 0);
        assert!(pool.get_client().is_none());
    }

    #[test]
    fn test_client_config() {
        let config = ClientConfig::new(
            "test-key".to_string(),
            "https://api.example.com".to_string(),
        );
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.example.com");
    }
}

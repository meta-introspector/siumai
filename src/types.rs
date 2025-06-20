//! Core Data Type Definitions
//!
//! Defines all data structures used in the LLM library.

use crate::error::LlmError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Provider type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderType {
    OpenAi,
    Anthropic,
    Gemini,
    XAI,
    Custom(String),
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAi => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Gemini => write!(f, "gemini"),
            Self::XAI => write!(f, "xai"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Common AI parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonParams {
    /// Model name
    pub model: String,
    /// Temperature parameter (0.0-2.0)
    pub temperature: Option<f32>,
    /// Maximum output tokens
    pub max_tokens: Option<u32>,
    /// top_p parameter
    pub top_p: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for CommonParams {
    fn default() -> Self {
        Self {
            model: String::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop_sequences: None,
            seed: None,
        }
    }
}

/// Provider-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderParams {
    /// A map for provider-specific parameters
    pub params: HashMap<String, serde_json::Value>,
}

impl ProviderParams {
    /// Creates new provider parameters
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Adds a parameter
    pub fn with_param<T: Serialize>(mut self, key: &str, value: T) -> Self {
        self.params
            .insert(key.to_string(), serde_json::to_value(value).unwrap());
        self
    }

    /// Gets a parameter
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.params
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Creates provider parameters from OpenAI parameters
    pub fn from_openai(openai_params: crate::params::OpenAiParams) -> Self {
        let mut params = HashMap::new();

        // Serialize the OpenAI params to a JSON value and then convert to HashMap
        if let Ok(json_value) = serde_json::to_value(&openai_params) {
            if let Ok(map) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(json_value)
            {
                params = map;
            }
        }

        Self { params }
    }
}

impl Default for ProviderParams {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP configuration
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Request timeout
    pub timeout: Option<Duration>,
    /// Connection timeout
    pub connect_timeout: Option<Duration>,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Proxy settings
    pub proxy: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            timeout: Some(Duration::from_secs(30)),
            connect_timeout: Some(Duration::from_secs(10)),
            headers: HashMap::new(),
            proxy: None,
            user_agent: Some("siumai/0.1.0".to_string()),
        }
    }
}

/// Message role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Developer, // Developer role for system-level instructions
    Tool,
}

/// Message content - supports multimodality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageContent {
    /// Plain text
    Text(String),
    /// Multimodal content
    MultiModal(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract text content if available
    pub fn text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(text) => Some(text),
            MessageContent::MultiModal(parts) => {
                // Return the first text part found
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        return Some(text);
                    }
                }
                None
            }
        }
    }

    /// Extract all text content
    pub fn all_text(&self) -> String {
        match self {
            MessageContent::Text(text) => text.clone(),
            MessageContent::MultiModal(parts) => {
                let mut result = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        if !result.is_empty() {
                            result.push(' ');
                        }
                        result.push_str(text);
                    }
                }
                result
            }
        }
    }
}

/// Content part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        image_url: String,
        detail: Option<String>,
    },
    Audio {
        audio_url: String,
        format: String,
    },
}

/// Cache control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheControl {
    /// Ephemeral cache
    Ephemeral,
    /// Persistent cache
    Persistent { ttl: Option<Duration> },
}

/// Message metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// Message ID
    pub id: Option<String>,
    /// Timestamp
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// Cache control (Anthropic-specific)
    pub cache_control: Option<CacheControl>,
    /// Custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for MessageMetadata {
    fn default() -> Self {
        Self {
            id: None,
            timestamp: None,
            cache_control: None,
            custom: HashMap::new(),
        }
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role
    pub role: MessageRole,
    /// Content
    pub content: MessageContent,
    /// Message metadata
    pub metadata: MessageMetadata,
    /// Tool calls
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Creates a user message
    pub fn user<S: Into<String>>(content: S) -> ChatMessageBuilder {
        ChatMessageBuilder::user(content)
    }

    /// Creates a system message
    pub fn system<S: Into<String>>(content: S) -> ChatMessageBuilder {
        ChatMessageBuilder::system(content)
    }

    /// Creates an assistant message
    pub fn assistant<S: Into<String>>(content: S) -> ChatMessageBuilder {
        ChatMessageBuilder::assistant(content)
    }

    /// Creates a developer message
    pub fn developer<S: Into<String>>(content: S) -> ChatMessageBuilder {
        ChatMessageBuilder::developer(content)
    }

    /// Creates a tool message
    pub fn tool<S: Into<String>>(content: S, tool_call_id: S) -> ChatMessageBuilder {
        ChatMessageBuilder::tool(content, tool_call_id)
    }

    /// Gets the text content of the message
    pub fn content_text(&self) -> Option<&str> {
        match &self.content {
            MessageContent::Text(text) => Some(text),
            MessageContent::MultiModal(parts) => parts.iter().find_map(|part| {
                if let ContentPart::Text { text } = part {
                    Some(text.as_str())
                } else {
                    None
                }
            }),
        }
    }
}

/// Chat message builder
#[derive(Debug, Clone)]
pub struct ChatMessageBuilder {
    role: MessageRole,
    content: Option<MessageContent>,
    metadata: MessageMetadata,
    tool_calls: Option<Vec<ToolCall>>,
    tool_call_id: Option<String>,
}

impl ChatMessageBuilder {
    /// Creates a user message builder
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a system message builder
    pub fn system<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::System,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates an assistant message builder
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a developer message builder
    pub fn developer<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Developer,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a tool message builder
    pub fn tool<S: Into<String>>(content: S, tool_call_id: S) -> Self {
        Self {
            role: MessageRole::Tool,
            content: Some(MessageContent::Text(content.into())),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Sets cache control
    pub fn cache_control(mut self, cache: CacheControl) -> Self {
        self.metadata.cache_control = Some(cache);
        self
    }

    /// Adds image content
    pub fn with_image(mut self, image_url: String, detail: Option<String>) -> Self {
        let image_part = ContentPart::Image { image_url, detail };

        match self.content {
            Some(MessageContent::Text(text)) => {
                self.content = Some(MessageContent::MultiModal(vec![
                    ContentPart::Text { text },
                    image_part,
                ]));
            }
            Some(MessageContent::MultiModal(ref mut parts)) => {
                parts.push(image_part);
            }
            None => {
                self.content = Some(MessageContent::MultiModal(vec![image_part]));
            }
        }

        self
    }

    /// Adds tool calls
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Builds the message
    pub fn build(self) -> ChatMessage {
        ChatMessage {
            role: self.role,
            content: self.content.unwrap_or(MessageContent::Text(String::new())),
            metadata: self.metadata,
            tool_calls: self.tool_calls,
            tool_call_id: self.tool_call_id,
        }
    }
}

// Tool calling types
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

/// Web search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchConfig {
    /// Whether web search is enabled
    pub enabled: bool,
    /// Maximum number of search results to retrieve
    pub max_results: Option<u32>,
    /// Search context size for providers that support it
    pub context_size: Option<WebSearchContextSize>,
    /// Custom search prompt for result integration
    pub search_prompt: Option<String>,
    /// Web search implementation strategy
    pub strategy: WebSearchStrategy,
    /// Provider-specific search parameters
    pub provider_params: HashMap<String, serde_json::Value>,
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_results: Some(5),
            context_size: None,
            search_prompt: None,
            strategy: WebSearchStrategy::Auto,
            provider_params: HashMap::new(),
        }
    }
}

/// Web search context size for providers that support it
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSearchContextSize {
    Small,
    Medium,
    Large,
}

/// Web search implementation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSearchStrategy {
    /// Automatically choose the best strategy for the provider
    Auto,
    /// Use provider's built-in search tools (OpenAI Responses API, xAI Live Search)
    BuiltIn,
    /// Use provider's web search tool (Anthropic web_search tool)
    Tool,
    /// Use external search API and inject results into context
    External,
}

/// Web search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchResult {
    /// Search result title
    pub title: String,
    /// Search result URL
    pub url: String,
    /// Search result snippet/description
    pub snippet: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
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

/// Chat request
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// List of messages
    pub messages: Vec<ChatMessage>,
    /// Tools available for the model to call
    pub tools: Option<Vec<Tool>>,
    /// Common parameters
    pub common_params: CommonParams,
    /// Provider-specific parameters
    pub provider_params: Option<ProviderParams>,
    /// HTTP configuration
    pub http_config: Option<HttpConfig>,
}

impl ChatRequest {
    /// Creates a request builder
    pub fn builder() -> ChatRequestBuilder {
        ChatRequestBuilder::new()
    }
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            common_params: CommonParams::default(),
            provider_params: None,
            http_config: None,
        }
    }
}

/// Chat request builder
#[derive(Debug, Clone)]
pub struct ChatRequestBuilder {
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    common_params: CommonParams,
    provider_params: Option<ProviderParams>,
    http_config: Option<HttpConfig>,
}

impl ChatRequestBuilder {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            common_params: CommonParams::default(),
            provider_params: None,
            http_config: None,
        }
    }

    /// Adds a message
    pub fn message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Adds multiple messages
    pub fn messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Sets tools
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Adds a tool
    pub fn tool(mut self, tool: Tool) -> Self {
        if self.tools.is_none() {
            self.tools = Some(Vec::new());
        }
        self.tools.as_mut().unwrap().push(tool);
        self
    }

    /// Sets common parameters
    pub fn common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Sets provider parameters
    pub fn provider_params(mut self, params: ProviderParams) -> Self {
        self.provider_params = Some(params);
        self
    }

    /// Builds the request
    pub fn build(self) -> ChatRequest {
        ChatRequest {
            messages: self.messages,
            tools: self.tools,
            common_params: self.common_params,
            provider_params: self.provider_params,
            http_config: self.http_config,
        }
    }
}

/// Chat response
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Response content
    pub content: MessageContent,
    /// Tool calls
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Usage statistics
    pub usage: Option<Usage>,
    /// Finish reason
    pub finish_reason: Option<FinishReason>,
    /// Response metadata
    pub metadata: ResponseMetadata,
    /// Provider-specific data
    pub provider_data: HashMap<String, serde_json::Value>,
}

impl ChatResponse {
    /// Gets the text content
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            MessageContent::Text(text) => Some(text),
            MessageContent::MultiModal(parts) => parts.iter().find_map(|part| {
                if let ContentPart::Text { text } = part {
                    Some(text.as_str())
                } else {
                    None
                }
            }),
        }
    }

    /// Gets the text content (alias for compatibility)
    pub fn content_text(&self) -> Option<&str> {
        self.text()
    }

    /// Gets thinking/reasoning content if available
    pub fn thinking(&self) -> Option<&str> {
        self.provider_data
            .get("thinking")
            .and_then(|v| v.as_str())
            .or_else(|| self.provider_data.get("reasoning").and_then(|v| v.as_str()))
    }

    /// Gets all text content
    pub fn all_text(&self) -> String {
        match &self.content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::MultiModal(parts) => parts
                .iter()
                .filter_map(|part| {
                    if let ContentPart::Text { text } = part {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Checks if there are any tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .map_or(false, |calls| !calls.is_empty())
    }
}

/// Finish reason
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    /// Normal completion
    Stop,
    /// Reached maximum length
    Length,
    /// Tool calls
    ToolCalls,
    /// Content filtered
    ContentFilter,
    /// Function call (for compatibility)
    FunctionCall,
    /// Other reason
    Other(String),
}

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Response ID
    pub id: Option<String>,
    /// Model name
    pub model: Option<String>,
    /// Creation time
    pub created: Option<chrono::DateTime<chrono::Utc>>,
    /// Provider name
    pub provider: String,
    /// Request ID
    pub request_id: Option<String>,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens
    pub prompt_tokens: Option<u32>,
    /// Output tokens
    pub completion_tokens: Option<u32>,
    /// Total tokens
    pub total_tokens: Option<u32>,
    /// Reasoning tokens (OpenAI o1-specific)
    pub reasoning_tokens: Option<u32>,
    /// Cache hit tokens (Anthropic-specific)
    pub cache_hit_tokens: Option<u32>,
    /// Cache creation tokens (Anthropic-specific)
    pub cache_creation_tokens: Option<u32>,
}

impl Usage {
    /// Merges usage statistics
    pub fn merge(&mut self, other: &Usage) {
        if let Some(other_prompt) = other.prompt_tokens {
            self.prompt_tokens = Some(self.prompt_tokens.unwrap_or(0) + other_prompt);
        }
        if let Some(other_completion) = other.completion_tokens {
            self.completion_tokens = Some(self.completion_tokens.unwrap_or(0) + other_completion);
        }
        if let Some(other_total) = other.total_tokens {
            self.total_tokens = Some(self.total_tokens.unwrap_or(0) + other_total);
        }
        if let Some(other_reasoning) = other.reasoning_tokens {
            self.reasoning_tokens = Some(self.reasoning_tokens.unwrap_or(0) + other_reasoning);
        }
    }
}

// Audio-related types
/// Text-to-speech request
#[derive(Debug, Clone)]
pub struct TtsRequest {
    /// Text to convert to speech
    pub text: String,
    /// Voice to use (provider-specific)
    pub voice: Option<String>,
    /// Audio format (mp3, wav, etc.)
    pub format: Option<String>,
    /// Speech speed (0.25 to 4.0)
    pub speed: Option<f32>,
    /// Audio quality/model
    pub model: Option<String>,
    /// Additional provider-specific parameters
    pub extra_params: std::collections::HashMap<String, serde_json::Value>,
}

impl TtsRequest {
    /// Create a new TTS request with text
    pub fn new(text: String) -> Self {
        Self {
            text,
            voice: None,
            format: None,
            speed: None,
            model: None,
            extra_params: std::collections::HashMap::new(),
        }
    }

    /// Set the voice
    pub fn with_voice(mut self, voice: String) -> Self {
        self.voice = Some(voice);
        self
    }

    /// Set the audio format
    pub fn with_format(mut self, format: String) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the speech speed
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }
}

/// Text-to-speech response
#[derive(Debug, Clone)]
pub struct TtsResponse {
    /// Generated audio data
    pub audio_data: Vec<u8>,
    /// Audio format
    pub format: String,
    /// Duration in seconds
    pub duration: Option<f32>,
    /// Sample rate
    pub sample_rate: Option<u32>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Speech-to-text request
#[derive(Debug, Clone)]
pub struct SttRequest {
    /// Audio data
    pub audio_data: Option<Vec<u8>>,
    /// File path (alternative to audio_data)
    pub file_path: Option<String>,
    /// Audio format
    pub format: Option<String>,
    /// Language code (e.g., "en-US")
    pub language: Option<String>,
    /// Model to use
    pub model: Option<String>,
    /// Enable word-level timestamps
    pub timestamp_granularities: Option<Vec<String>>,
    /// Additional provider-specific parameters
    pub extra_params: std::collections::HashMap<String, serde_json::Value>,
}

impl SttRequest {
    /// Create STT request from audio data
    pub fn from_audio(audio_data: Vec<u8>) -> Self {
        Self {
            audio_data: Some(audio_data),
            file_path: None,
            format: None,
            language: None,
            model: None,
            timestamp_granularities: None,
            extra_params: std::collections::HashMap::new(),
        }
    }

    /// Create STT request from file path
    pub fn from_file(file_path: String) -> Self {
        Self {
            audio_data: None,
            file_path: Some(file_path),
            format: None,
            language: None,
            model: None,
            timestamp_granularities: None,
            extra_params: std::collections::HashMap::new(),
        }
    }
}

/// Speech-to-text response
#[derive(Debug, Clone)]
pub struct SttResponse {
    /// Transcribed text
    pub text: String,
    /// Language detected
    pub language: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f32>,
    /// Word-level timestamps
    pub words: Option<Vec<WordTimestamp>>,
    /// Duration of audio in seconds
    pub duration: Option<f32>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Word-level timestamp information
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    /// The word
    pub word: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Confidence score for this word
    pub confidence: Option<f32>,
}

/// Audio translation request (speech to English text)
#[derive(Debug, Clone)]
pub struct AudioTranslationRequest {
    /// Audio data
    pub audio_data: Option<Vec<u8>>,
    /// File path (alternative to audio_data)
    pub file_path: Option<String>,
    /// Audio format
    pub format: Option<String>,
    /// Model to use
    pub model: Option<String>,
    /// Additional provider-specific parameters
    pub extra_params: std::collections::HashMap<String, serde_json::Value>,
}

impl AudioTranslationRequest {
    /// Create translation request from audio data
    pub fn from_audio(audio_data: Vec<u8>) -> Self {
        Self {
            audio_data: Some(audio_data),
            file_path: None,
            format: None,
            model: None,
            extra_params: std::collections::HashMap::new(),
        }
    }

    /// Create translation request from file path
    pub fn from_file(file_path: String) -> Self {
        Self {
            audio_data: None,
            file_path: Some(file_path),
            format: None,
            model: None,
            extra_params: std::collections::HashMap::new(),
        }
    }
}

/// Voice information
#[derive(Debug, Clone)]
pub struct VoiceInfo {
    /// Voice ID/name
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Voice description
    pub description: Option<String>,
    /// Language code
    pub language: Option<String>,
    /// Gender (male, female, neutral)
    pub gender: Option<String>,
    /// Voice category (standard, premium, neural, etc.)
    pub category: Option<String>,
}

/// Language information
#[derive(Debug, Clone)]
pub struct LanguageInfo {
    /// Language code (e.g., "en-US")
    pub code: String,
    /// Human-readable name
    pub name: String,
    /// Whether this language supports transcription
    pub supports_transcription: bool,
    /// Whether this language supports translation
    pub supports_translation: bool,
}

/// Audio features that providers can support
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AudioFeature {
    /// Basic text-to-speech conversion
    TextToSpeech,
    /// Streaming text-to-speech conversion
    StreamingTTS,
    /// Basic speech-to-text conversion
    SpeechToText,
    /// Audio translation (speech to English text)
    AudioTranslation,
    /// Real-time audio processing
    RealtimeProcessing,
    /// Speaker diarization (identifying different speakers)
    SpeakerDiarization,
    /// Character-level timing information
    CharacterTiming,
    /// Audio event detection (laughter, applause, etc.)
    AudioEventDetection,
    /// Voice cloning capabilities
    VoiceCloning,
    /// Audio enhancement and noise reduction
    AudioEnhancement,
    /// Multi-modal audio-visual processing
    MultimodalAudio,
}

// Stream types
use futures::Stream;
use std::pin::Pin;

/// Audio stream for streaming TTS
pub type AudioStream = Pin<Box<dyn Stream<Item = Result<AudioStreamEvent, LlmError>> + Send>>;

/// Audio stream events
#[derive(Debug, Clone)]
pub enum AudioStreamEvent {
    /// Audio data chunk
    AudioData { data: Vec<u8> },
    /// Stream metadata
    Metadata {
        format: String,
        sample_rate: Option<u32>,
        duration: Option<f32>,
    },
    /// Stream completed
    Complete,
    /// Stream error
    Error { error: String },
}

// Image generation types
/// Image generation request
#[derive(Debug, Clone, Default)]
pub struct ImageGenerationRequest {
    /// Text prompt describing the image
    pub prompt: String,
    /// Negative prompt (what to avoid)
    pub negative_prompt: Option<String>,
    /// Image size (e.g., "1024x1024")
    pub size: Option<String>,
    /// Number of images to generate
    pub count: u32,
    /// Model to use for generation
    pub model: Option<String>,
    /// Quality setting
    pub quality: Option<String>,
    /// Style setting
    pub style: Option<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Number of inference steps
    pub steps: Option<u32>,
    /// Guidance scale
    pub guidance_scale: Option<f32>,
    /// Whether to enhance the prompt
    pub enhance_prompt: Option<bool>,
    /// Response format (url or b64_json)
    pub response_format: Option<String>,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Image edit request
#[derive(Debug, Clone)]
pub struct ImageEditRequest {
    /// Original image data
    pub image: Vec<u8>,
    /// Mask image data (optional)
    pub mask: Option<Vec<u8>>,
    /// Text prompt for editing
    pub prompt: String,
    /// Number of images to generate
    pub count: Option<u32>,
    /// Image size
    pub size: Option<String>,
    /// Response format
    pub response_format: Option<String>,
    /// Additional parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Image variation request
#[derive(Debug, Clone)]
pub struct ImageVariationRequest {
    /// Original image data
    pub image: Vec<u8>,
    /// Number of variations to generate
    pub count: Option<u32>,
    /// Image size
    pub size: Option<String>,
    /// Response format
    pub response_format: Option<String>,
    /// Additional parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Image generation response
#[derive(Debug, Clone)]
pub struct ImageGenerationResponse {
    /// Generated images
    pub images: Vec<GeneratedImage>,
    /// Request metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A single generated image
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Image URL (if response_format is "url")
    pub url: Option<String>,
    /// Base64 encoded image data (if response_format is "b64_json")
    pub b64_json: Option<String>,
    /// Image format
    pub format: Option<String>,
    /// Image dimensions
    pub width: Option<u32>,
    /// Image height
    pub height: Option<u32>,
    /// Revised prompt (if prompt was enhanced)
    pub revised_prompt: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Embedding response
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// List of embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Model used for embeddings
    pub model: String,
    /// Usage information
    pub usage: Option<EmbeddingUsage>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Embedding usage information
#[derive(Debug, Clone)]
pub struct EmbeddingUsage {
    /// Number of prompt tokens
    pub prompt_tokens: u32,
    /// Total tokens processed
    pub total_tokens: u32,
}

// File management types
/// File upload request
#[derive(Debug, Clone)]
pub struct FileUploadRequest {
    /// File content as bytes
    pub content: Vec<u8>,
    /// Original filename
    pub filename: String,
    /// MIME type
    pub mime_type: Option<String>,
    /// Purpose of the file (e.g., "assistants", "fine-tune")
    pub purpose: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// File object metadata
#[derive(Debug, Clone)]
pub struct FileObject {
    /// File ID
    pub id: String,
    /// Original filename
    pub filename: String,
    /// File size in bytes
    pub bytes: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// File purpose
    pub purpose: String,
    /// File status
    pub status: String,
    /// MIME type
    pub mime_type: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// File list query parameters
#[derive(Debug, Clone, Default)]
pub struct FileListQuery {
    /// Filter by purpose
    pub purpose: Option<String>,
    /// Limit number of results
    pub limit: Option<u32>,
    /// Pagination cursor
    pub after: Option<String>,
    /// Sort order
    pub order: Option<String>,
}

/// File list response
#[derive(Debug, Clone)]
pub struct FileListResponse {
    /// List of files
    pub files: Vec<FileObject>,
    /// Whether there are more results
    pub has_more: bool,
    /// Next page cursor
    pub next_cursor: Option<String>,
}

/// File deletion response
#[derive(Debug, Clone)]
pub struct FileDeleteResponse {
    /// File ID that was deleted
    pub id: String,
    /// Whether deletion was successful
    pub deleted: bool,
}

// Moderation types
/// Moderation request
#[derive(Debug, Clone)]
pub struct ModerationRequest {
    /// Input text to moderate
    pub input: String,
    /// Model to use for moderation
    pub model: Option<String>,
}

/// Moderation response
#[derive(Debug, Clone)]
pub struct ModerationResponse {
    /// Moderation results
    pub results: Vec<ModerationResult>,
    /// Model used
    pub model: String,
}

/// Individual moderation result
#[derive(Debug, Clone)]
pub struct ModerationResult {
    /// Whether content was flagged
    pub flagged: bool,
    /// Category scores
    pub categories: HashMap<String, bool>,
    /// Category confidence scores
    pub category_scores: HashMap<String, f32>,
}

// Model information types
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

// Completion types
/// Text completion request
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Input prompt
    pub prompt: String,
    /// Model to use
    pub model: Option<String>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for randomness
    pub temperature: Option<f32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<u32>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Number of completions to generate
    pub n: Option<u32>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Additional parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Text completion response
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// Generated text
    pub text: String,
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Usage information
    pub usage: Option<Usage>,
    /// Model used
    pub model: Option<String>,
}

/// Completion stream for streaming completions
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = Result<CompletionStreamEvent, LlmError>> + Send>>;

/// Completion stream events
#[derive(Debug, Clone)]
pub enum CompletionStreamEvent {
    /// Text delta
    TextDelta { delta: String },
    /// Completion finished
    Complete { response: CompletionResponse },
    /// Stream error
    Error { error: String },
}

// Other placeholder types - to be completed in subsequent implementations
pub type VisionRequest = ();
pub type VisionResponse = ();
pub type ImageGenRequest = (); // Keep for backward compatibility
pub type ImageResponse = (); // Keep for backward compatibility
pub type JsonSchema = ();
pub type StructuredResponse = ();
pub type BatchRequest = ();
pub type BatchResponse = ();
pub type CacheConfig = ();
pub type ThinkingResponse = ();
pub type SearchConfig = ();
pub type ExecutionResponse = ();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user;

    #[test]
    fn test_message_creation() {
        // Test builder pattern
        let msg = ChatMessage::user("Hello").build();
        assert_eq!(msg.role, MessageRole::User);

        if let MessageContent::Text(text) = msg.content {
            assert_eq!(text, "Hello");
        } else {
            panic!("Expected text content");
        }

        // Test direct macro usage
        let direct_msg = user!("Hello");
        assert_eq!(direct_msg.role, MessageRole::User);

        if let MessageContent::Text(text) = direct_msg.content {
            assert_eq!(text, "Hello");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_provider_params() {
        let params = ProviderParams::new()
            .with_param("temperature", 0.7)
            .with_param("max_tokens", 1000);

        let temp: Option<f64> = params.get("temperature");
        assert_eq!(temp, Some(0.7));

        let tokens: Option<u32> = params.get("max_tokens");
        assert_eq!(tokens, Some(1000));
    }
}

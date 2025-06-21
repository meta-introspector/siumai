//! Core Trait Definitions
//!
//! Defines the core capability traits for the LLM library, using a capability separation design pattern.
//! This follows the design principles of trait separation, parameter sharing, and provider-specific extensions.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::types::*;

/// Core chat capability trait - the most fundamental LLM capability.
///
/// This trait provides only the essential chat functionality that all LLM providers must implement.
/// It follows the single responsibility principle by focusing solely on core chat operations.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/guides/tools
/// - Anthropic: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
/// - xAI: https://docs.x.ai/docs/guides/function-calling
#[async_trait]
pub trait ChatCapability: Send + Sync {
    /// Sends a chat request to the provider with a sequence of messages.
    ///
    /// # Arguments
    /// * `messages` - The conversation history as a list of chat messages
    ///
    /// # Returns
    /// The provider's response or an error
    ///
    /// # Default Implementation
    /// By default, this calls `chat_with_tools` with no tools.
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a chat request to the provider with a sequence of messages and tools.
    ///
    /// # Arguments
    /// * `messages` - The conversation history as a list of chat messages
    /// * `tools` - Optional list of tools to use in the chat
    ///
    /// # Returns
    /// The provider's response or an error
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError>;

    /// Sends a streaming chat request to the provider.
    ///
    /// # Arguments
    /// * `messages` - The conversation history as a list of chat messages
    /// * `tools` - Optional list of tools to use in the chat
    ///
    /// # Returns
    /// A stream of chat events
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError>;
}

/// Extended chat capabilities providing convenience methods and advanced features.
///
/// This trait extends the core ChatCapability with useful convenience methods
/// that are commonly needed but not essential for basic chat functionality.
/// It follows the interface segregation principle by separating optional features.
#[async_trait]
pub trait ChatExtensions: ChatCapability {
    /// Get current memory contents if provider supports memory.
    ///
    /// # Returns
    /// Optional list of messages representing the current memory state
    async fn memory_contents(&self) -> Result<Option<Vec<ChatMessage>>, LlmError> {
        Ok(None)
    }

    /// Summarizes a conversation history into a concise 2-3 sentence summary.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages to summarize
    ///
    /// # Returns
    /// A string containing the summary
    async fn summarize_history(&self, messages: Vec<ChatMessage>) -> Result<String, LlmError> {
        let prompt = format!(
            "Summarize in 2-3 sentences:\n{}",
            messages
                .iter()
                .map(|m| format!("{:?}: {}", m.role, m.content_text().unwrap_or("")))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let request_messages = vec![ChatMessage::user(prompt).build()];
        let response = self.chat(request_messages).await?;

        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in summary response".to_string()))
            .map(|s| s.to_string())
    }

    /// Simple text completion - just send a prompt and get a response.
    ///
    /// # Arguments
    /// * `prompt` - The text prompt to send
    ///
    /// # Returns
    /// The response text
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = quick_openai().await?;
    /// let response = client.ask("What is the capital of France?".to_string()).await?;
    /// println!("{}", response);
    /// # Ok(())
    /// # }
    /// ```
    async fn ask(&self, prompt: String) -> Result<String, LlmError> {
        let message = ChatMessage::user(prompt).build();
        let response = self.chat(vec![message]).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))
            .map(|s| s.to_string())
    }

    /// Simple system-prompted completion.
    ///
    /// # Arguments
    /// * `system_prompt` - The system instruction
    /// * `user_prompt` - The user prompt
    ///
    /// # Returns
    /// The response text
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = quick_openai().await?;
    /// let response = client.ask_with_system(
    ///     "You are a helpful assistant that responds in JSON".to_string(),
    ///     "List 3 colors".to_string()
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn ask_with_system(
        &self,
        system_prompt: String,
        user_prompt: String,
    ) -> Result<String, LlmError> {
        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(user_prompt).build(),
        ];
        let response = self.chat(messages).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))
            .map(|s| s.to_string())
    }

    /// Continue a conversation with a new user message.
    ///
    /// # Arguments
    /// * `conversation` - Existing conversation messages
    /// * `new_message` - New user message to add
    ///
    /// # Returns
    /// The response and updated conversation
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = quick_openai().await?;
    /// let mut conversation = vec![
    ///     ChatMessage::system("You are a helpful assistant").build()
    /// ];
    ///
    /// let (response, updated_conversation) = client
    ///     .continue_conversation(conversation, "Hello!".to_string())
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn continue_conversation(
        &self,
        mut conversation: Vec<ChatMessage>,
        new_message: String,
    ) -> Result<(String, Vec<ChatMessage>), LlmError> {
        conversation.push(ChatMessage::user(new_message).build());

        let response = self.chat(conversation.clone()).await?;
        let response_text = response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))?
            .to_string();

        conversation.push(ChatMessage::assistant(response_text.clone()).build());

        Ok((response_text, conversation))
    }

    /// Translate text to another language.
    ///
    /// # Arguments
    /// * `text` - Text to translate
    /// * `target_language` - Target language (e.g., "French", "Spanish", "中文")
    ///
    /// # Returns
    /// Translated text
    async fn translate(
        &self,
        text: String,
        target_language: String,
    ) -> Result<String, LlmError> {
        let prompt = format!(
            "Translate the following text to {}: {}",
            target_language,
            text
        );
        self.ask(prompt).await
    }

    /// Explain a concept in simple terms.
    ///
    /// # Arguments
    /// * `concept` - The concept to explain
    /// * `audience` - Target audience (e.g., "a 5-year-old", "a beginner programmer")
    ///
    /// # Returns
    /// Simple explanation
    async fn explain(
        &self,
        concept: String,
        audience: Option<String>,
    ) -> Result<String, LlmError> {
        let audience_str = audience
            .map(|a| format!(" to {}", a))
            .unwrap_or_else(|| " in simple terms".to_string());

        let prompt = format!("Explain {}{}", concept, audience_str);
        self.ask(prompt).await
    }

    /// Generate creative content based on a prompt.
    ///
    /// # Arguments
    /// * `content_type` - Type of content (e.g., "story", "poem", "email")
    /// * `prompt` - Creative prompt
    ///
    /// # Returns
    /// Generated content
    async fn generate(
        &self,
        content_type: String,
        prompt: String,
    ) -> Result<String, LlmError> {
        let system_prompt = format!(
            "You are a creative writer. Generate a {} based on the user's prompt.",
            content_type
        );
        self.ask_with_system(system_prompt, prompt).await
    }
}

/// Automatic implementation of ChatExtensions for all types that implement ChatCapability
impl<T: ChatCapability> ChatExtensions for T {}

/// Unified audio processing capability interface.
///
/// This interface provides a single entry point for all audio-related functionality,
/// including text-to-speech, speech-to-text, audio translation, and real-time processing.
/// Use the `supported_features` property to discover which features are available.
///
/// # Feature Discovery
/// Always check `supported_features()` before calling specific methods to ensure
/// the provider supports the desired functionality.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/guides/speech-to-text
/// - Google: https://cloud.google.com/speech-to-text/docs
#[async_trait]
pub trait AudioCapability {
    /// Get all audio features supported by this provider.
    fn supported_features(&self) -> &[AudioFeature];

    /// Convert text to speech with full configuration support.
    ///
    /// # Arguments
    /// * `request` - The TTS request configuration
    ///
    /// # Returns
    /// Audio response with generated audio data
    ///
    /// # Errors
    /// Returns `UnsupportedOperation` if text-to-speech is not supported.
    /// Check `supported_features()` first.
    async fn text_to_speech(&self, _request: TtsRequest) -> Result<TtsResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Text-to-speech not supported by this provider".to_string(),
        ))
    }

    /// Convert text to speech with streaming output.
    ///
    /// # Arguments
    /// * `request` - The TTS request configuration
    ///
    /// # Returns
    /// Stream of audio events
    ///
    /// # Errors
    /// Returns `UnsupportedOperation` if streaming TTS is not supported.
    async fn text_to_speech_stream(&self, _request: TtsRequest) -> Result<AudioStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming text-to-speech not supported by this provider".to_string(),
        ))
    }

    /// Convert speech to text with full configuration support.
    ///
    /// # Arguments
    /// * `request` - The STT request configuration
    ///
    /// # Returns
    /// Text response with transcribed content
    ///
    /// # Errors
    /// Returns `UnsupportedOperation` if speech-to-text is not supported.
    async fn speech_to_text(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Speech-to-text not supported by this provider".to_string(),
        ))
    }

    /// Translate audio to English text.
    ///
    /// # Arguments
    /// * `request` - The audio translation request
    ///
    /// # Returns
    /// Text response with translated content
    ///
    /// # Errors
    /// Returns `UnsupportedOperation` if audio translation is not supported.
    async fn translate_audio(
        &self,
        _request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Audio translation not supported by this provider".to_string(),
        ))
    }

    /// Get available voices for this provider.
    ///
    /// # Returns
    /// List of available voice configurations
    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Voice listing not supported by this provider".to_string(),
        ))
    }

    /// Get supported languages for transcription and translation.
    ///
    /// # Returns
    /// List of supported language information
    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Language listing not supported by this provider".to_string(),
        ))
    }

    /// Get supported input/output audio formats.
    ///
    /// # Returns
    /// List of supported audio format strings
    fn get_supported_audio_formats(&self) -> Vec<String> {
        vec!["mp3".to_string(), "wav".to_string(), "ogg".to_string()] // Default formats
    }

    // === Convenience Methods ===

    /// Simple text-to-speech conversion (convenience method).
    ///
    /// # Arguments
    /// * `text` - The text to convert to speech
    ///
    /// # Returns
    /// Audio data as bytes
    async fn speech(&self, text: String) -> Result<Vec<u8>, LlmError> {
        let request = TtsRequest::new(text);
        let response = self.text_to_speech(request).await?;
        Ok(response.audio_data)
    }

    /// Simple audio transcription (convenience method).
    ///
    /// # Arguments
    /// * `audio` - The audio data to transcribe
    ///
    /// # Returns
    /// Transcribed text
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, LlmError> {
        let request = SttRequest::from_audio(audio);
        let response = self.speech_to_text(request).await?;
        Ok(response.text)
    }

    /// Simple file transcription (convenience method).
    ///
    /// # Arguments
    /// * `file_path` - Path to the audio file
    ///
    /// # Returns
    /// Transcribed text
    async fn transcribe_file(&self, file_path: String) -> Result<String, LlmError> {
        let request = SttRequest::from_file(file_path);
        let response = self.speech_to_text(request).await?;
        Ok(response.text)
    }

    /// Simple audio translation (convenience method).
    ///
    /// # Arguments
    /// * `audio` - The audio data to translate
    ///
    /// # Returns
    /// Translated text in English
    async fn translate(&self, audio: Vec<u8>) -> Result<String, LlmError> {
        let request = AudioTranslationRequest::from_audio(audio);
        let response = self.translate_audio(request).await?;
        Ok(response.text)
    }

    /// Simple file translation (convenience method).
    ///
    /// # Arguments
    /// * `file_path` - Path to the audio file
    ///
    /// # Returns
    /// Translated text in English
    async fn translate_file(&self, file_path: String) -> Result<String, LlmError> {
        let request = AudioTranslationRequest::from_file(file_path);
        let response = self.translate_audio(request).await?;
        Ok(response.text)
    }
}

/// Vision capability trait for image analysis and generation.
///
/// This trait provides image understanding and generation capabilities.
/// Different providers may support different aspects of vision processing.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/guides/vision
/// - Google: https://cloud.google.com/vision/docs
/// - Anthropic: https://docs.anthropic.com/en/docs/vision
#[async_trait]
pub trait VisionCapability {
    /// Analyzes an image with optional text prompt.
    ///
    /// # Arguments
    /// * `request` - The vision analysis request
    ///
    /// # Returns
    /// Analysis results including descriptions, detected objects, etc.
    async fn analyze_image(&self, request: VisionRequest) -> Result<VisionResponse, LlmError>;

    /// Generates an image from a text prompt.
    ///
    /// # Arguments
    /// * `request` - The image generation request
    ///
    /// # Returns
    /// Generated image data and metadata
    async fn generate_image(&self, request: ImageGenRequest) -> Result<ImageResponse, LlmError>;

    /// Get supported image formats for input.
    fn get_supported_input_formats(&self) -> Vec<String> {
        vec!["jpeg".to_string(), "png".to_string(), "webp".to_string()]
    }

    /// Get supported image formats for output.
    fn get_supported_output_formats(&self) -> Vec<String> {
        vec!["png".to_string(), "jpeg".to_string()]
    }
}

/// Embedding capability trait for vector embeddings.
///
/// This trait provides text embedding functionality for semantic search,
/// similarity comparison, and other vector-based operations.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/guides/embeddings
/// - Google: https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings
/// - Anthropic: Currently not supported
#[async_trait]
pub trait EmbeddingCapability {
    /// Generates embeddings for the given input texts.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    ///
    /// # Returns
    /// List of embedding vectors (one per input text)
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError>;

    /// Get the dimension of embeddings produced by this provider.
    ///
    /// # Returns
    /// Number of dimensions in the embedding vectors
    fn embedding_dimension(&self) -> usize;

    /// Get the maximum number of tokens that can be embedded at once.
    ///
    /// # Returns
    /// Maximum token limit per embedding request
    fn max_tokens_per_embedding(&self) -> usize {
        8192 // Common default
    }

    /// Get supported embedding models for this provider.
    ///
    /// # Returns
    /// List of available embedding model names
    fn supported_embedding_models(&self) -> Vec<String> {
        vec!["default".to_string()]
    }
}

/// Image generation capability trait.
///
/// This trait provides image generation, editing, and variation creation
/// capabilities across different providers.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/api-reference/images
/// - Stability AI: https://platform.stability.ai/docs/api-reference
#[async_trait]
pub trait ImageGenerationCapability {
    /// Generate images from text prompts.
    ///
    /// # Arguments
    /// * `request` - The image generation request
    ///
    /// # Returns
    /// Generated images with metadata
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;

    /// Edit an existing image based on a prompt.
    ///
    /// # Arguments
    /// * `request` - The image editing request
    ///
    /// # Returns
    /// Edited images with metadata
    async fn edit_image(
        &self,
        _request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Image editing not supported by this provider".to_string(),
        ))
    }

    /// Create variations of an existing image.
    ///
    /// # Arguments
    /// * `request` - The image variation request
    ///
    /// # Returns
    /// Image variations with metadata
    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Image variations not supported by this provider".to_string(),
        ))
    }

    /// Get supported image sizes for this provider.
    fn get_supported_sizes(&self) -> Vec<String>;

    /// Get supported response formats for this provider.
    fn get_supported_formats(&self) -> Vec<String>;

    /// Check if the provider supports image editing.
    fn supports_image_editing(&self) -> bool {
        false
    }

    /// Check if the provider supports image variations.
    fn supports_image_variations(&self) -> bool {
        false
    }

    /// Simple image generation (convenience method).
    ///
    /// # Arguments
    /// * `prompt` - Text description of the image to generate
    /// * `size` - Optional image size (e.g., "1024x1024")
    /// * `count` - Number of images to generate (default: 1)
    ///
    /// # Returns
    /// List of image URLs or base64 data
    async fn generate_image(
        &self,
        prompt: String,
        size: Option<String>,
        count: Option<u32>,
    ) -> Result<Vec<String>, LlmError> {
        let request = ImageGenerationRequest {
            prompt,
            size,
            count: count.unwrap_or(1),
            ..Default::default()
        };

        let response = self.generate_images(request).await?;
        Ok(response
            .images
            .into_iter()
            .filter_map(|img| img.url)
            .collect())
    }
}

/// File management capability for uploading and managing files.
///
/// This interface provides a unified API for file operations across different
/// providers (OpenAI, Anthropic, etc.).
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/api-reference/files
/// - Anthropic: https://docs.anthropic.com/en/api/messages
#[async_trait]
pub trait FileManagementCapability {
    /// Upload a file to the provider's storage.
    ///
    /// # Arguments
    /// * `request` - The file upload request
    ///
    /// # Returns
    /// File object with metadata
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError>;

    /// List files with optional filtering.
    ///
    /// # Arguments
    /// * `query` - Optional query parameters for filtering
    ///
    /// # Returns
    /// Paginated list of files
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError>;

    /// Retrieve file metadata.
    ///
    /// # Arguments
    /// * `file_id` - The file identifier
    ///
    /// # Returns
    /// File object with metadata
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError>;

    /// Delete a file permanently.
    ///
    /// # Arguments
    /// * `file_id` - The file identifier
    ///
    /// # Returns
    /// Deletion confirmation
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError>;

    /// Get file content as bytes.
    ///
    /// # Arguments
    /// * `file_id` - The file identifier
    ///
    /// # Returns
    /// Raw file content
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError>;
}

/// Content moderation capability.
///
/// This trait provides content moderation functionality to check for
/// policy violations, harmful content, etc.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/api-reference/moderations
#[async_trait]
pub trait ModerationCapability {
    /// Moderate content for policy violations.
    ///
    /// # Arguments
    /// * `request` - The moderation request
    ///
    /// # Returns
    /// Moderation results with flagged categories
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse, LlmError>;

    /// Get supported moderation categories.
    ///
    /// # Returns
    /// List of moderation categories this provider supports
    fn supported_categories(&self) -> Vec<String> {
        vec![
            "hate".to_string(),
            "hate/threatening".to_string(),
            "harassment".to_string(),
            "harassment/threatening".to_string(),
            "self-harm".to_string(),
            "self-harm/intent".to_string(),
            "self-harm/instructions".to_string(),
            "sexual".to_string(),
            "sexual/minors".to_string(),
            "violence".to_string(),
            "violence/graphic".to_string(),
        ]
    }
}

/// Model listing capability.
///
/// This trait provides functionality to list and get information about
/// available models from the provider.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/api-reference/models
/// - Anthropic: Models are typically hardcoded
#[async_trait]
pub trait ModelListingCapability {
    /// Get available models from the provider.
    ///
    /// # Returns
    /// List of available models with metadata
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;

    /// Get detailed information about a specific model.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier
    ///
    /// # Returns
    /// Detailed model information
    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError>;

    /// Check if a model is available.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier
    ///
    /// # Returns
    /// Whether the model is available
    async fn is_model_available(&self, model_id: String) -> Result<bool, LlmError> {
        match self.get_model(model_id).await {
            Ok(_) => Ok(true),
            Err(LlmError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// Text completion capability (non-chat).
///
/// This trait provides traditional text completion functionality,
/// as opposed to conversational chat.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/api-reference/completions
#[async_trait]
pub trait CompletionCapability {
    /// Generate text completion from a prompt.
    ///
    /// # Arguments
    /// * `request` - The completion request
    ///
    /// # Returns
    /// Generated completion text
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;

    /// Generate streaming text completion.
    ///
    /// # Arguments
    /// * `request` - The completion request
    ///
    /// # Returns
    /// Stream of completion events
    async fn complete_stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming completion not supported by this provider".to_string(),
        ))
    }
}

/// OpenAI-specific capabilities.
///
/// This trait provides OpenAI-specific functionality that's not available
/// in other providers.
///
/// # API References
/// - OpenAI: https://platform.openai.com/docs/api-reference
#[async_trait]
pub trait OpenAiCapability {
    /// Chat with structured output using JSON schema.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages
    /// * `schema` - JSON schema for structured output
    ///
    /// # Returns
    /// Structured response matching the schema
    async fn chat_with_structured_output(
        &self,
        messages: Vec<ChatMessage>,
        schema: JsonSchema,
    ) -> Result<StructuredResponse, LlmError>;

    /// Create a batch processing job.
    ///
    /// # Arguments
    /// * `requests` - List of requests to process in batch
    ///
    /// # Returns
    /// Batch job information
    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchResponse, LlmError>;

    /// Use the Responses API instead of Chat Completions.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages
    /// * `tools` - Optional built-in tools (web search, file search, etc.)
    ///
    /// # Returns
    /// Response from the Responses API
    async fn chat_with_responses_api(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<OpenAiBuiltInTool>>,
    ) -> Result<ChatResponse, LlmError>;
}

/// Anthropic-specific capabilities.
#[async_trait]
pub trait AnthropicCapability {
    /// Caches prompts.
    async fn chat_with_cache(
        &self,
        request: ChatRequest,
        cache_config: CacheConfig,
    ) -> Result<ChatResponse, LlmError>;

    /// Thinking process.
    async fn chat_with_thinking(&self, request: ChatRequest) -> Result<ThinkingResponse, LlmError>;
}

/// Gemini-specific capabilities.
#[async_trait]
pub trait GeminiCapability {
    /// Search-augmented generation.
    async fn chat_with_search(
        &self,
        request: ChatRequest,
        search_config: SearchConfig,
    ) -> Result<ChatResponse, LlmError>;

    /// Code execution.
    async fn execute_code(
        &self,
        code: String,
        language: String,
    ) -> Result<ExecutionResponse, LlmError>;
}

/// Core provider trait.
pub trait LlmProvider: Send + Sync {
    /// Provider name.
    fn provider_name(&self) -> &'static str;

    /// List of supported models.
    fn supported_models(&self) -> Vec<String>;

    /// Gets capability information.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Gets the HTTP client.
    fn http_client(&self) -> &reqwest::Client;
}

/// Provider capability information.
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    pub chat: bool,
    pub audio: bool,
    pub vision: bool,
    pub tools: bool,
    pub embedding: bool,
    pub streaming: bool,
    pub file_management: bool,
    pub custom_features: HashMap<String, bool>,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            chat: false,
            audio: false,
            vision: false,
            tools: false,
            embedding: false,
            streaming: false,
            file_management: false,
            custom_features: HashMap::new(),
        }
    }
}

impl ProviderCapabilities {
    /// Creates new capability information.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables chat capability.
    pub fn with_chat(mut self) -> Self {
        self.chat = true;
        self
    }

    /// Enables audio capability.
    pub fn with_audio(mut self) -> Self {
        self.audio = true;
        self
    }

    /// Enables vision capability.
    pub fn with_vision(mut self) -> Self {
        self.vision = true;
        self
    }

    /// Enables tool capability.
    pub fn with_tools(mut self) -> Self {
        self.tools = true;
        self
    }

    /// Enables embedding capability.
    pub fn with_embedding(mut self) -> Self {
        self.embedding = true;
        self
    }

    /// Enables streaming.
    pub fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    /// Enables file management capability.
    pub fn with_file_management(mut self) -> Self {
        self.file_management = true;
        self
    }

    /// Adds a custom feature.
    pub fn with_custom_feature(mut self, name: impl Into<String>, enabled: bool) -> Self {
        self.custom_features.insert(name.into(), enabled);
        self
    }

    /// Checks if a feature is supported.
    pub fn supports(&self, feature: &str) -> bool {
        match feature {
            "chat" => self.chat,
            "audio" => self.audio,
            "vision" => self.vision,
            "tools" => self.tools,
            "embedding" => self.embedding,
            "streaming" => self.streaming,
            "file_management" => self.file_management,
            _ => self.custom_features.get(feature).copied().unwrap_or(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_capabilities() {
        let caps = ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_custom_feature("custom_feature", true);

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("custom_feature"));
        assert!(!caps.supports("audio"));
    }
}

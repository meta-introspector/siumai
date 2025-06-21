# Changelog

## [0.3.0] - 2025-06-21

### Added

- `ChatExtensions` trait with convenience methods (ask, translate, explain)
- Capability proxies: `AudioCapabilityProxy`, `EmbeddingCapabilityProxy`, `VisionCapabilityProxy`
- Static string methods (`user_static`, `system_static`) for zero-copy literals
- LRU response cache with configurable capacity
- `as_any()` method for type-safe client casting

### Fixed

- Streaming output JSON parsing errors caused by network packet truncation
- UTF-8 character handling in streaming responses across all providers
- Inconsistent streaming architecture between providers

### Changed

- **BREAKING**: Capability access returns proxies directly (no `Result<Proxy, Error>`)
- **BREAKING**: Capability checks are advisory only, never block operations
- Split `ChatCapability` into core functionality and extensions
- Improved error handling with better retry logic and HTTP status handling
- Optimized parameter validation and string processing performance
- Refactored streaming implementations with dedicated modules for better maintainability
- Added line/JSON buffering mechanisms to handle incomplete data chunks
- Unified streaming architecture across OpenAI, Anthropic, Ollama, and Gemini providers

### Removed

- `register_capability()` and `get_capability()` methods
- `with_capability()`, `with_audio()`, `with_embedding()` deprecated methods
- `FinishReason::FunctionCall` (use `ToolCalls` instead)
- Automatic capability warnings

## [0.2.0] - 2025-06-21

### Added

- Ollama provider support (chat, streaming, embeddings, model management)
- Multimodal support for vision-capable models
- `PartialEq` support for `MessageContent` and `ContentPart`

## [0.1.0] - 2025-06-20

### Added

- Initial release with unified LLM interface
- Providers: OpenAI, Anthropic Claude, Google Gemini, xAI, OpenRouter, DeepSeek
- Capabilities: Chat, Audio, Vision, Tools, Embeddings
- Streaming support and multimodal content
- Retry mechanisms and parameter validation
- Macros: `user!()`, `system!()`, `assistant!()`, `tool!()`

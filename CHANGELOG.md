# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-06-21

### Added

- **Ollama Provider Support**: Full integration with Ollama for local AI models
  - Chat and completion capabilities
  - Streaming support for real-time responses
  - Embedding generation with local models
  - Model management (list, pull, delete, copy)
  - Multimodal support for vision-capable models
- **Enhanced Examples**: Added `ollama_example.rs` and `ollama_advanced.rs`
- **Documentation**: Updated README with Ollama usage examples and configuration

### Changed

- Updated `MessageContent` and `ContentPart` to support `PartialEq` for testing
- Enhanced parameter validation to support Ollama-specific parameters
- Improved error handling for local model scenarios

## [0.1.0] - 2025-06-20

### Added

- Initial release of Siumai unified LLM interface library
- Support for multiple providers: OpenAI, Anthropic Claude, Google Gemini
- OpenAI-compatible providers: xAI, OpenRouter, DeepSeek
- Capability-based traits: Chat, Audio, Vision, Tools, Embeddings
- Two interface approaches: `Provider::*()` and `Siumai::builder()`
- Streaming support with async/await
- Multimodal content support (text, images, audio)
- Built-in retry mechanisms with exponential backoff
- Parameter validation and optimization
- Convenient macros: `user!()`, `system!()`, `assistant!()`, `tool!()`
- Comprehensive examples and documentation

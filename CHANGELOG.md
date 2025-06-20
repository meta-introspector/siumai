# Changelog

All notable changes to this project will be documented in this file.

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

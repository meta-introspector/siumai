# Siumai Examples

Practical examples for the Siumai Rust library, organized by learning path and use case.

## Quick Start

| I need to... | Go to |
|--------------|-------|
| **Get started quickly** | [quick_start.rs](01_getting_started/quick_start.rs) |
| **Learn basic chat** | [chat_basics.rs](02_core_features/chat_basics.rs) |
| **Compare providers** | [provider_comparison.rs](01_getting_started/provider_comparison.rs) |
| **Use streaming** | [streaming_chat.rs](02_core_features/streaming_chat.rs) |
| **Set up Ollama** | [basic_setup.rs](04_providers/ollama/basic_setup.rs) |
| **Handle errors** | [error_handling.rs](02_core_features/error_handling.rs) |
| **Build a chatbot** | [simple_chatbot.rs](05_use_cases/simple_chatbot.rs) |
| **Use OpenAI features** | [openai/](04_providers/openai/) |

## Directory Structure

### 01_getting_started
*First-time users*

- [quick_start.rs](01_getting_started/quick_start.rs) - Basic usage with multiple providers
- [provider_comparison.rs](01_getting_started/provider_comparison.rs) - Compare different AI providers
- [basic_usage.rs](01_getting_started/basic_usage.rs) - Core concepts and message types
- [convenience_methods.rs](01_getting_started/convenience_methods.rs) - Simplified APIs

### 02_core_features
*Essential functionality*

- [chat_basics.rs](02_core_features/chat_basics.rs) - Foundation of AI interactions
- [streaming_chat.rs](02_core_features/streaming_chat.rs) - Real-time response streaming
- [unified_interface.rs](02_core_features/unified_interface.rs) - Provider-agnostic interface
- [error_handling.rs](02_core_features/error_handling.rs) - Production-ready error management
- [parameter_mapping.rs](02_core_features/parameter_mapping.rs) - Parameter conversion between providers
- [capability_detection.rs](02_core_features/capability_detection.rs) - Feature detection

### 03_advanced_features
*Specialized capabilities*

- [thinking_models.rs](03_advanced_features/thinking_models.rs) - AI reasoning and thinking process
- [multimodal_processing.rs](03_advanced_features/multimodal_processing.rs) - Text, image, and audio
- [batch_processing.rs](03_advanced_features/batch_processing.rs) - High-volume concurrent processing
- [custom_configurations.rs](03_advanced_features/custom_configurations.rs) - Advanced setup patterns

### 04_providers
*Provider-specific features*

| Provider | Features | Directory |
|----------|----------|-----------|
| OpenAI | GPT models, DALL-E, Whisper | [openai/](04_providers/openai/) |
| Anthropic | Claude models, thinking | [anthropic/](04_providers/anthropic/) |
| Google | Gemini models, files API | [google/](04_providers/google/) |
| Ollama | Local models, privacy | [ollama/](04_providers/ollama/) |
| OpenAI Compatible | DeepSeek, Groq, etc. | [openai_compatible/](04_providers/openai_compatible/) |

### 05_use_cases
*Complete applications*

- [simple_chatbot.rs](05_use_cases/simple_chatbot.rs) - Interactive chatbot with memory
- [code_assistant.rs](05_use_cases/code_assistant.rs) - Programming helper
- [content_generator.rs](05_use_cases/content_generator.rs) - Text generation tool
- [api_integration.rs](05_use_cases/api_integration.rs) - REST API with AI capabilities

## Setup

Set API keys for the providers you want to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
```

For Ollama (local):
```bash
# Install Ollama from https://ollama.ai
ollama serve
ollama pull llama3.2
```

Run examples:

```bash
# Getting started
cargo run --example quick_start
cargo run --example provider_comparison

# Core features
cargo run --example chat_basics
cargo run --example streaming_chat

# Provider-specific
cargo run --example basic_setup  # Ollama
cargo run --example enhanced_features  # OpenAI

# Use cases
cargo run --example simple_chatbot
```

## Learning Path

**Beginner**: Start with `quick_start.rs` → `provider_comparison.rs` → `chat_basics.rs`

**Intermediate**: Focus on `streaming_chat.rs` → `unified_interface.rs` → `simple_chatbot.rs`

**Advanced**: Study `thinking_models.rs` → `batch_processing.rs` → provider-specific features

**Production**: Explore `error_handling.rs` → `custom_configurations.rs` → `api_integration.rs`

## Key Concepts

### Message Types
```rust
use siumai::prelude::*;

let messages = vec![
    system!("You are a helpful assistant"),
    user!("Hello!"),
    assistant!("Hi there! How can I help?"),
];
```

### Provider Creation
```rust
// OpenAI
let client = LlmBuilder::new()
    .openai()
    .api_key("your-key")
    .model("gpt-4o-mini")
    .build()
    .await?;

// Anthropic
let client = LlmBuilder::new()
    .anthropic()
    .api_key("your-key")
    .model("claude-3-5-haiku-20241022")
    .build()
    .await?;

// Ollama (local)
let client = LlmBuilder::new()
    .ollama()
    .base_url("http://localhost:11434")
    .model("llama3.2")
    .build()
    .await?;
```

### Basic Chat
```rust
let response = client.chat(messages).await?;
if let Some(text) = response.content_text() {
    println!("AI: {}", text);
}
```

### Streaming
```rust
let mut stream = client.chat_stream(messages, None).await?;
while let Some(event) = stream.next().await {
    match event? {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            print!("{}", delta);
        }
        ChatStreamEvent::Done { .. } => break,
        _ => {}
    }
}
```

## Error Handling

```rust
match client.chat(messages).await {
    Ok(response) => {
        // Handle successful response
    }
    Err(LlmError::AuthenticationError(msg)) => {
        // Handle auth errors
    }
    Err(LlmError::RateLimitError(msg)) => {
        // Handle rate limits
    }
    Err(e) => {
        // Handle other errors
    }
}
```

## Best Practices

1. **Always handle errors gracefully**
2. **Use environment variables for API keys**
3. **Monitor token usage in production**
4. **Implement retry logic for transient errors**
5. **Choose the right provider for your use case**
6. **Use streaming for better user experience**
7. **Test with multiple providers for reliability**

## Resources

- [Main Documentation](../README.md)
- [API Reference](https://docs.rs/siumai/)
- [GitHub Repository](https://github.com/YumchaLabs/siumai)
- [Crates.io](https://crates.io/crates/siumai)

# Siumai Unified Interface Examples

This directory contains example code demonstrating the functionality of the Siumai unified interface.

## What is the Siumai Interface?

The Siumai interface is a unified LLM provider interface. It allows you to:

- 🔄 **Dynamically Switch Providers** - Switch between different LLM providers at runtime.
- 🔍 **Capability Detection** - Automatically detect the features supported by a provider.
- 🛡️ **Type Safety** - Maintain Rust's compile-time type checking.
- 📦 **Unified API** - Interact with different providers using the same code.

## Basic Usage

### Creating a Siumai Provider

```rust
use siumai::prelude::*;

// Method 1: Create from an existing client
let openai_client = llm()
    .openai()
    .api_key("your-key")
    .model("gpt-4")
    .build()
    .await?;

let siumai = Siumai::new(Box::new(openai_client));

// Method 2: Using the siumai() builder (Planned)
let siumai = siumai()
    .provider(ProviderType::OpenAi)
    .api_key("your-key")
    .model("gpt-4")
    .build()
    .await?;
```

### Basic Chat

```rust
let messages = vec![user!("Hello, world!")];
let response = siumai.chat(messages).await?;
println!("Response: {}", response.text().unwrap_or(""));
```

### Capability Detection

```rust
// Check provider information
println!("Provider: {}", siumai.provider_name());
println!("Type: {:?}", siumai.metadata().provider_type);

// Check for specific capabilities
if siumai.supports("audio") {
    println!("✅ Audio processing supported");
}

if siumai.supports("vision") {
    println!("✅ Vision processing supported");
}

// Get all capabilities
let caps = siumai.capabilities();
println!("Streaming: {}", caps.streaming);
println!("Tools: {}", caps.tools);
```

### Streaming Response

```rust
if siumai.supports("streaming") {
    let mut stream = siumai.chat_stream(messages, None).await?;
    
    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                print!("{}", delta);
            }
            ChatStreamEvent::StreamEnd { .. } => {
                println!("\n✅ Complete");
                break;
            }
            _ => {}
        }
    }
}
```

## Advanced Usage

### Provider-Independent Functions

```rust
async fn chat_with_any_provider(provider: &Siumai) -> Result<String, LlmError> {
    let messages = vec![user!("What is AI?")];
    let response = provider.chat(messages).await?;
    Ok(response.text().unwrap_or("").to_string())
}

// Can be used with any provider
let openai_siumai = Siumai::new(Box::new(openai_client));
let anthropic_siumai = Siumai::new(Box::new(anthropic_client));

let openai_response = chat_with_any_provider(&openai_siumai).await?;
let anthropic_response = chat_with_any_provider(&anthropic_siumai).await?;
```

### Task-Based Provider Selection

```rust
async fn select_provider_for_task(task: &str) -> Result<Siumai, LlmError> {
    match task {
        "image_generation" => {
            // Select OpenAI for image generation
            let client = llm().openai().model("dall-e-3").build().await?;
            Ok(Siumai::new(Box::new(client)))
        }
        "reasoning" => {
            // Select Anthropic for complex reasoning
            let client = llm().anthropic().model("claude-3-opus").build().await?;
            Ok(Siumai::new(Box::new(client)))
        }
        _ => {
            // Default to GPT-4
            let client = llm().openai().model("gpt-4").build().await?;
            Ok(Siumai::new(Box::new(client)))
        }
    }
}
```

### Provider Fallback Strategy

```rust
async fn chat_with_fallback(message: &str) -> Result<String, LlmError> {
    let providers = vec![
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-sonnet"),
        ("openai", "gpt-3.5-turbo"), // Fallback to a cheaper model
    ];

    for (provider_name, model) in providers {
        if let Ok(client) = create_client(provider_name, model).await {
            let siumai = Siumai::new(Box::new(client));
            if let Ok(response) = siumai.chat(vec![user!(message)]).await {
                return Ok(response.text().unwrap_or("").to_string());
            }
        }
    }

    Err(LlmError::InternalError("All providers failed".to_string()))
}
```

## Example Files

- **`siumai_interface.rs`** - A complete demonstration of the siumai interface.

## Running the Examples

```bash
# Run the siumai interface example
cargo run --example siumai_interface

```


## Advantages

1.  **Type Safety** - Compile-time checks prevent runtime errors.
2.  **Performance** - Zero-cost abstractions maintain Rust's performance benefits.
3.  **Flexibility** - Dynamic provider switching and capability detection.
4.  **Consistency** - A unified API design that is easy to learn and use.
5.  **Extensibility** - Easy to add new providers and features.

## Future Plans

- [ ] Complete implementation of the `siumai()` builder.
- [ ] Support for more providers (Gemini, xAI, etc.).
- [ ] Advanced capability composition and type-safe access.
- [ ] Performance optimizations and caching mechanisms.
- [ ] More comprehensive examples and documentation.

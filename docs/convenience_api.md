# Convenience API Guide

This document explains the simplified APIs and convenience methods that make Siumai easier to use for common tasks.

## 1. Quick Client Creation

### Before (Complex Configuration)
```rust
let client = llm()
    .openai()
    .api_key("your-api-key")
    .model("gpt-4")
    .temperature(0.7)
    .max_tokens(2048)
    .build()
    .await?;
```

### After (Simple Defaults)
```rust
// Uses OPENAI_API_KEY env var and sensible defaults
let client = quick_openai().await?;

// Or with custom model
let client = quick_openai_with_model("gpt-4").await?;
```

### Available Quick Functions
- `quick_openai()` - OpenAI with gpt-4o-mini
- `quick_anthropic()` - Anthropic with claude-3-5-sonnet
- `quick_gemini()` - Gemini with gemini-1.5-flash
- `quick_*_with_model(model)` - Custom model variants

## 2. Preset Configurations

### Fast Configuration (Interactive Apps)
```rust
let client = llm()
    .fast()  // 30s timeout, 5s connect timeout
    .openai()
    .model("gpt-4o-mini")
    .build()
    .await?;
```

### Long-Running Configuration (Batch Processing)
```rust
let client = llm()
    .long_running()  // 300s timeout, 30s connect timeout
    .openai()
    .model("gpt-4")
    .build()
    .await?;
```

### Production Defaults
```rust
let client = llm()
    .with_defaults()  // 60s timeout, compression enabled
    .openai()
    .model("gpt-4")
    .build()
    .await?;
```

## 3. Convenience Chat Methods

### Simple Ask
```rust
// Before: Complex message construction
let messages = vec![ChatMessage::user("What is 2+2?").build()];
let response = client.chat(messages).await?;
let text = response.content_text().unwrap();

// After: One-liner
let response = client.ask("What is 2+2?").await?;
```

### Ask with System Prompt
```rust
let response = client.ask_with_system(
    "You are a helpful math tutor",
    "Explain the Pythagorean theorem"
).await?;
```

### Translation
```rust
let response = client.translate("Hello, how are you?", "Spanish").await?;
```

### Explanation
```rust
let response = client.explain("blockchain", Some("a 10-year-old")).await?;
```

### Creative Generation
```rust
let response = client.generate("haiku", "about programming").await?;
```

### Conversation Continuation
```rust
let mut conversation = vec![system!("You are a helpful assistant")];

let (response, updated_conversation) = client
    .continue_conversation(conversation, "Hello!".to_string())
    .await?;
```

## 4. Message Creation Macros

### Simple Messages
```rust
// Before: Verbose construction
let msg = ChatMessage::user("Hello").build();

// After: Concise macros
let msg = user!("Hello");
let msg = system!("You are helpful");
let msg = assistant!("I can help");
```

### Messages with Images
```rust
let msg = user!("Describe this image", image: "https://example.com/image.jpg");
let msg = user!("Analyze this", image: "url", detail: "high");
```

### Tool Messages
```rust
let msg = tool!("Function result", id: "call_123");
```

## 5. Parameter Presets

### Creative Tasks
```rust
let params = CommonParams::creative("gpt-4");
// Sets: temperature=0.9, max_tokens=2048, top_p=0.9
```

### Factual/Analytical Tasks
```rust
let params = CommonParams::factual("gpt-4");
// Sets: temperature=0.1, max_tokens=1024, top_p=0.95
```

### Code Generation
```rust
let params = CommonParams::coding("gpt-4");
// Sets: temperature=0.2, max_tokens=4096, top_p=0.95
```

### Provider-Specific Presets
```rust
let openai_params = ProviderParams::openai();
let anthropic_params = ProviderParams::anthropic();
let gemini_params = ProviderParams::gemini();
```

## 6. Method Chaining for Parameters

```rust
let params = CommonParams::new("gpt-4")
    .with_temperature(0.7)
    .with_max_tokens(2048)
    .with_top_p(0.9)
    .with_stop_sequences(vec!["END".to_string()]);
```

## 7. Complete Example

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Quick client creation
    let client = quick_openai().await?;
    
    // Simple conversation
    let response = client.ask("What's the weather like?").await?;
    println!("Response: {}", response);
    
    // Creative task
    let poem = client.generate("poem", "about Rust programming").await?;
    println!("Poem: {}", poem);
    
    // Translation
    let spanish = client.translate("Hello world", "Spanish").await?;
    println!("Spanish: {}", spanish);
    
    // Complex conversation with macros
    let messages = vec![
        system!("You are a helpful coding assistant"),
        user!("How do I create a vector in Rust?"),
    ];
    
    let response = client.chat(messages).await?;
    if let Some(text) = response.content_text() {
        println!("Coding help: {}", text);
    }
    
    Ok(())
}
```

## Benefits

1. **Reduced Boilerplate**: Common tasks require less code
2. **Sensible Defaults**: Production-ready configurations out of the box
3. **Progressive Disclosure**: Simple API for beginners, full control for experts
4. **Type Safety**: All convenience methods maintain compile-time safety
5. **Consistency**: Uniform patterns across all providers

## Migration Guide

Existing code continues to work unchanged. The convenience methods are additive and don't break existing APIs. You can gradually adopt them where they make sense.

### When to Use Convenience Methods
- ✅ Prototyping and quick experiments
- ✅ Simple applications with standard requirements
- ✅ Getting started with the library
- ✅ Common tasks like translation, explanation, etc.

### When to Use Full Builder API
- ✅ Complex configurations
- ✅ Custom HTTP clients
- ✅ Provider-specific features
- ✅ Fine-grained control over parameters
- ✅ Production applications with specific requirements

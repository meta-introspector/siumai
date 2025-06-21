# Core Features

Essential functionality that every Siumai user should understand. These examples demonstrate the fundamental capabilities of the library.

## Examples

### [chat_basics.rs](chat_basics.rs)
**💬 Foundation of AI interactions**

Master the fundamentals of chat-based AI interactions.

```bash
cargo run --example chat_basics
```

**What you'll learn:**
- Message types and conversation management
- Response metadata and usage statistics
- Context management strategies
- Conversation history handling

### [streaming_chat.rs](streaming_chat.rs)
**🌊 Real-time response streaming**

Learn how to handle streaming responses for better user experience.

```bash
cargo run --example streaming_chat
```

**What you'll learn:**
- Processing stream events
- Real-time content display
- Performance optimization
- Error handling in streams

### [unified_interface.rs](unified_interface.rs)
**🔄 Provider-agnostic interface**

Use the same code with different AI providers.

```bash
cargo run --example unified_interface
```

**What you'll learn:**
- Provider abstraction
- Dynamic provider switching
- Capability detection
- Fallback strategies

### [error_handling.rs](error_handling.rs)
**🛡️ Production-ready error management**

Handle errors gracefully in production applications.

```bash
cargo run --example error_handling
```

**What you'll learn:**
- Error types and classification
- Retry strategies
- Rate limit handling
- Graceful degradation

### [parameter_mapping.rs](parameter_mapping.rs)
**🔄 Parameter conversion between providers**

Understand how parameters are mapped between different providers.

```bash
cargo run --example parameter_mapping
```

**What you'll learn:**
- Common vs provider-specific parameters
- Parameter validation
- Default value handling
- Custom parameter mapping

### [capability_detection.rs](capability_detection.rs)
**🔍 Feature detection**

Detect and use provider capabilities dynamically.

```bash
cargo run --example capability_detection
```

**What you'll learn:**
- Capability checking
- Feature availability
- Graceful feature degradation
- Provider metadata

### [response_cache.rs](response_cache.rs)
**💾 Response caching for performance**

Dramatically improve performance and reduce costs with intelligent response caching.

```bash
cargo run --example response_cache
```

**What you'll learn:**
- Cache implementation patterns
- Performance optimization strategies
- Cost reduction techniques
- Production-ready caching integration
- Thread-safe cache management

## Prerequisites

Make sure you have API keys set up:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
# or for local development
ollama serve && ollama pull llama3.2
```

## Learning Path

### For Chat Applications
1. [chat_basics.rs](chat_basics.rs) - Learn conversation fundamentals
2. [streaming_chat.rs](streaming_chat.rs) - Add real-time responses
3. [response_cache.rs](response_cache.rs) - Optimize performance with caching
4. [error_handling.rs](error_handling.rs) - Make it production-ready

### For Multi-Provider Applications
1. [unified_interface.rs](unified_interface.rs) - Provider abstraction
2. [capability_detection.rs](capability_detection.rs) - Feature detection
3. [parameter_mapping.rs](parameter_mapping.rs) - Parameter handling

### For Production Applications
1. [error_handling.rs](error_handling.rs) - Robust error handling
2. [response_cache.rs](response_cache.rs) - Performance optimization
3. [capability_detection.rs](capability_detection.rs) - Graceful degradation
4. [parameter_mapping.rs](parameter_mapping.rs) - Fine-tuned control

## Key Patterns

### Basic Chat Pattern
```rust
use siumai::prelude::*;

let client = LlmBuilder::new()
    .openai()
    .api_key(&api_key)
    .model("gpt-4o-mini")
    .build()
    .await?;

let messages = vec![
    system!("You are a helpful assistant"),
    user!("Hello!")
];

let response = client.chat(messages).await?;
```

### Streaming Pattern
```rust
use futures_util::StreamExt;

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

### Error Handling Pattern
```rust
match client.chat(messages).await {
    Ok(response) => {
        // Handle success
    }
    Err(LlmError::RateLimitError(_)) => {
        // Wait and retry
        tokio::time::sleep(Duration::from_secs(1)).await;
        // Retry logic
    }
    Err(LlmError::AuthenticationError(_)) => {
        // Check API key
    }
    Err(e) => {
        // Handle other errors
    }
}
```

### Provider Abstraction Pattern
```rust
async fn chat_with_any_provider(
    provider: &LlmClient,
    message: &str
) -> Result<String, LlmError> {
    let messages = vec![user!(message)];
    let response = provider.chat(messages).await?;
    Ok(response.content_text().unwrap_or_default().to_string())
}
```

## Performance Tips

### For Chat Applications
- Use streaming for better perceived performance
- Implement proper conversation history management
- Monitor token usage to control costs

### For High-Volume Applications
- Implement connection pooling
- Use batch processing where possible
- Add proper rate limiting

### For Real-Time Applications
- Prefer streaming over regular chat
- Use faster providers (Groq) for low latency
- Implement proper error recovery

## Common Patterns

### Conversation Management
```rust
let mut conversation = vec![
    system!("You are a helpful assistant")
];

// Add user message
conversation.push(user!("Hello!"));

// Get response and add to conversation
let response = client.chat(conversation.clone()).await?;
if let Some(text) = response.content_text() {
    conversation.push(assistant!(text));
}
```

### Provider Fallback
```rust
let providers = vec![primary_client, fallback_client];

for client in providers {
    match client.chat(messages.clone()).await {
        Ok(response) => return Ok(response),
        Err(_) => continue, // Try next provider
    }
}
```

### Capability-Based Usage
```rust
if client.supports_streaming() {
    // Use streaming
    let stream = client.chat_stream(messages, None).await?;
    // Handle stream
} else {
    // Fall back to regular chat
    let response = client.chat(messages).await?;
    // Handle response
}
```

## Next Steps

After mastering these core features:

- **[Advanced Features](../03_advanced_features/)**: Thinking models, multimodal processing
- **[Provider Features](../04_providers/)**: Provider-specific capabilities
- **[Use Cases](../05_use_cases/)**: Complete application examples

## Troubleshooting

### Streaming Issues
- Check if the provider supports streaming
- Verify network connectivity
- Handle stream errors properly

### Performance Issues
- Monitor token usage
- Use appropriate models for your use case
- Implement proper caching

### Error Handling Issues
- Always handle specific error types
- Implement retry logic for transient errors
- Log errors for debugging

# Advanced Features

Sophisticated AI capabilities for production applications with Siumai. These examples demonstrate advanced patterns and specialized use cases.

## Examples

### [thinking_models.rs](thinking_models.rs)
**🧠 AI reasoning and thinking process**

Explore models that show their reasoning process, like Claude's thinking or o1's chain of thought.

```bash
cargo run --example thinking_models
```

**What you'll learn:**
- Working with reasoning models
- Accessing thinking processes
- Optimizing for reasoning tasks
- Understanding model limitations

### [multimodal_processing.rs](multimodal_processing.rs)
**🎭 Text, image, and audio processing**

Handle multiple types of content in a single conversation.

```bash
cargo run --example multimodal_processing
```

**What you'll learn:**
- Image analysis and description
- Audio processing and transcription
- Combining different modalities
- Handling multimodal responses

### [batch_processing.rs](batch_processing.rs)
**⚡ High-volume concurrent processing**

Process large numbers of requests efficiently with proper rate limiting.

```bash
cargo run --example batch_processing
```

**What you'll learn:**
- Concurrent request handling
- Rate limiting strategies
- Progress tracking
- Error handling at scale

### [custom_configurations.rs](custom_configurations.rs)
**🔧 Advanced setup patterns**

Advanced configuration patterns for production deployments.

```bash
cargo run --example custom_configurations
```

**What you'll learn:**
- Custom parameter mapping
- Provider-specific optimizations
- Configuration management
- Performance tuning

## Prerequisites

These examples require more advanced setup:

```bash
# API keys for advanced features
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# For multimodal examples
export ELEVENLABS_API_KEY="your-key"  # Optional for audio

# For local processing
ollama serve
ollama pull llama3.2
ollama pull llava  # For vision tasks
```

## Learning Path

### For AI Researchers
1. [thinking_models.rs](thinking_models.rs) - Understand reasoning processes
2. [multimodal_processing.rs](multimodal_processing.rs) - Explore multimodal AI
3. [custom_configurations.rs](custom_configurations.rs) - Fine-tune performance

### For Production Engineers
1. [batch_processing.rs](batch_processing.rs) - Scale processing
2. [custom_configurations.rs](custom_configurations.rs) - Optimize for production
3. [thinking_models.rs](thinking_models.rs) - Leverage advanced models

### For Application Developers
1. [multimodal_processing.rs](multimodal_processing.rs) - Rich user experiences
2. [batch_processing.rs](batch_processing.rs) - Handle user scale
3. [custom_configurations.rs](custom_configurations.rs) - Optimize costs

## Key Concepts

### Thinking Models
```rust
// Access reasoning process
let response = client.chat(messages).await?;
if let Some(thinking) = response.thinking {
    println!("AI reasoning: {}", thinking);
}
```

### Multimodal Processing
```rust
// Combine text and images
let message = ChatMessage::user("Describe this image")
    .with_image("path/to/image.jpg", Some("high"))
    .build();
```

### Batch Processing
```rust
// Process multiple requests concurrently
let futures: Vec<_> = requests.into_iter()
    .map(|req| process_request(client.clone(), req))
    .collect();

let results = futures::future::join_all(futures).await;
```

### Custom Configurations
```rust
// Fine-tune for specific use cases
let client = LlmBuilder::new()
    .openai()
    .model("gpt-4")
    .temperature(0.1)  // Low for consistency
    .max_tokens(2000)
    .custom_parameter("reasoning_effort", "high")
    .build()
    .await?;
```

## Performance Considerations

### Thinking Models
- Higher latency due to reasoning process
- More expensive per token
- Better quality for complex tasks
- Consider caching for repeated queries

### Multimodal Processing
- Larger request sizes
- Different pricing for different modalities
- Provider-specific capabilities
- Format and size limitations

### Batch Processing
- Rate limiting is crucial
- Memory usage scales with concurrency
- Error handling becomes complex
- Progress tracking is important

### Custom Configurations
- Provider-specific optimizations
- Cost vs. quality trade-offs
- Latency vs. throughput balance
- Monitoring and alerting needs

## Common Patterns

### Reasoning Chain Analysis
```rust
async fn analyze_with_reasoning(
    client: &LlmClient,
    problem: &str
) -> Result<(String, Option<String>), LlmError> {
    let messages = vec![
        system!("Think step by step and show your reasoning"),
        user!(problem)
    ];
    
    let response = client.chat(messages).await?;
    Ok((
        response.content_text().unwrap_or_default().to_string(),
        response.thinking
    ))
}
```

### Multimodal Content Builder
```rust
fn build_multimodal_message(
    text: &str,
    image_path: Option<&str>,
    audio_path: Option<&str>
) -> ChatMessage {
    let mut builder = ChatMessage::user(text);
    
    if let Some(image) = image_path {
        builder = builder.with_image(image, Some("high"));
    }
    
    if let Some(audio) = audio_path {
        builder = builder.with_audio(audio, "mp3");
    }
    
    builder.build()
}
```

### Batch Processor with Rate Limiting
```rust
use tokio::time::{sleep, Duration};
use futures::stream::{self, StreamExt};

async fn process_batch<T, F, Fut>(
    items: Vec<T>,
    processor: F,
    concurrency: usize,
    rate_limit: Duration,
) -> Vec<Result<String, LlmError>>
where
    F: Fn(T) -> Fut + Clone,
    Fut: Future<Output = Result<String, LlmError>>,
{
    stream::iter(items)
        .map(|item| {
            let proc = processor.clone();
            async move {
                sleep(rate_limit).await;
                proc(item).await
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await
}
```

## Troubleshooting

### Thinking Models
- Not all providers support thinking
- Thinking content may be large
- Additional costs for reasoning tokens
- Longer response times

### Multimodal Issues
- File size and format limitations
- Provider-specific support varies
- Higher costs for image/audio processing
- Network bandwidth considerations

### Batch Processing Issues
- Rate limit errors
- Memory exhaustion
- Partial failures
- Progress tracking complexity

### Configuration Issues
- Provider-specific parameters
- Invalid parameter combinations
- Performance vs. cost trade-offs
- Monitoring complexity

## Next Steps

After mastering these advanced features:

- **[Provider Features](../04_providers/)**: Provider-specific advanced capabilities
- **[Use Cases](../05_use_cases/)**: Complete applications using advanced features
- **Production Deployment**: Scaling and monitoring considerations

## Resources

- [Thinking Models Guide](../docs/thinking-models.md)
- [Multimodal Best Practices](../docs/multimodal.md)
- [Batch Processing Patterns](../docs/batch-processing.md)
- [Performance Optimization](../docs/performance.md)

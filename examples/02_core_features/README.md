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

## 🔑 Key Patterns

### Essential Patterns You'll Learn

**Basic Chat**: Simple request-response interactions
- Create client → Send messages → Handle response
- Foundation for all AI interactions

**Streaming**: Real-time response processing
- Process responses as they arrive
- Better user experience for long responses

**Embeddings**: Vector representations of text
- Convert text to numerical vectors
- Enable semantic similarity and search
- Support different models and dimensions

**Error Handling**: Production-ready error management
- Handle different error types appropriately
- Implement retry logic and fallbacks

**Provider Abstraction**: Write once, run anywhere
- Same code works with different AI providers
- Easy provider switching and fallbacks

**Capability Detection**: Adaptive functionality
- Check what features are available
- Graceful degradation when features aren't supported

## 🚀 Performance Tips

### Chat Applications
- Use streaming for better perceived performance
- Manage conversation history efficiently
- Monitor token usage to control costs

### High-Volume Applications
- Implement connection pooling
- Use batch processing where possible
- Add proper rate limiting

### Real-Time Applications
- Prefer streaming over regular chat
- Use faster providers for low latency
- Implement proper error recovery

## 🔄 Common Patterns

### Conversation Management
Build and maintain conversation history properly for multi-turn interactions.

### Provider Fallback
Implement fallback logic to try multiple providers for reliability.

### Capability-Based Usage
Check provider capabilities and adapt functionality accordingly.

### Response Caching
Cache responses to improve performance and reduce costs.

## 🎯 Next Steps

After mastering these core features:

- **[Advanced Features](../03_advanced_features/)** - Thinking models, batch processing
- **[Provider Features](../04_providers/)** - Provider-specific capabilities
- **[Use Cases](../05_use_cases/)** - Complete application examples

## 🔧 Troubleshooting

**Streaming Issues**: Check provider support, network connectivity, error handling

**Performance Issues**: Monitor token usage, choose appropriate models, implement caching

**Error Handling**: Handle specific error types, implement retry logic, log for debugging

---

**Ready to dive deeper? Start with [chat_basics.rs](chat_basics.rs) to master the fundamentals! 💪**

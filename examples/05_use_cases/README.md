# 🎯 Use Cases - Real-World Applications

This directory contains comprehensive examples of real-world applications built with Siumai. These examples demonstrate how to integrate AI capabilities into production systems and solve practical problems.

## 📋 Available Use Cases

### 💻 [Code Assistant](./code_assistant.rs)
**AI-powered coding helper with comprehensive analysis capabilities**

Features:
- Code explanation and documentation generation
- Bug detection and fixing suggestions
- Code refactoring recommendations
- Multi-language support (Rust, Python, JavaScript, Java, C++, etc.)
- Interactive code review
- Performance optimization suggestions

```bash
# Run the code assistant
cargo run --example code_assistant

# Example commands:
explain src/main.rs
review src/lib.rs
optimize src/performance.rs
document src/api.rs
fix src/buggy_code.rs
refactor src/legacy.rs
```

**Use Cases:**
- Development workflow automation
- Code quality improvement
- Learning and education
- Legacy code modernization
- Team code reviews

---

### ✍️ [Content Generator](./content_generator.rs)
**AI content creation tool for multiple formats and platforms**

Features:
- Blog post and article generation
- Marketing copy creation
- Social media content for multiple platforms
- Professional email templates
- Technical documentation
- SEO-optimized content
- Creative writing and storytelling

```bash
# Run the content generator
cargo run --example content_generator

# Example commands:
blog "AI in Healthcare"
marketing "SaaS Analytics Platform"
social "Remote Work Tips"
email "Product Launch Announcement"
docs "API Integration Guide"
seo "machine learning tutorials"
creative "A story about time travel"
```

**Use Cases:**
- Content marketing automation
- Social media management
- Technical writing
- Creative projects
- SEO content creation

---

### 🌐 [API Integration](./api_integration.rs)
**REST API service with AI capabilities**

Features:
- HTTP server with AI endpoints
- Request/response handling and validation
- Authentication and rate limiting
- Async processing and streaming
- Error handling and logging
- Usage statistics and monitoring

```bash
# Run the API server
cargo run --example api_integration

# Test endpoints:
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-key-123" \
  -d '{"message": "Hello, AI!"}'

curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-key-123" \
  -d '{"prompt": "Write a blog post", "content_type": "blog"}'
```

**Available Endpoints:**
- `POST /api/chat` - Interactive chat with AI
- `POST /api/generate` - Content generation
- `POST /api/analyze` - Text analysis
- `GET /api/health` - Health monitoring
- `GET /api/stats` - Usage statistics

**Use Cases:**
- Microservice architecture
- SaaS applications
- API-first development
- Integration platforms
- Production AI services

---

### 🤖 [Simple Chatbot](./simple_chatbot.rs)
**Basic chatbot implementation with conversation management**

Features:
- Multi-turn conversation handling
- Context management
- Personality customization
- Memory and persistence
- Error handling and recovery

```bash
# Run the simple chatbot
cargo run --example simple_chatbot
```

**Use Cases:**
- Customer support automation
- Interactive applications
- Learning and experimentation
- Prototype development

---

## 🚀 Getting Started

### Prerequisites

1. **Set up API keys:**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GROQ_API_KEY="your-groq-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

2. **Install dependencies:**
   ```bash
   cargo build --examples
   ```

### Running Examples

Each example can be run independently:

```bash
# Code assistant
cargo run --example code_assistant

# Content generator
cargo run --example content_generator

# API integration
cargo run --example api_integration

# Simple chatbot
cargo run --example simple_chatbot
```

## 🏗️ Architecture Patterns

### Common Patterns Used

1. **Builder Pattern**
   - Flexible AI client configuration
   - Provider-agnostic setup
   - Parameter customization

2. **Strategy Pattern**
   - Multiple AI providers
   - Fallback mechanisms
   - Provider switching

3. **Command Pattern**
   - Interactive CLI interfaces
   - Action dispatching
   - Extensible commands

4. **Observer Pattern**
   - Usage statistics
   - Event logging
   - Monitoring hooks

### Production Considerations

1. **Error Handling**
   - Graceful degradation
   - Retry strategies
   - User-friendly messages

2. **Performance**
   - Async processing
   - Connection pooling
   - Response caching

3. **Security**
   - Input validation
   - Authentication
   - Rate limiting

4. **Monitoring**
   - Usage tracking
   - Performance metrics
   - Health checks

## 🔧 Customization

### Extending Examples

1. **Add New Commands**
   ```rust
   match command {
       "new_command" => {
           // Your implementation
       }
       // ... existing commands
   }
   ```

2. **Custom AI Parameters**
   ```rust
   let ai = SiumaiBuilder::new()
       .provider(Provider::OpenAI)
       .model("gpt-4")
       .temperature(0.8)
       .max_tokens(2000)
       .build()
       .await?;
   ```

3. **Add New Providers**
   ```rust
   let ai = SiumaiBuilder::new()
       .provider(Provider::Custom("your-provider"))
       .api_key(&api_key)
       .build()
       .await?;
   ```

## 📊 Performance Tips

1. **Connection Reuse**
   - Keep AI clients alive
   - Use connection pooling
   - Avoid frequent reconnections

2. **Request Optimization**
   - Batch similar requests
   - Use appropriate model sizes
   - Optimize prompt lengths

3. **Caching**
   - Cache frequent responses
   - Use semantic caching
   - Implement TTL strategies

4. **Monitoring**
   - Track response times
   - Monitor token usage
   - Set up alerts

## 🔗 Integration Examples

### Database Integration
```rust
// Example with database persistence
struct ChatHistory {
    db: Database,
    ai: Box<dyn ChatCapability>,
}

impl ChatHistory {
    async fn save_conversation(&self, messages: &[ChatMessage]) {
        // Save to database
    }
    
    async fn load_conversation(&self, user_id: &str) -> Vec<ChatMessage> {
        // Load from database
    }
}
```

### Web Framework Integration
```rust
// Example with web framework
#[post("/chat")]
async fn chat_endpoint(
    request: Json<ChatRequest>,
    ai: Data<Box<dyn ChatCapability>>,
) -> Result<Json<ChatResponse>, Error> {
    let response = ai.chat(&request.messages).await?;
    Ok(Json(ChatResponse { text: response.text() }))
}
```

## 🎯 Next Steps

1. **Explore Advanced Features**
   - Check out [Advanced Features](../03_advanced_features/)
   - Learn about [Provider-Specific Features](../04_providers/)

2. **Build Your Own Use Case**
   - Start with a simple example
   - Add your specific requirements
   - Integrate with your existing systems

3. **Production Deployment**
   - Add proper logging
   - Implement monitoring
   - Set up CI/CD pipelines
   - Configure load balancing

4. **Community Contributions**
   - Share your use cases
   - Contribute improvements
   - Report issues and feedback

## 📚 Additional Resources

- [Core Features](../02_core_features/) - Learn the fundamentals
- [Provider Documentation](../04_providers/) - Provider-specific guides
- [API Reference](https://docs.rs/siumai) - Complete API documentation
- [GitHub Repository](https://github.com/yumchalabs/siumai) - Source code and issues

---

**Happy building! 🚀**

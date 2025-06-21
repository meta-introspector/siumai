# 🧠 Anthropic Claude Provider

This directory contains examples demonstrating Anthropic Claude's capabilities through the Siumai library. Claude is known for its constitutional AI approach, excellent reasoning abilities, and large context windows.

## 📋 Available Examples

### 🧠 [Basic Chat](./basic_chat.rs)
**Comprehensive introduction to Claude's core capabilities**

Features demonstrated:
- Model selection and comparison (Haiku, Sonnet, Opus)
- Parameter optimization for different use cases
- Context window management (up to 200K tokens)
- Cost-effective usage patterns
- Claude-specific features and strengths

```bash
# Run the basic chat example
cargo run --example anthropic_basic_chat
```

**Key Learning Points:**
- When to use each Claude model
- How to optimize parameters for your use case
- Managing large context windows effectively
- Cost optimization strategies

---

### 🤔 [Thinking Showcase](./thinking_showcase.rs)
**Explore Claude's advanced reasoning and thinking processes**

Features demonstrated:
- Step-by-step problem solving
- Complex reasoning analysis
- Mathematical problem solving
- Logical reasoning chains
- Creative problem solving

```bash
# Run the thinking showcase
cargo run --example anthropic_thinking_showcase
```

**Key Learning Points:**
- How to access Claude's reasoning process
- Techniques for complex problem solving
- Mathematical and logical reasoning patterns
- Creative thinking applications

---

## 🚀 Getting Started

### Prerequisites

1. **Get your Anthropic API key:**
   - Sign up at [Anthropic Console](https://console.anthropic.com/)
   - Create an API key
   - Set the environment variable:
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

2. **Install dependencies:**
   ```bash
   cargo build --examples
   ```

### Quick Start

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = SiumaiBuilder::new()
        .provider(Provider::Anthropic)
        .api_key(&std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-sonnet")
        .temperature(0.3)
        .max_tokens(1000)
        .build()
        .await?;

    let messages = vec![
        ChatMessage::user("Explain quantum computing in simple terms")
    ];

    let response = ai.chat(&messages).await?;
    println!("Claude: {}", response.text().unwrap_or_default());
    
    Ok(())
}
```

## 🎯 Claude Model Guide

### Model Selection

| Model | Best For | Context | Speed | Cost |
|-------|----------|---------|-------|------|
| **Claude 3 Haiku** | Simple tasks, fast responses | 200K | ⭐⭐⭐ | ⭐⭐⭐ |
| **Claude 3 Sonnet** | Balanced performance | 200K | ⭐⭐ | ⭐⭐ |
| **Claude 3 Opus** | Complex reasoning, highest quality | 200K | ⭐ | ⭐ |

### When to Use Each Model

**Claude 3 Haiku:**
- Quick Q&A
- Simple content generation
- Fast prototyping
- High-volume, cost-sensitive applications

**Claude 3 Sonnet:**
- General-purpose applications
- Balanced quality and speed needs
- Most production use cases
- Educational and tutoring applications

**Claude 3 Opus:**
- Complex analysis and reasoning
- High-stakes content generation
- Research and academic work
- Advanced problem-solving tasks

## ⚙️ Parameter Optimization

### Temperature Guidelines

```rust
// For factual, consistent responses
.temperature(0.0 - 0.3)

// For balanced creativity and accuracy
.temperature(0.3 - 0.7)

// For creative and varied responses
.temperature(0.7 - 1.0)
```

### Token Management

```rust
// Short responses (tweets, quick answers)
.max_tokens(100 - 300)

// Medium responses (explanations, summaries)
.max_tokens(300 - 1000)

// Long responses (articles, detailed analysis)
.max_tokens(1000 - 4000)
```

## 🌟 Claude's Unique Strengths

### Constitutional AI
- Built-in safety and helpfulness
- Ethical reasoning capabilities
- Reduced harmful outputs
- Consistent helpful behavior

### Large Context Windows
- Up to 200,000 tokens
- Excellent for long documents
- Maintains coherence across conversations
- Perfect for research and analysis

### Reasoning Excellence
- Step-by-step problem solving
- Complex logical reasoning
- Mathematical problem solving
- Creative and analytical thinking

### Code Understanding
- Strong programming knowledge
- Code review and optimization
- Multi-language support
- Architecture and design guidance

## 💰 Cost Optimization

### Model Selection Strategy
```rust
// Use Haiku for simple tasks
let simple_ai = SiumaiBuilder::new()
    .model("claude-3-haiku")
    .max_tokens(200)
    .build().await?;

// Use Sonnet for balanced needs
let balanced_ai = SiumaiBuilder::new()
    .model("claude-3-sonnet")
    .max_tokens(1000)
    .build().await?;

// Use Opus only for complex tasks
let complex_ai = SiumaiBuilder::new()
    .model("claude-3-opus")
    .max_tokens(2000)
    .build().await?;
```

### Batch Processing
```rust
// Combine multiple questions
let batch_prompt = "Answer these questions:\n\
    1. What is machine learning?\n\
    2. How does it differ from AI?\n\
    3. What are common applications?";
```

### Context Management
```rust
// Summarize long conversations
if conversation.len() > 20 {
    let summary = ai.chat(&[
        ChatMessage::user("Summarize our conversation so far")
    ]).await?;
    // Start new conversation with summary
}
```

## 🔧 Advanced Usage Patterns

### Conversation Management
```rust
struct ClaudeConversation {
    ai: Box<dyn ChatCapability>,
    messages: Vec<ChatMessage>,
    max_context: usize,
}

impl ClaudeConversation {
    async fn add_message(&mut self, message: ChatMessage) -> Result<String> {
        self.messages.push(message);
        
        // Manage context length
        if self.messages.len() > self.max_context {
            self.summarize_and_truncate().await?;
        }
        
        let response = self.ai.chat(&self.messages).await?;
        let text = response.text().unwrap_or_default();
        
        self.messages.push(ChatMessage::assistant(&text));
        Ok(text)
    }
}
```

### Thinking Process Analysis
```rust
// Request step-by-step reasoning
let thinking_prompt = "Think through this step by step:\n\
    Show your reasoning process for: {}";

// Analyze complex problems
let analysis_prompt = "Analyze this problem considering:\n\
    1. Key factors and constraints\n\
    2. Potential solutions\n\
    3. Trade-offs and implications\n\
    4. Recommended approach";
```

## 📊 Performance Tips

### Response Time Optimization
- Use Haiku for speed-critical applications
- Set appropriate `max_tokens` limits
- Implement request caching
- Use streaming for long responses

### Quality Optimization
- Use Opus for highest quality needs
- Provide clear, specific prompts
- Include relevant context
- Use system messages for consistency

### Cost Optimization
- Monitor token usage with `response.usage()`
- Batch similar requests
- Use appropriate model for task complexity
- Implement response caching

## 🛠️ Integration Examples

### Web Service Integration
```rust
#[post("/chat")]
async fn chat_endpoint(
    request: Json<ChatRequest>,
    claude: Data<Box<dyn ChatCapability>>,
) -> Result<Json<ChatResponse>> {
    let response = claude.chat(&request.messages).await?;
    Ok(Json(ChatResponse {
        text: response.text().unwrap_or_default(),
        usage: response.usage(),
    }))
}
```

### Batch Processing
```rust
async fn process_batch(
    claude: &dyn ChatCapability,
    tasks: Vec<String>
) -> Result<Vec<String>> {
    let batch_prompt = format!(
        "Process these tasks:\n{}",
        tasks.iter().enumerate()
            .map(|(i, task)| format!("{}. {}", i + 1, task))
            .collect::<Vec<_>>()
            .join("\n")
    );
    
    let response = claude.chat(&[
        ChatMessage::user(&batch_prompt)
    ]).await?;
    
    // Parse batch response
    Ok(parse_batch_response(response.text().unwrap_or_default()))
}
```

## 🔗 Related Resources

- [Anthropic Documentation](https://docs.anthropic.com/)
- [Claude Model Cards](https://www.anthropic.com/claude)
- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073)
- [Anthropic Safety Research](https://www.anthropic.com/safety)

## 🎯 Next Steps

1. **Explore Basic Features**
   - Run the basic chat example
   - Try different models and parameters
   - Experiment with context management

2. **Advanced Reasoning**
   - Run the thinking showcase
   - Try complex problem-solving tasks
   - Explore mathematical reasoning

3. **Production Integration**
   - Implement conversation management
   - Add usage monitoring
   - Set up cost tracking
   - Create fallback mechanisms

4. **Optimization**
   - Benchmark different models
   - Optimize for your use case
   - Implement caching strategies
   - Monitor performance metrics

---

**Ready to explore Claude's capabilities? Start with the basic chat example! 🚀**

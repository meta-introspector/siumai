# 🏠 Ollama Local Provider

This directory contains examples demonstrating Ollama's local AI capabilities through the Siumai library. Ollama enables you to run AI models locally for privacy, cost control, and offline usage.

## 📋 Available Examples

### 🚀 [Basic Setup](./basic_setup.rs)
**Getting started with local AI using Ollama**

Features demonstrated:
- Ollama installation and setup
- Local model management
- Basic chat functionality with local models
- Configuration for different local models
- Performance optimization for local inference

```bash
# Run the basic setup example
cargo run --example ollama_basic_setup
```

**Key Learning Points:**
- How to set up Ollama locally
- Managing and switching between models
- Understanding local model capabilities
- Performance vs. quality trade-offs

## 🚀 Getting Started

### Prerequisites

1. **Install Ollama:**
   ```bash
   # Visit https://ollama.ai for installation instructions
   # Or use package managers:
   
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start Ollama service:**
   ```bash
   ollama serve
   ```

3. **Pull a model:**
   ```bash
   # Recommended starter models
   ollama pull llama3.2          # General purpose, 3B params
   ollama pull llama3.2:1b       # Lightweight, 1B params
   ollama pull codellama         # Code-focused model
   ollama pull mistral           # Alternative general model
   ```

### Quick Start

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.7)
        .build()
        .await?;

    let messages = vec![
        user!("Explain the benefits of local AI models")
    ];

    let response = ai.chat(messages).await?;
    println!("Local AI: {}", response.content_text().unwrap_or_default());
    
    Ok(())
}
```

## 🎯 Model Selection Guide

### Recommended Models

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|-------|---------|
| **llama3.2:1b** | 1B | Quick responses, simple tasks | ⭐⭐⭐ | ⭐⭐ |
| **llama3.2** | 3B | General purpose, balanced | ⭐⭐ | ⭐⭐⭐ |
| **llama3.1:8b** | 8B | High quality responses | ⭐ | ⭐⭐⭐ |
| **codellama** | 7B | Code generation and analysis | ⭐⭐ | ⭐⭐⭐ |
| **mistral** | 7B | Alternative general model | ⭐⭐ | ⭐⭐⭐ |

### Model Management

```bash
# List available models
ollama list

# Pull a new model
ollama pull model-name

# Remove a model
ollama rm model-name

# Show model information
ollama show model-name
```

## 🌟 Ollama's Unique Benefits

### Privacy & Security
- All processing happens locally
- No data sent to external servers
- Complete control over your data
- Offline capability

### Cost Control
- No per-token charges
- One-time setup cost
- Unlimited usage once installed
- No API rate limits

### Customization
- Fine-tune models for specific tasks
- Create custom model configurations
- Experiment without external costs
- Full control over model behavior

### Performance
- No network latency
- Consistent response times
- Scalable based on hardware
- No external dependencies

## ⚙️ Configuration & Optimization

### Hardware Requirements

**Minimum:**
- 8GB RAM for small models (1B-3B params)
- 4GB available disk space
- Modern CPU

**Recommended:**
- 16GB+ RAM for larger models
- GPU with CUDA support (optional)
- SSD storage for faster model loading

### Performance Tuning

```rust
// Optimize for speed
let fast_ai = LlmBuilder::new()
    .ollama()
    .model("llama3.2:1b")
    .temperature(0.3)
    .max_tokens(200)
    .build().await?;

// Optimize for quality
let quality_ai = LlmBuilder::new()
    .ollama()
    .model("llama3.1:8b")
    .temperature(0.7)
    .max_tokens(1000)
    .build().await?;
```

### Custom Parameters

Ollama supports specific parameters:
- `num_ctx`: Context window size
- `num_batch`: Batch size for processing
- `num_gpu`: Number of GPU layers
- `num_thread`: CPU threads to use

## 🔧 Troubleshooting

### Common Issues

**"Connection refused"**
- Ensure Ollama is running: `ollama serve`
- Check if port 11434 is available
- Verify firewall settings

**"Model not found"**
- Pull the model first: `ollama pull model-name`
- Check available models: `ollama list`
- Verify model name spelling

**Slow performance**
- Try smaller models (1B-3B params)
- Increase available RAM
- Use SSD storage
- Enable GPU acceleration if available

### Performance Tips

1. **Model Selection**: Start with smaller models
2. **Memory**: Ensure sufficient RAM
3. **Storage**: Use SSD for model storage
4. **GPU**: Enable GPU acceleration if available
5. **Context**: Limit context window size for speed

## 🔗 Related Resources

- [Ollama Official Website](https://ollama.ai/)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Model Library](https://ollama.ai/library)
- [Installation Guide](https://ollama.ai/download)

## 🎯 Next Steps

1. **Basic Setup**
   - Install and configure Ollama
   - Pull your first model
   - Run the basic setup example

2. **Model Exploration**
   - Try different model sizes
   - Compare performance vs. quality
   - Find the best model for your use case

3. **Integration**
   - Integrate with your applications
   - Implement error handling
   - Set up monitoring and logging

4. **Advanced Usage**
   - Explore custom model configurations
   - Set up GPU acceleration
   - Optimize for your specific hardware

---

**Ready to run AI locally? Start with [basic_setup.rs](basic_setup.rs)! 🏠**

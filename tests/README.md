# Real LLM Integration Tests

This directory contains comprehensive integration tests for all supported LLM providers. These tests use real API keys and make actual API calls, so they are ignored by default to prevent accidental usage during normal testing.

## 🚀 Quick Start

### Set Environment Variables

First, set the API keys for the providers you want to test:

```bash
# Required API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export OPENROUTER_API_KEY="your-openrouter-key"
export GROQ_API_KEY="your-groq-key"
export XAI_API_KEY="your-xai-key"

# Optional Base URL Overrides (for proxies/custom endpoints)
export OPENAI_BASE_URL="https://your-proxy.com/v1"
export ANTHROPIC_BASE_URL="https://your-proxy.com"
```

### Run Tests

```bash
# Test all available providers (skips providers without API keys)
cargo test test_all_available_providers -- --ignored

# Test specific providers
cargo test test_openai_integration -- --ignored
cargo test test_anthropic_integration -- --ignored
cargo test test_gemini_integration -- --ignored
cargo test test_deepseek_integration -- --ignored
cargo test test_openrouter_integration -- --ignored
cargo test test_groq_integration -- --ignored
cargo test test_xai_integration -- --ignored
```

## 📋 Test Coverage

Each provider test includes comprehensive coverage of:

### ✅ Non-streaming Chat
- Basic request/response functionality
- Message handling (system, user, assistant)
- Response content validation
- Usage statistics verification

### 🌊 Streaming Chat
- Real-time response streaming
- Content delta accumulation
- Thinking/reasoning content capture
- Stream completion handling
- Error handling

### 🔢 Embeddings (if supported)
- Text embedding generation
- Multiple text batch processing
- Dimension validation
- Usage statistics

### 🧠 Reasoning/Thinking (if supported)
- Advanced reasoning capabilities
- Model-specific reasoning features:
  - **OpenAI**: o1 models with reasoning tokens
  - **Anthropic**: Thinking with configurable budget
  - **Gemini**: Dynamic thinking
  - **DeepSeek**: Reasoner models
  - **OpenRouter**: o1 models through proxy
  - **xAI**: Grok reasoning capabilities

## 🏗️ Provider Capabilities Matrix

| Provider   | Chat | Streaming | Embeddings | Reasoning | Models Used |
|------------|------|-----------|------------|-----------|-------------|
| OpenAI     | ✅   | ✅        | ✅         | ✅        | gpt-4o-mini, o1-mini |
| Anthropic  | ✅   | ✅        | ❌         | ✅        | claude-3-5-haiku, claude-3-5-sonnet |
| Gemini     | ✅   | ✅        | ✅         | ✅        | gemini-1.5-flash, gemini-1.5-pro |
| DeepSeek   | ✅   | ✅        | ❌         | ✅        | deepseek-chat, deepseek-reasoner |
| OpenRouter | ✅   | ✅        | ❌         | ✅        | gpt-4o, gpt-4-turbo |
| Groq       | ✅   | ✅        | ❌         | ❌        | llama-3.1-8b |
| xAI        | ✅   | ✅        | ❌         | ✅        | grok-beta, grok-vision-beta |

## 🔧 Configuration

### Environment Variables

#### Required API Keys
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key  
- `GEMINI_API_KEY`: Google Gemini API key
- `DEEPSEEK_API_KEY`: DeepSeek API key
- `OPENROUTER_API_KEY`: OpenRouter API key
- `GROQ_API_KEY`: Groq API key
- `XAI_API_KEY`: xAI API key

#### Optional Base URL Overrides
- `OPENAI_BASE_URL`: Override OpenAI base URL (for proxies/custom endpoints) - only used if set
- `ANTHROPIC_BASE_URL`: Override Anthropic base URL - only used if set

### Test Behavior

- **Automatic Skipping**: Tests automatically skip providers without API keys
- **Comprehensive Logging**: Detailed output for each test phase
- **Error Handling**: Clear error messages and panic on failures
- **Usage Tracking**: Reports token usage and costs where available

## 📊 Example Output

```
🚀 Running integration tests for all available providers...

✅ Testing OpenAI provider...
  📝 Testing non-streaming chat for OpenAI...
    ✅ Non-streaming chat successful: 4
    📊 Usage: 15 prompt + 1 completion = 16 total tokens
  🌊 Testing streaming chat for OpenAI...
    ✅ Streaming chat successful
    📝 Accumulated content: 1
2
3
4
5
    📊 Usage: 18 prompt + 11 completion = 29 total tokens
  🔢 Testing embedding for OpenAI...
    ✅ Embedding successful: 2 embeddings with 1536 dimensions
    📊 Usage: 4 total tokens
  🧠 Testing OpenAI reasoning for OpenAI...
    ✅ OpenAI reasoning successful
    📝 Response: To solve this step by step...
    🧠 Reasoning tokens: 1247
    📊 Usage: 35 prompt + 89 completion = 124 total tokens

⏭️ Skipping Anthropic (no API key)
...

📊 Test Summary:
   Tested providers: ["OpenAI", "Gemini"]
   Skipped providers: ["Anthropic", "DeepSeek", "OpenRouter", "Groq", "xAI"]
   Total providers tested: 2/7
```

## 🛠️ Development

### Adding New Providers

1. Add provider configuration to `get_provider_configs()`
2. Add provider case to `test_provider_integration()`
3. Add individual test function if needed
4. Add reasoning test function if the provider supports it

### Modifying Test Cases

The test prompts are designed to be:
- **Brief**: To minimize API costs
- **Deterministic**: To produce consistent results
- **Comprehensive**: To test all major features

Feel free to modify the test prompts in the helper functions to better suit your testing needs.

## ⚠️ Important Notes

- **API Costs**: These tests make real API calls and will incur costs
- **Rate Limits**: Be aware of provider rate limits when running tests
- **Network Required**: Tests require internet connectivity
- **API Keys**: Never commit API keys to version control
- **Ignored by Default**: Tests are ignored to prevent accidental execution

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Key**: Set the required environment variable
2. **Invalid API Key**: Check your API key is correct and has sufficient permissions
3. **Rate Limiting**: Wait and retry, or use different API keys
4. **Network Issues**: Check internet connectivity
5. **Model Unavailable**: Some models may not be available in all regions

### Debug Mode

For more detailed output, run tests with:

```bash
RUST_LOG=debug cargo test test_all_available_providers -- --ignored --nocapture
```

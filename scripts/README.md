# Scripts

This directory contains utility scripts for the Siumai project.

## 🧪 Integration Test Scripts

### `run_integration_tests.sh` (Linux/macOS)

Interactive script to run real LLM integration tests with environment setup.

**Usage:**
```bash
./scripts/run_integration_tests.sh
```

**Features:**
- Automatically loads `.env` file if present
- Checks for existing API keys
- Prompts for missing API keys interactively
- Shows configuration summary
- Provides test selection menu
- Handles optional base URL overrides

### `run_integration_tests.bat` (Windows)

Windows batch file version of the integration test runner.

**Usage:**
```cmd
scripts\run_integration_tests.bat
```

**Features:**
- Automatically loads `.env` file if present
- Same functionality as the shell script
- Windows-compatible batch commands
- Interactive prompts for API keys
- Test selection menu

## 🔧 Prerequisites

Before running the integration test scripts:

1. **Rust and Cargo**: Ensure you have Rust installed
2. **API Keys**: Have your LLM provider API keys ready
3. **Internet Connection**: Tests make real API calls

## 📋 Supported Providers

The scripts support testing with these providers:

| Provider   | Environment Variable | Required |
|------------|---------------------|----------|
| OpenAI     | `OPENAI_API_KEY`    | No       |
| Anthropic  | `ANTHROPIC_API_KEY` | No       |
| Gemini     | `GEMINI_API_KEY`    | No       |
| DeepSeek   | `DEEPSEEK_API_KEY`  | No       |
| OpenRouter | `OPENROUTER_API_KEY`| No       |
| Groq       | `GROQ_API_KEY`      | No       |
| xAI        | `XAI_API_KEY`       | No       |

**Note:** You only need API keys for providers you want to test. The scripts will automatically skip providers without API keys.

## 🔧 Optional Configuration

### Base URL Overrides

For proxy or custom endpoint usage:

```bash
# OpenAI custom endpoint
export OPENAI_BASE_URL="https://your-proxy.com/v1"

# Anthropic custom endpoint  
export ANTHROPIC_BASE_URL="https://your-proxy.com"
```

These are only used if the environment variables are set.

## 🚀 Quick Start

1. **Set up environment variables** (optional):
   ```bash
   # Option 1: Create .env file from template
   cp .env.example .env
   # Edit .env file with your API keys

   # Option 2: Export variables directly
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   ```

2. **Make script executable** (Linux/macOS only):
   ```bash
   chmod +x scripts/run_integration_tests.sh
   ```

3. **Run the script**:
   ```bash
   # Linux/macOS
   ./scripts/run_integration_tests.sh
   
   # Windows
   scripts\run_integration_tests.bat
   ```

4. **Follow the prompts**:
   - Enter API keys when prompted (or skip)
   - Choose test type from menu
   - Review test results

## 💡 Tips

- **Start with one provider**: Test with just OpenAI or Anthropic first
- **Check API limits**: Be aware of rate limits and costs
- **Use test keys**: Consider using separate API keys for testing
- **Monitor usage**: Check your API usage after running tests

## 🔍 Troubleshooting

### Common Issues

1. **Permission denied** (Linux/macOS):
   ```bash
   chmod +x scripts/run_integration_tests.sh
   ```

2. **API key errors**: 
   - Verify your API keys are correct
   - Check if your account has sufficient credits
   - Ensure API keys have required permissions

3. **Network issues**:
   - Check internet connectivity
   - Verify firewall settings
   - Try with base URL overrides if using proxies

### Getting Help

If you encounter issues:

1. Check the test output for specific error messages
2. Verify your API keys and account status
3. Review the [main documentation](../README.md)
4. Check the [test documentation](../tests/README.md)

## 📁 File Structure

```
scripts/
├── README.md                    # This file
├── run_integration_tests.sh     # Linux/macOS script
└── run_integration_tests.bat    # Windows script
```

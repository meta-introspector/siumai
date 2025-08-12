#!/bin/bash

# Ollama Integration Test Script
# This script specifically tests Ollama functionality including streaming, non-streaming, thinking, and embeddings

set -e

echo "🦙 Ollama Integration Test Script"
echo "================================"
echo ""

# Configuration
OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-"http://localhost:11434"}
CHAT_MODEL=${OLLAMA_CHAT_MODEL:-"llama3.2:3b"}
REASONING_MODEL=${OLLAMA_REASONING_MODEL:-"deepseek-r1:8b"}
EMBEDDING_MODEL=${OLLAMA_EMBEDDING_MODEL:-"nomic-embed-text"}

echo "📋 Configuration:"
echo "   Base URL: $OLLAMA_BASE_URL"
echo "   Chat Model: $CHAT_MODEL"
echo "   Reasoning Model: $REASONING_MODEL"
echo "   Embedding Model: $EMBEDDING_MODEL"
echo ""

# Function to check if Ollama is running
check_ollama() {
    echo "🔍 Checking if Ollama is running..."
    if curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
        echo "✅ Ollama is running at $OLLAMA_BASE_URL"
        return 0
    else
        echo "❌ Ollama is not running at $OLLAMA_BASE_URL"
        echo ""
        echo "💡 To start Ollama:"
        echo "   ollama serve"
        echo ""
        return 1
    fi
}

# Function to check if a model is available
check_model() {
    local model=$1
    local model_type=$2
    
    echo "🔍 Checking if $model_type model '$model' is available..."
    if curl -s "$OLLAMA_BASE_URL/api/tags" | grep -q "\"name\":\"$model\""; then
        echo "✅ Model '$model' is available"
        return 0
    else
        echo "❌ Model '$model' is not available"
        echo "💡 To install: ollama pull $model"
        return 1
    fi
}

# Function to pull a model if not available
ensure_model() {
    local model=$1
    local model_type=$2
    
    if ! check_model "$model" "$model_type"; then
        echo "📥 Pulling $model_type model '$model'..."
        if command -v ollama > /dev/null 2>&1; then
            ollama pull "$model"
            if [ $? -eq 0 ]; then
                echo "✅ Successfully pulled '$model'"
            else
                echo "❌ Failed to pull '$model'"
                return 1
            fi
        else
            echo "❌ 'ollama' command not found. Please install Ollama CLI."
            return 1
        fi
    fi
    return 0
}

# Function to run a specific test
run_test() {
    local test_name=$1
    echo ""
    echo "🧪 Running $test_name..."
    echo "----------------------------------------"
    
    export OLLAMA_BASE_URL="$OLLAMA_BASE_URL"
    
    if cargo test "$test_name" -- --ignored --nocapture; then
        echo "✅ $test_name passed"
    else
        echo "❌ $test_name failed"
        return 1
    fi
}

# Main execution
main() {
    # Check if Ollama is running
    if ! check_ollama; then
        exit 1
    fi
    
    echo ""
    echo "📦 Checking required models..."
    
    # Track which models are available
    chat_available=true
    reasoning_available=true
    embedding_available=true
    
    # Check chat model
    if ! check_model "$CHAT_MODEL" "chat"; then
        echo "⚠️ Chat model not available"
        chat_available=false
    fi
    
    # Check reasoning model
    if ! check_model "$REASONING_MODEL" "reasoning"; then
        echo "⚠️ Reasoning model not available"
        reasoning_available=false
    fi
    
    # Check embedding model
    if ! check_model "$EMBEDDING_MODEL" "embedding"; then
        echo "⚠️ Embedding model not available"
        embedding_available=false
    fi
    
    echo ""
    echo "🤖 Model availability summary:"
    echo "   Chat: $([ "$chat_available" = true ] && echo "✅" || echo "❌") $CHAT_MODEL"
    echo "   Reasoning: $([ "$reasoning_available" = true ] && echo "✅" || echo "❌") $REASONING_MODEL"
    echo "   Embedding: $([ "$embedding_available" = true ] && echo "✅" || echo "❌") $EMBEDDING_MODEL"
    echo ""
    
    # Ask user if they want to pull missing models
    missing_models=()
    [ "$chat_available" = false ] && missing_models+=("$CHAT_MODEL")
    [ "$reasoning_available" = false ] && missing_models+=("$REASONING_MODEL")
    [ "$embedding_available" = false ] && missing_models+=("$EMBEDDING_MODEL")
    
    if [ ${#missing_models[@]} -gt 0 ]; then
        echo "❓ Some models are missing. Would you like to pull them automatically?"
        echo "   Missing models: ${missing_models[*]}"
        echo ""
        read -p "Pull missing models? (y/N): " pull_models
        
        if [[ "$pull_models" =~ ^[Yy]$ ]]; then
            echo ""
            echo "📥 Pulling missing models..."
            
            for model in "${missing_models[@]}"; do
                echo "📥 Pulling $model..."
                if ollama pull "$model"; then
                    echo "✅ Successfully pulled $model"
                else
                    echo "❌ Failed to pull $model"
                fi
            done
            
            echo ""
            echo "🔄 Re-checking model availability..."
            check_model "$CHAT_MODEL" "chat" && chat_available=true
            check_model "$REASONING_MODEL" "reasoning" && reasoning_available=true
            check_model "$EMBEDDING_MODEL" "embedding" && embedding_available=true
        fi
    fi
    
    echo ""
    echo "🧪 Running Ollama integration tests..."
    echo ""
    
    # Set environment variables for the tests
    export OLLAMA_BASE_URL="$OLLAMA_BASE_URL"
    export OLLAMA_CHAT_MODEL="$CHAT_MODEL"
    export OLLAMA_REASONING_MODEL="$REASONING_MODEL"
    export OLLAMA_EMBEDDING_MODEL="$EMBEDDING_MODEL"
    
    # Run the main Ollama integration test
    if run_test "test_ollama_integration"; then
        echo ""
        echo "🎉 All Ollama tests completed successfully!"
    else
        echo ""
        echo "❌ Some Ollama tests failed. Check the output above for details."
        exit 1
    fi
    
    echo ""
    echo "📊 Test Summary:"
    echo "   ✅ Non-streaming chat: Tested"
    echo "   ✅ Streaming chat: Tested"
    echo "   $([ "$embedding_available" = true ] && echo "✅" || echo "⏭️") Embeddings: $([ "$embedding_available" = true ] && echo "Tested" || echo "Skipped (model not available)")"
    echo "   $([ "$reasoning_available" = true ] && echo "✅" || echo "⏭️") Reasoning: $([ "$reasoning_available" = true ] && echo "Tested" || echo "Skipped (model not available)")"
    echo ""
    echo "💡 Tips:"
    echo "   - Use simple questions to save time and tokens"
    echo "   - Models are cached locally after first pull"
    echo "   - Check 'ollama list' to see installed models"
    echo "   - Use 'ollama rm <model>' to remove unused models"
    echo ""
}

# Run main function
main "$@"

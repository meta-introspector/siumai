@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🦙 Ollama Integration Test Script
echo ================================
echo.

REM Configuration
if not defined OLLAMA_BASE_URL set OLLAMA_BASE_URL=http://localhost:11434
if not defined OLLAMA_CHAT_MODEL set OLLAMA_CHAT_MODEL=llama3.2:3b
if not defined OLLAMA_REASONING_MODEL set OLLAMA_REASONING_MODEL=deepseek-r1:8b
if not defined OLLAMA_EMBEDDING_MODEL set OLLAMA_EMBEDDING_MODEL=nomic-embed-text

echo 📋 Configuration:
echo    Base URL: !OLLAMA_BASE_URL!
echo    Chat Model: !OLLAMA_CHAT_MODEL!
echo    Reasoning Model: !OLLAMA_REASONING_MODEL!
echo    Embedding Model: !OLLAMA_EMBEDDING_MODEL!
echo.

REM Function to check if Ollama is running
echo 🔍 Checking if Ollama is running...
curl -s "!OLLAMA_BASE_URL!/api/tags" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Ollama is running at !OLLAMA_BASE_URL!
) else (
    echo ❌ Ollama is not running at !OLLAMA_BASE_URL!
    echo.
    echo 💡 To start Ollama:
    echo    ollama serve
    echo.
    pause
    exit /b 1
)

echo.
echo 📦 Checking required models...

REM Check chat model
set chat_available=false
curl -s "!OLLAMA_BASE_URL!/api/tags" | findstr "!OLLAMA_CHAT_MODEL!" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Chat model '!OLLAMA_CHAT_MODEL!' is available
    set chat_available=true
) else (
    echo ❌ Chat model '!OLLAMA_CHAT_MODEL!' is not available
    echo 💡 To install: ollama pull !OLLAMA_CHAT_MODEL!
)

REM Check reasoning model
set reasoning_available=false
curl -s "!OLLAMA_BASE_URL!/api/tags" | findstr "!OLLAMA_REASONING_MODEL!" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Reasoning model '!OLLAMA_REASONING_MODEL!' is available
    set reasoning_available=true
) else (
    echo ❌ Reasoning model '!OLLAMA_REASONING_MODEL!' is not available
    echo 💡 To install: ollama pull !OLLAMA_REASONING_MODEL!
)

REM Check embedding model
set embedding_available=false
curl -s "!OLLAMA_BASE_URL!/api/tags" | findstr "!OLLAMA_EMBEDDING_MODEL!" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Embedding model '!OLLAMA_EMBEDDING_MODEL!' is available
    set embedding_available=true
) else (
    echo ❌ Embedding model '!OLLAMA_EMBEDDING_MODEL!' is not available
    echo 💡 To install: ollama pull !OLLAMA_EMBEDDING_MODEL!
)

echo.
echo 🤖 Model availability summary:
if "!chat_available!"=="true" (
    echo    Chat: ✅ !OLLAMA_CHAT_MODEL!
) else (
    echo    Chat: ❌ !OLLAMA_CHAT_MODEL!
)
if "!reasoning_available!"=="true" (
    echo    Reasoning: ✅ !OLLAMA_REASONING_MODEL!
) else (
    echo    Reasoning: ❌ !OLLAMA_REASONING_MODEL!
)
if "!embedding_available!"=="true" (
    echo    Embedding: ✅ !OLLAMA_EMBEDDING_MODEL!
) else (
    echo    Embedding: ❌ !OLLAMA_EMBEDDING_MODEL!
)
echo.

REM Check if any models are missing
set missing_models=
if "!chat_available!"=="false" set missing_models=!missing_models! !OLLAMA_CHAT_MODEL!
if "!reasoning_available!"=="false" set missing_models=!missing_models! !OLLAMA_REASONING_MODEL!
if "!embedding_available!"=="false" set missing_models=!missing_models! !OLLAMA_EMBEDDING_MODEL!

if not "!missing_models!"=="" (
    echo ❓ Some models are missing. Would you like to pull them automatically?
    echo    Missing models:!missing_models!
    echo.
    set /p "pull_models=Pull missing models? (y/N): "
    
    if /i "!pull_models!"=="y" (
        echo.
        echo 📥 Pulling missing models...
        
        if "!chat_available!"=="false" (
            echo 📥 Pulling !OLLAMA_CHAT_MODEL!...
            ollama pull "!OLLAMA_CHAT_MODEL!"
            if !errorlevel! equ 0 (
                echo ✅ Successfully pulled !OLLAMA_CHAT_MODEL!
                set chat_available=true
            ) else (
                echo ❌ Failed to pull !OLLAMA_CHAT_MODEL!
            )
        )
        
        if "!reasoning_available!"=="false" (
            echo 📥 Pulling !OLLAMA_REASONING_MODEL!...
            ollama pull "!OLLAMA_REASONING_MODEL!"
            if !errorlevel! equ 0 (
                echo ✅ Successfully pulled !OLLAMA_REASONING_MODEL!
                set reasoning_available=true
            ) else (
                echo ❌ Failed to pull !OLLAMA_REASONING_MODEL!
            )
        )
        
        if "!embedding_available!"=="false" (
            echo 📥 Pulling !OLLAMA_EMBEDDING_MODEL!...
            ollama pull "!OLLAMA_EMBEDDING_MODEL!"
            if !errorlevel! equ 0 (
                echo ✅ Successfully pulled !OLLAMA_EMBEDDING_MODEL!
                set embedding_available=true
            ) else (
                echo ❌ Failed to pull !OLLAMA_EMBEDDING_MODEL!
            )
        )
    )
)

echo.
echo 🧪 Running Ollama integration tests...
echo.

REM Run the main Ollama integration test
echo 🧪 Running test_ollama_integration...
echo ----------------------------------------

cargo test test_ollama_integration -- --ignored --nocapture
if !errorlevel! equ 0 (
    echo ✅ test_ollama_integration passed
    echo.
    echo 🎉 All Ollama tests completed successfully!
) else (
    echo ❌ test_ollama_integration failed
    echo.
    echo ❌ Some Ollama tests failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo 📊 Test Summary:
echo    ✅ Non-streaming chat: Tested
echo    ✅ Streaming chat: Tested
if "!embedding_available!"=="true" (
    echo    ✅ Embeddings: Tested
) else (
    echo    ⏭️ Embeddings: Skipped ^(model not available^)
)
if "!reasoning_available!"=="true" (
    echo    ✅ Reasoning: Tested
) else (
    echo    ⏭️ Reasoning: Skipped ^(model not available^)
)
echo.
echo 💡 Tips:
echo    - Use simple questions to save time and tokens
echo    - Models are cached locally after first pull
echo    - Check 'ollama list' to see installed models
echo    - Use 'ollama rm ^<model^>' to remove unused models
echo.
pause

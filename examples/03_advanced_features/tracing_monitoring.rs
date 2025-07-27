//! Comprehensive Tracing and Monitoring Example
//!
//! This example demonstrates all available tracing and debugging capabilities in siumai:
//! 1. Built-in automatic debug logging (zero configuration)
//! 2. reqwest HTTP-level debugging
//! 3. Combined tracing approaches
//! 4. Production-ready structured logging
//!
//! ## Usage Methods
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//!
//! # Method 1: Built-in siumai debug logging (recommended)
//! SIUMAI_LOG_LEVEL=debug cargo run --example tracing_monitoring
//!
//! # Method 2: HTTP-level debugging with reqwest
//! RUST_LOG=reqwest=debug cargo run --example tracing_monitoring
//!
//! # Method 3: Combined approach (maximum visibility)
//! RUST_LOG=reqwest=debug SIUMAI_LOG_LEVEL=debug cargo run --example tracing_monitoring
//!
//! # Method 4: Production JSON logging
//! SIUMAI_LOG_LEVEL=info SIUMAI_LOG_FORMAT=json SIUMAI_LOG_FILE=app.log cargo run --example tracing_monitoring
//! ```

use futures_util::stream::StreamExt;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Note: We'll use builder-based tracing configuration instead of global initialization
    // let _guard = init_tracing_from_env()?;

    println!("🔍 Comprehensive Tracing and Monitoring Example");
    println!("===============================================");
    println!();

    // Check for API key
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("⚠️  OPENAI_API_KEY not set, using demo key (will show tracing)");
        "demo-key".to_string()
    });

    // Show current tracing configuration
    show_tracing_configuration();

    // Create OpenAI client with builder-based tracing configuration
    let client = Provider::openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .debug_tracing() // 🎯 Using debug tracing for development!
        .pretty_json(true)
        .temperature(0.7)
        .build()
        .await?;

    // Demonstrate different tracing scenarios
    println!("📊 Scenario 1: Basic Chat with Automatic Tracing");
    demonstrate_basic_tracing(&client).await?;
    println!();

    println!("🔧 Scenario 2: Tool Usage Tracing");
    demonstrate_tool_tracing(&client).await?;
    println!();

    println!("⚡ Scenario 3: Streaming with Automatic Tracing");
    demonstrate_streaming_tracing(&client).await?;
    println!();

    println!("🎯 Scenario 4: Different Tracing Configuration Methods");
    demonstrate_tracing_configurations(&api_key).await?;
    println!();

    // Show summary
    show_tracing_summary();

    Ok(())
}

/// Show current tracing configuration
fn show_tracing_configuration() {
    println!("🔧 Current Tracing Configuration:");

    if let Ok(level) = std::env::var("SIUMAI_LOG_LEVEL") {
        println!("  📊 SIUMAI_LOG_LEVEL: {level}");
    } else {
        println!("  📊 SIUMAI_LOG_LEVEL: info (default)");
    }

    if let Ok(format) = std::env::var("SIUMAI_LOG_FORMAT") {
        println!("  📝 SIUMAI_LOG_FORMAT: {format}");
    } else {
        println!("  📝 SIUMAI_LOG_FORMAT: text (default)");
    }

    if let Ok(file) = std::env::var("SIUMAI_LOG_FILE") {
        println!("  📁 SIUMAI_LOG_FILE: {file}");
    }

    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        println!("  🌐 RUST_LOG: {rust_log} (HTTP-level tracing)");
    }

    println!();
    println!("💡 Available tracing methods:");
    println!("  1. Environment variables: SIUMAI_LOG_LEVEL=debug");
    println!("  2. Builder methods: .debug_tracing(), .minimal_tracing(), .json_tracing()");
    println!("  3. Custom configuration: .tracing(TracingConfig::custom())");
    println!("  4. HTTP-level tracing: RUST_LOG=reqwest=debug");
    println!();

    // Note: Tracing configuration methods will be demonstrated in the scenarios below
}

/// Demonstrate basic automatic tracing
async fn demonstrate_basic_tracing(
    client: &impl ChatCapability,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Watch the automatic tracing output above and below this message:");
    println!("   ✅ Request start with clean, readable format");
    println!("   ✅ Complete request body with all parameters");
    println!("   ✅ Response timing and error details");
    println!();

    let messages = vec![
        system!("You are a helpful assistant that explains technical concepts clearly."),
        user!("What is automatic debug logging in software development?"),
    ];

    // This single call automatically generates:
    // - INFO: Starting OpenAI chat request (with structured context)
    // - DEBUG: Sending OpenAI API request (with full request body)
    // - ERROR/INFO: Response with timing and status
    let result = client.chat(messages).await;

    match result {
        Ok(response) => {
            println!("✅ Request successful! Check the tracing output above for:");
            println!("   📊 Clean, readable log format");
            println!("   📝 Complete request body with all messages and parameters");
            println!("   ⏱️  Automatic timing: duration_ms field");
            if let Some(text) = response.text() {
                let preview = if text.len() > 100 {
                    format!("{}...", &text[..100])
                } else {
                    text.clone()
                };
                println!("🤖 Response: {preview}");
            }
        }
        Err(e) => {
            println!("❌ Request failed: {e}");
            println!("💡 Notice how the tracing automatically captured:");
            println!("   🔴 HTTP status code (401)");
            println!("   📄 Complete error response body");
            println!("   ⏱️  Request duration (even for failed requests)");
            println!("   🏷️  Structured context for easy filtering");
        }
    }

    Ok(())
}

/// Demonstrate tool usage tracing
async fn demonstrate_tool_tracing(
    client: &impl ChatCapability,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Watch how tools are automatically traced in the request body:");
    println!("   ✅ Tool definitions automatically included in request body");
    println!("   ✅ Complete tool schemas and parameters");
    println!("   ✅ Clean, readable tracing output");
    println!();

    let weather_tool = Tool {
        r#type: "function".to_string(),
        function: ToolFunction {
            name: "get_weather".to_string(),
            description: "Get weather information for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }),
        },
    };

    let messages = vec![
        system!(
            "You are a weather assistant. Use the get_weather function when asked about weather."
        ),
        user!("What's the weather like in Tokyo?"),
    ];

    // This automatically traces:
    // - tool_count: 1 in the span context
    // - Complete tool definitions in the request body
    // - Tool schemas and parameters
    let result = client
        .chat_with_tools(messages, Some(vec![weather_tool]))
        .await;

    match result {
        Ok(response) => {
            println!("✅ Tool request successful! Notice in the tracing output:");
            println!("   🛠️  Complete tool definitions in request_body");
            println!("   📋 Tool parameters schema automatically included");
            println!("   📊 Clean, readable log format");
            if response.has_tool_calls() {
                println!("🔧 Tool calls were made (check debug logs for details)");
            }
        }
        Err(e) => {
            println!("❌ Tool request failed: {e}");
            println!("💡 Even failed requests show complete tool tracing:");
            println!("   🛠️  Full tool definitions captured in request body");
            println!("   ⏱️  Request timing still recorded");
            println!("   📊 Clean, readable error output");
        }
    }

    Ok(())
}

/// Demonstrate streaming with automatic tracing
async fn demonstrate_streaming_tracing(
    client: &impl ChatCapability,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Streaming requests also get automatic tracing:");
    println!("   ✅ Stream start and end events");
    println!("   ✅ Individual chunk processing (if stream tracing enabled)");
    println!("   ✅ Total streaming duration");
    println!();

    let messages = vec![user!("Write a short poem about Rust programming language")];

    // Streaming requests are also automatically traced
    match client.chat_stream(messages, None).await {
        Ok(mut stream) => {
            println!("✅ Stream started! Watch for automatic tracing of:");
            println!("   📡 Stream initialization");
            println!("   🔄 Chunk processing (if enabled)");
            println!("   ⏱️  Total streaming duration");

            let mut chunk_count = 0;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(_) => {
                        chunk_count += 1;
                        if chunk_count <= 3 {
                            print!(".");
                        }
                    }
                    Err(_) => break,
                }
            }
            println!("\n📊 Processed {chunk_count} chunks (all automatically traced)");
        }
        Err(e) => {
            println!("❌ Streaming failed: {e}");
            println!("💡 Even failed streams show tracing:");
            println!("   🔴 Stream initialization errors");
            println!("   ⏱️  Time to failure");
            println!("   📋 Complete request context");
        }
    }

    Ok(())
}

/// Show tracing summary and recommendations
fn show_tracing_summary() {
    println!("🎯 Built-in Tracing Summary");
    println!("===========================");
    println!();
    println!("🔍 What siumai automatically traces (zero manual instrumentation):");
    println!("  ✅ Request details: URL, method, headers, timing");
    println!("  ✅ Complete request bodies: messages, parameters, tool definitions");
    println!("  ✅ Response details: content, token usage, status codes");
    println!("  ✅ Error information: HTTP status, complete error responses");
    println!("  ✅ Performance metrics: automatic request timing (duration_ms)");
    println!("  ✅ Source location: file, line number, function name");
    println!("  ✅ Thread information: thread ID and name");
    println!();
    println!("🚀 Key Advantages of Built-in Tracing:");
    println!("  ✅ Zero manual instrumentation - no debug!() or info!() calls needed");
    println!("  ✅ Consistent structured format across all requests");
    println!("  ✅ Automatic performance monitoring without timing code");
    println!("  ✅ Production-ready with multiple output formats");
    println!("  ✅ Environment variable controlled - no code changes");
    println!("  ✅ Clean, readable output without verbose span information");
    println!();
    println!("💡 Available Output Formats:");
    println!("  � text: Human-readable for development");
    println!("  📊 json: Single-line structured for production");
    println!("  � json-compact: Minimal size for storage optimization");
    println!("  � json-compact: Minimal size for storage optimization");
    println!();
    println!("🔧 Configuration Methods:");
    println!("  🌍 Environment: SIUMAI_LOG_LEVEL=debug SIUMAI_LOG_FORMAT=json-compact");
    println!("  🏗️  Builder: .debug_tracing(), .json_tracing(), .minimal_tracing()");
    println!("  ⚙️  Custom: .tracing(TracingConfig::builder()...build())");
    println!();
    println!("📚 The beauty: You get comprehensive tracing without writing any tracing code!");
}

/// Demonstrate different tracing configuration methods
async fn demonstrate_tracing_configurations(
    _api_key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Available Tracing Configuration Methods:");
    println!();

    println!("✅ Currently Active: .debug_tracing() (as used in the main client above)");
    println!("   🎨 Produces human-readable text formatted output");
    println!("   📊 Includes request details: URL, timing, status codes");
    println!("   📝 Shows complete request bodies and error details");
    println!("   ⏱️  Automatic performance timing (duration_ms)");
    println!();

    println!("🔧 Other Available Builder Methods:");
    println!("   .minimal_tracing()     - Info level, LLM only, minimal output");
    println!("   .json_tracing()        - Production JSON, warn level");
    println!("   .enable_tracing()      - Alias for debug_tracing()");
    println!("   .disable_tracing()     - Explicitly disable all tracing");
    println!();

    println!("⚙️  Custom Configuration Example:");
    println!("   .tracing(TracingConfig::builder()");
    println!("       .log_level_str(\"info\")?");
    println!("       .output_format(OutputFormat::JsonCompact)");
    println!("       .enable_performance_monitoring(true)");
    println!("       .build())");
    println!();

    println!("🌐 Unified Interface Support:");
    println!("   Siumai::builder().openai().debug_tracing().build()");
    println!(
        "   (Same methods available on both Provider::openai() and Siumai::builder().openai())"
    );
    println!();

    println!("💡 Key Benefits:");
    println!("   ✅ Zero manual instrumentation - no debug!() calls needed");
    println!("   ✅ Consistent structured output across all configuration methods");
    println!("   ✅ Client-specific configuration (each client can have different tracing)");
    println!("   ✅ Automatic performance monitoring without timing code");
    println!("   ✅ Production-ready with multiple output formats");
    println!();

    println!("🔍 What You Saw Above:");
    println!("   📊 Clean, readable log format");
    println!("   📝 Complete request bodies (messages, tools, parameters)");
    println!("   🔴 Detailed error information (status codes, error messages)");
    println!("   ⏱️  Automatic timing (duration_ms field)");
    println!("   🛠️  Tool definitions automatically captured");

    Ok(())
}

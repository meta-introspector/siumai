//! 🔍 Capability Detection - Feature detection and graceful degradation
//!
//! This example demonstrates how to detect and adapt to provider capabilities:
//! - Runtime capability detection
//! - Feature availability checking
//! - Graceful degradation when features aren't available
//! - Provider metadata and limitations
//!
//! Before running, set your API keys:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example capability_detection
//! ```

use siumai::prelude::*;
use siumai::traits::ChatCapability;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Capability Detection - Feature detection and graceful degradation\n");

    // Demonstrate different aspects of capability detection
    demonstrate_basic_capability_detection().await;
    demonstrate_feature_availability().await;
    demonstrate_graceful_feature_degradation().await;
    demonstrate_provider_metadata().await;
    demonstrate_adaptive_behavior().await;

    println!("\n✅ Capability detection examples completed!");
    Ok(())
}

/// Demonstrate basic capability detection
async fn demonstrate_basic_capability_detection() {
    println!("🔍 Basic Capability Detection:\n");

    let providers = create_test_providers().await;
    
    for (name, client) in providers {
        println!("   Provider: {}", name);
        
        let capabilities = detect_capabilities(client.as_ref(), &name).await;
        
        println!("      📋 Detected Capabilities:");
        for (feature, supported) in capabilities {
            let status = if supported { "✅" } else { "❌" };
            println!("         {} {}", status, feature);
        }
        println!();
    }
}

/// Demonstrate feature availability checking
async fn demonstrate_feature_availability() {
    println!("🎯 Feature Availability Checking:\n");

    let providers = create_test_providers().await;
    
    let features_to_test = vec![
        "streaming",
        "vision",
        "audio",
        "tools",
        "json_mode",
        "thinking",
    ];

    for feature in features_to_test {
        println!("   Feature: {}", feature);
        
        for (name, client) in &providers {
            let available = check_feature_availability(client.as_ref(), &name, feature).await;
            let status = if available { "✅" } else { "❌" };
            println!("      {} {}", status, name);
        }
        println!();
    }
}

/// Demonstrate graceful feature degradation
async fn demonstrate_graceful_feature_degradation() {
    println!("🎭 Graceful Feature Degradation:\n");

    let providers = create_test_providers().await;
    
    if let Some((name, client)) = providers.into_iter().next() {
        println!("   Using provider: {}", name);
        
        // Test streaming with fallback
        println!("   Testing streaming with fallback:");
        match try_streaming_with_fallback(client.as_ref()).await {
            Ok(response) => {
                println!("      ✅ Got response: {}", &response[..response.len().min(100)]);
            }
            Err(e) => {
                println!("      ❌ Failed: {}", e);
            }
        }
        
        // Test vision with fallback
        println!("\n   Testing vision with fallback:");
        match try_vision_with_fallback(client.as_ref()).await {
            Ok(response) => {
                println!("      ✅ Got response: {}", &response[..response.len().min(100)]);
            }
            Err(e) => {
                println!("      ❌ Failed: {}", e);
            }
        }
    } else {
        println!("   ⚠️  No providers available for testing");
    }
    
    println!();
}

/// Demonstrate provider metadata detection
async fn demonstrate_provider_metadata() {
    println!("📊 Provider Metadata:\n");

    let providers = create_test_providers().await;
    
    for (name, client) in providers {
        println!("   Provider: {}", name);
        
        let metadata = get_provider_metadata(client.as_ref(), &name).await;
        
        println!("      📋 Metadata:");
        println!("         Type: {}", metadata.provider_type);
        println!("         Model: {}", metadata.model);
        println!("         Max tokens: {}", metadata.max_tokens.unwrap_or_default());
        println!("         Context window: {}", metadata.context_window);
        println!("         Supports streaming: {}", metadata.supports_streaming);
        println!("         Supports vision: {}", metadata.supports_vision);
        println!("         Cost per 1K tokens: ${:.4}", metadata.cost_per_1k_tokens);
        println!();
    }
}

/// Demonstrate adaptive behavior based on capabilities
async fn demonstrate_adaptive_behavior() {
    println!("🤖 Adaptive Behavior:\n");

    let providers = create_test_providers().await;
    
    if let Some((name, client)) = providers.into_iter().next() {
        println!("   Using provider: {}", name);
        
        // Adapt behavior based on capabilities
        let message = "Explain machine learning in simple terms";
        
        match adaptive_chat(client.as_ref(), &name, message).await {
            Ok(response) => {
                println!("   ✅ Adaptive response received");
                println!("   Response: {}", &response[..response.len().min(150)]);
            }
            Err(e) => {
                println!("   ❌ Adaptive chat failed: {}", e);
            }
        }
    } else {
        println!("   ⚠️  No providers available for testing");
    }
    
    println!();
}

/// Create test providers for capability detection
async fn create_test_providers() -> Vec<(String, Box<dyn ChatCapability + Send + Sync>)> {
    let mut providers = Vec::new();

    // Try OpenAI
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        if let Ok(client) = LlmBuilder::new()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .build()
            .await
        {
            providers.push(("OpenAI".to_string(), Box::new(client) as Box<dyn ChatCapability + Send + Sync>));
        }
    }

    // Try Anthropic
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        if let Ok(client) = LlmBuilder::new()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-haiku-20241022")
            .build()
            .await
        {
            providers.push(("Anthropic".to_string(), Box::new(client) as Box<dyn ChatCapability + Send + Sync>));
        }
    }

    // Try Ollama
    if let Ok(client) = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .build()
        .await
    {
        // Test if Ollama is actually available
        let test_messages = vec![user!("Hi")];
        if client.chat(test_messages).await.is_ok() {
            providers.push(("Ollama".to_string(), Box::new(client) as Box<dyn ChatCapability + Send + Sync>));
        }
    }

    providers
}

/// Detect capabilities of a provider
async fn detect_capabilities(client: &dyn ChatCapability, provider_name: &str) -> HashMap<String, bool> {
    let mut capabilities = HashMap::new();

    // Basic chat (all providers should support this)
    capabilities.insert("Basic Chat".to_string(), true);

    // Streaming support
    capabilities.insert("Streaming".to_string(), test_streaming_support(client).await);

    // Vision support (simplified detection)
    capabilities.insert("Vision".to_string(), provider_supports_vision(provider_name));

    // Audio support
    capabilities.insert("Audio".to_string(), provider_supports_audio(provider_name));

    // Tool calling
    capabilities.insert("Tools".to_string(), provider_supports_tools(provider_name));

    // JSON mode
    capabilities.insert("JSON Mode".to_string(), provider_supports_json_mode(provider_name));

    // Thinking process
    capabilities.insert("Thinking".to_string(), provider_supports_thinking(provider_name));

    capabilities
}

/// Check if a specific feature is available
async fn check_feature_availability(client: &dyn ChatCapability, provider_name: &str, feature: &str) -> bool {
    match feature {
        "streaming" => test_streaming_support(client).await,
        "vision" => provider_supports_vision(provider_name),
        "audio" => provider_supports_audio(provider_name),
        "tools" => provider_supports_tools(provider_name),
        "json_mode" => provider_supports_json_mode(provider_name),
        "thinking" => provider_supports_thinking(provider_name),
        _ => false,
    }
}

/// Test streaming support
async fn test_streaming_support(client: &dyn ChatCapability) -> bool {
    // In a real implementation, you would try to create a stream
    // For now, we'll assume all providers support streaming
    true
}

/// Check if provider supports vision
fn provider_supports_vision(provider_name: &str) -> bool {
    matches!(provider_name, "OpenAI" | "Anthropic")
}

/// Check if provider supports audio
fn provider_supports_audio(provider_name: &str) -> bool {
    matches!(provider_name, "OpenAI")
}

/// Check if provider supports tools
fn provider_supports_tools(provider_name: &str) -> bool {
    matches!(provider_name, "OpenAI" | "Anthropic")
}

/// Check if provider supports JSON mode
fn provider_supports_json_mode(provider_name: &str) -> bool {
    matches!(provider_name, "OpenAI" | "Anthropic")
}

/// Check if provider supports thinking process
fn provider_supports_thinking(provider_name: &str) -> bool {
    matches!(provider_name, "Anthropic")
}

/// Try streaming with fallback to regular chat
async fn try_streaming_with_fallback(client: &dyn ChatCapability) -> Result<String, LlmError> {
    let messages = vec![user!("Count from 1 to 5")];
    
    // Try streaming first
    match client.chat_stream(messages.clone(), None).await {
        Ok(mut stream) => {
            use futures_util::StreamExt;
            let mut result = String::new();
            
            while let Some(event) = stream.next().await {
                match event? {
                    ChatStreamEvent::ContentDelta { delta, .. } => {
                        result.push_str(&delta);
                    }
                    ChatStreamEvent::Done { .. } => break,
                    _ => {}
                }
            }
            
            Ok(format!("Streaming: {}", result))
        }
        Err(_) => {
            // Fallback to regular chat
            let response = client.chat(messages).await?;
            Ok(format!("Fallback: {}", response.content_text().unwrap_or_default()))
        }
    }
}

/// Try vision with fallback to text-only
async fn try_vision_with_fallback(client: &dyn ChatCapability) -> Result<String, LlmError> {
    // Try vision request (simplified)
    let messages = vec![user!("Describe what you see in this image: [image would be here]")];
    
    match client.chat(messages.clone()).await {
        Ok(response) => {
            Ok(format!("Vision: {}", response.content_text().unwrap_or_default()))
        }
        Err(_) => {
            // Fallback to text-only
            let fallback_messages = vec![user!("Explain how image analysis works")];
            let response = client.chat(fallback_messages).await?;
            Ok(format!("Text fallback: {}", response.content_text().unwrap_or_default()))
        }
    }
}

/// Provider metadata structure
#[derive(Debug)]
struct ProviderMetadata {
    provider_type: String,
    model: String,
    max_tokens: Option<u32>,
    context_window: u32,
    supports_streaming: bool,
    supports_vision: bool,
    cost_per_1k_tokens: f64,
}

/// Get provider metadata
async fn get_provider_metadata(client: &dyn ChatCapability, provider_name: &str) -> ProviderMetadata {
    match provider_name {
        "OpenAI" => ProviderMetadata {
            provider_type: "Cloud API".to_string(),
            model: "gpt-4o-mini".to_string(),
            max_tokens: Some(4096),
            context_window: 128000,
            supports_streaming: true,
            supports_vision: true,
            cost_per_1k_tokens: 0.15,
        },
        "Anthropic" => ProviderMetadata {
            provider_type: "Cloud API".to_string(),
            model: "claude-3-5-haiku".to_string(),
            max_tokens: Some(4096),
            context_window: 200000,
            supports_streaming: true,
            supports_vision: true,
            cost_per_1k_tokens: 0.25,
        },
        "Ollama" => ProviderMetadata {
            provider_type: "Local".to_string(),
            model: "llama3.2".to_string(),
            max_tokens: Some(2048),
            context_window: 8192,
            supports_streaming: true,
            supports_vision: false,
            cost_per_1k_tokens: 0.0,
        },
        _ => ProviderMetadata {
            provider_type: "Unknown".to_string(),
            model: "unknown".to_string(),
            max_tokens: None,
            context_window: 4096,
            supports_streaming: false,
            supports_vision: false,
            cost_per_1k_tokens: 0.0,
        },
    }
}

/// Adaptive chat that adjusts behavior based on capabilities
async fn adaptive_chat(client: &dyn ChatCapability, provider_name: &str, message: &str) -> Result<String, LlmError> {
    let metadata = get_provider_metadata(client, provider_name).await;
    
    // Adjust message based on capabilities
    let adjusted_message = if metadata.supports_vision {
        format!("{} (Note: This provider supports vision)", message)
    } else {
        message.to_string()
    };
    
    let messages = vec![user!(&adjusted_message)];
    
    // Use streaming if supported, otherwise regular chat
    if metadata.supports_streaming {
        // Try streaming
        match client.chat_stream(messages.clone(), None).await {
            Ok(mut stream) => {
                use futures_util::StreamExt;
                let mut result = String::new();
                
                while let Some(event) = stream.next().await {
                    match event? {
                        ChatStreamEvent::ContentDelta { delta, .. } => {
                            result.push_str(&delta);
                        }
                        ChatStreamEvent::Done { .. } => break,
                        _ => {}
                    }
                }
                
                Ok(result)
            }
            Err(_) => {
                // Fallback to regular chat
                let response = client.chat(messages).await?;
                Ok(response.content_text().unwrap_or_default().to_string())
            }
        }
    } else {
        // Use regular chat
        let response = client.chat(messages).await?;
        Ok(response.content_text().unwrap_or_default().to_string())
    }
}

/*
🎯 Key Capability Detection Concepts:

Detection Methods:
- Runtime testing: Try features and handle failures
- Provider metadata: Known capabilities per provider
- API introspection: Query provider capabilities
- Feature flags: Configuration-based feature control

Graceful Degradation:
- Fallback to simpler features when advanced ones fail
- Progressive enhancement based on capabilities
- User notification of limited functionality
- Alternative workflows for missing features

Best Practices:
1. Cache capability detection results
2. Provide clear fallback behaviors
3. Test capabilities at startup
4. Monitor feature usage and failures
5. Document capability requirements
6. Handle capability changes gracefully

Production Considerations:
- Performance impact of capability detection
- Caching strategies for capability results
- Monitoring capability availability
- User experience with degraded features
- Cost implications of different capabilities

Next Steps:
- parameter_mapping.rs: Parameter handling across providers
- ../03_advanced_features/: Advanced capability patterns
- ../04_providers/: Provider-specific capabilities
*/

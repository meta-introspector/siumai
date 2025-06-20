//! Advanced Ollama Provider Example
//!
//! This example demonstrates advanced features of the Ollama provider,
//! including custom parameters, model management, and performance monitoring.

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Ollama Provider Example");
    println!("===================================");

    // Create an advanced Ollama client using the builder
    let client = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2:latest")
        .keep_alive("10m")           // Keep model in memory for 10 minutes
        .raw(false)                  // Use templating
        .format("json")              // Request JSON format
        .option("numa", serde_json::Value::Bool(true))  // Enable NUMA support
        .option("num_ctx", serde_json::Value::Number(
            serde_json::Number::from(4096)
        ))                          // Set context window
        .option("num_gpu", serde_json::Value::Number(
            serde_json::Number::from(1)
        ))                          // Use 1 GPU layer
        .option("temperature", serde_json::Value::Number(
            serde_json::Number::from_f64(0.8).unwrap()
        ))
        .option("top_p", serde_json::Value::Number(
            serde_json::Number::from_f64(0.9).unwrap()
        ))
        .build()
        .await?;
    
    println!("✅ Advanced Ollama client created");
    println!("🔧 Provider: {}", LlmClient::provider_name(&client));

    // Test health check
    println!("\n🏥 Health Check");
    println!("---------------");
    
    match client.health_check().await {
        Ok(is_healthy) => {
            if is_healthy {
                println!("✅ Ollama server is healthy");
                
                // Get version info
                match client.version().await {
                    Ok(version) => println!("📦 Ollama version: {}", version),
                    Err(e) => println!("⚠️  Could not get version: {}", e),
                }
            } else {
                println!("❌ Ollama server is not responding");
                return Ok(());
            }
        }
        Err(e) => {
            println!("❌ Health check failed: {}", e);
            return Ok(());
        }
    }

    // Test text generation with custom parameters
    println!("\n📝 Text Generation with Custom Parameters");
    println!("------------------------------------------");
    
    let prompt = "Write a haiku about artificial intelligence:";
    
    match client.generate(prompt.to_string()).await {
        Ok(response) => {
            println!("🤖 Generated text:");
            println!("{}", response);
        }
        Err(e) => {
            println!("❌ Generation failed: {}", e);
        }
    }

    // Test streaming generation
    println!("\n🌊 Streaming Text Generation");
    println!("----------------------------");
    
    let prompt = "Explain quantum computing in simple terms:";
    
    match client.generate_stream(prompt.to_string()).await {
        Ok(mut stream) => {
            println!("🤖 Generated text (streaming):");
            use futures_util::StreamExt;
            
            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                        print!("{}", delta);
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    }
                    Ok(ChatStreamEvent::Done { .. }) => {
                        println!("\n✅ Generation completed");
                        break;
                    }
                    Err(e) => {
                        println!("\n❌ Stream error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        }
        Err(e) => {
            println!("❌ Streaming generation failed: {}", e);
        }
    }

    // Test chat with performance monitoring
    println!("\n⚡ Chat with Performance Monitoring");
    println!("-----------------------------------");
    
    let messages = vec![
        user!("What are the benefits of using Rust for systems programming?"),
    ];

    let start_time = std::time::Instant::now();
    
    match client.chat_with_tools(messages, None).await {
        Ok(response) => {
            let duration = start_time.elapsed();
            
            match &response.content {
                MessageContent::Text(text) => println!("🤖 Assistant: {}", text),
                MessageContent::MultiModal(parts) => {
                    println!("🤖 Assistant (multimodal):");
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => println!("  Text: {}", text),
                            ContentPart::Image { image_url, .. } => println!("  Image: {}", image_url),
                            ContentPart::Audio { .. } => println!("  Audio content"),
                        }
                    }
                }
            }
            println!("⏱️  Response time: {:?}", duration);
            
            if let Some(usage) = response.usage {
                println!("📊 Token usage:");
                println!("   - Prompt tokens: {}", usage.prompt_tokens);
                println!("   - Completion tokens: {}", usage.completion_tokens);
                println!("   - Total tokens: {}", usage.total_tokens);
            }
            
            // Display performance metadata
            if !response.metadata.is_empty() {
                println!("🔍 Performance metrics:");
                for (key, value) in &response.metadata {
                    println!("   - {}: {}", key, value);
                }
            }
        }
        Err(e) => {
            println!("❌ Chat failed: {}", e);
        }
    }

    // Test multimodal capabilities (if supported)
    println!("\n🖼️  Multimodal Test");
    println!("------------------");
    
    let multimodal_messages = vec![
        ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![
                ContentPart::Text { 
                    text: "Describe this image:".to_string() 
                },
                ContentPart::Image { 
                    image_url: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A".to_string(),
                    detail: None,
                },
            ]),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    match client.chat_with_tools(multimodal_messages, None).await {
        Ok(response) => {
            match &response.content {
                MessageContent::Text(text) => println!("🤖 Vision response: {}", text),
                MessageContent::MultiModal(parts) => {
                    println!("🤖 Vision response (multimodal):");
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => println!("  Text: {}", text),
                            ContentPart::Image { image_url, .. } => println!("  Image: {}", image_url),
                            ContentPart::Audio { .. } => println!("  Audio content"),
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Vision test failed: {}", e);
            println!("💡 This model might not support vision capabilities");
        }
    }

    println!("\n🎯 Advanced Example Completed!");
    println!("\n💡 Advanced Tips:");
    println!("  - Use keep_alive to control model memory usage");
    println!("  - Adjust num_ctx for longer conversations");
    println!("  - Use num_gpu to control GPU utilization");
    println!("  - Monitor performance metrics for optimization");

    Ok(())
}

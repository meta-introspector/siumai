//! Ollama Provider Example
//!
//! This example demonstrates how to use the Ollama provider with the siumai library.
//! 
//! Prerequisites:
//! - Ollama must be installed and running on your system
//! - At least one model must be available (e.g., llama3.2)
//!
//! To run this example:
//! ```bash
//! cargo run --example ollama_example
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦙 Ollama Provider Example");
    println!("========================");

    // Create an Ollama client
    let client = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")  // Default Ollama URL
        .model("llama3.2:latest")            // Use your available model
        .temperature(0.7)
        .max_tokens(1000)
        .build()
        .await?;

    println!("✅ Ollama client created successfully");
    println!("🔧 Provider: {}", LlmClient::provider_name(&client));

    // Test basic chat functionality
    println!("\n💬 Testing Chat Functionality");
    println!("------------------------------");
    
    let messages = vec![
        user!("Hello! Can you tell me a short joke about programming?"),
    ];

    match client.chat_with_tools(messages, None).await {
        Ok(response) => {
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
            if let Some(usage) = response.usage {
                println!("📊 Usage: {} prompt tokens, {} completion tokens", 
                    usage.prompt_tokens, usage.completion_tokens);
            }
        }
        Err(e) => {
            println!("❌ Chat failed: {}", e);
            println!("💡 Make sure Ollama is running and the model is available");
        }
    }

    // Test streaming chat
    println!("\n🌊 Testing Streaming Chat");
    println!("-------------------------");

    let messages = vec![
        user!("Please write a detailed introduction about the Rust programming language, including its features, advantages, and use cases. Write about 300-500 words and take your time to explain thoroughly."),
    ];

    match client.chat_stream(messages, None).await {
        Ok(mut stream) => {
            println!("🤖 Assistant (streaming): ");
            use futures_util::StreamExt;
            
            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                        print!("{}", delta);
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    }
                    Ok(ChatStreamEvent::Done { .. }) => {
                        println!("\n✅ Stream completed");
                        break;
                    }
                    Err(e) => {
                        println!("\n❌ Stream error: {}", e);
                        break;
                    }
                    _ => {} // Handle other event types if needed
                }
            }
        }
        Err(e) => {
            println!("❌ Streaming failed: {}", e);
        }
    }

    // Test embeddings (if supported by your model)
    println!("\n🔢 Testing Embeddings");
    println!("---------------------");
    
    let texts = vec![
        "Hello world".to_string(),
        "Rust programming language".to_string(),
    ];

    match client.embed(texts).await {
        Ok(response) => {
            println!("✅ Generated {} embeddings", response.embeddings.len());
            for (i, embedding) in response.embeddings.iter().enumerate() {
                println!("📊 Embedding {}: {} dimensions", i + 1, embedding.len());
            }
        }
        Err(e) => {
            println!("❌ Embeddings failed: {}", e);
            println!("💡 Make sure you have an embedding model available (e.g., nomic-embed-text)");
        }
    }

    // Test model listing
    println!("\n📋 Testing Model Listing");
    println!("------------------------");
    
    match client.list_models().await {
        Ok(models) => {
            println!("✅ Available models:");
            for model in models.iter().take(5) { // Show first 5 models
                println!("  🤖 {}", model.id);
                if let Some(description) = &model.description {
                    println!("     📝 {}", description);
                }
            }
            if models.len() > 5 {
                println!("  ... and {} more models", models.len() - 5);
            }
        }
        Err(e) => {
            println!("❌ Model listing failed: {}", e);
        }
    }

    println!("\n🎉 Example completed!");
    println!("\n💡 Tips:");
    println!("  - Install models with: ollama pull <model-name>");
    println!("  - List available models with: ollama list");
    println!("  - Check Ollama status with: ollama ps");

    Ok(())
}

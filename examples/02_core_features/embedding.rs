//! 🔗 Unified Embedding Interface - Text embeddings across providers
//!
//! This example demonstrates the complete embedding functionality through Siumai's unified interface:
//! - Direct embedding calls through Siumai unified client
//! - Capability-based embedding access and detection
//! - Provider comparison (OpenAI, Ollama, Anthropic)
//! - Error handling for unsupported providers
//! - Interface comparison: unified vs provider-specific
//!
//! Before running, set your API keys:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export OLLAMA_BASE_URL="http://localhost:11434"  # Optional, defaults to localhost
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example embedding
//! ```

use siumai::prelude::*;
use siumai::traits::EmbeddingCapability;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Unified Embedding Interface Demo");
    println!("===================================\n");

    // Example texts to embed
    let texts = vec![
        "Hello, world!".to_string(),
        "Rust is a systems programming language.".to_string(),
        "Machine learning and AI are transforming technology.".to_string(),
    ];

    // Demo 1: OpenAI Embeddings (if available)
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        println!("1️⃣ OpenAI Embeddings via Unified Interface");
        println!("-------------------------------------------");

        match demonstrate_openai_embeddings(&api_key, &texts).await {
            Ok(_) => println!("✅ OpenAI embeddings completed successfully\n"),
            Err(e) => println!("❌ OpenAI embeddings failed: {e}\n"),
        }
    } else {
        println!("⏭️ Skipping OpenAI (OPENAI_API_KEY not set)\n");
    }

    // Demo 2: Ollama Embeddings (if available)
    println!("2️⃣ Ollama Embeddings via Unified Interface");
    println!("------------------------------------------");

    match demonstrate_ollama_embeddings(&texts).await {
        Ok(_) => println!("✅ Ollama embeddings completed successfully\n"),
        Err(e) => println!("❌ Ollama embeddings failed: {e}\n"),
    }

    // Demo 3: Capability Detection and Error Handling
    println!("3️⃣ Capability Detection and Error Handling");
    println!("-------------------------------------------");

    demonstrate_capability_detection().await?;

    // Demo 4: Provider-Specific vs Unified Interface
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        println!("4️⃣ Provider-Specific vs Unified Interface");
        println!("------------------------------------------");

        demonstrate_interface_comparison(&api_key, &texts).await?;
    }

    println!("🎉 All embedding demos completed!");
    Ok(())
}

/// Demonstrate OpenAI embeddings through unified interface
async fn demonstrate_openai_embeddings(
    api_key: &str,
    texts: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    // Create unified client
    let client = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("text-embedding-3-small") // Specify embedding model
        .build()
        .await?;

    // Method 1: Direct embedding call
    println!("   📊 Direct embedding call:");
    let response = client.embed(texts.to_vec()).await?;
    println!("      Got {} embeddings", response.embeddings.len());
    println!(
        "      Embedding dimension: {}",
        response.embeddings[0].len()
    );
    println!("      Model used: {}", response.model);

    if let Some(usage) = &response.usage {
        println!("      Tokens used: {}", usage.total_tokens);
    }

    // Method 2: Through capability proxy
    println!("   🔧 Through capability proxy:");
    let embedding_proxy = client.embedding_capability();

    if embedding_proxy.is_reported_as_supported() {
        println!("      ✅ Embedding capability is supported");
        println!(
            "      Max tokens per request: {}",
            embedding_proxy.max_tokens_per_embedding()
        );
        println!(
            "      Supported models: {:?}",
            embedding_proxy.supported_embedding_models()
        );

        let response2 = embedding_proxy.embed(vec![texts[0].clone()]).await?;
        println!(
            "      Single embedding dimension: {}",
            response2.embeddings[0].len()
        );
    } else {
        println!("      ❌ Embedding capability not reported as supported");
    }

    Ok(())
}

/// Demonstrate Ollama embeddings through unified interface
async fn demonstrate_ollama_embeddings(texts: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());

    // Create unified client
    let client = Siumai::builder()
        .ollama()
        .base_url(&base_url)
        .model("nomic-embed-text") // Ollama embedding model
        .build()
        .await?;

    // Direct embedding call
    println!("   📊 Ollama embedding call:");
    let response = client.embed(texts.to_vec()).await?;
    println!("      Got {} embeddings", response.embeddings.len());
    println!(
        "      Embedding dimension: {}",
        response.embeddings[0].len()
    );
    println!("      Model used: {}", response.model);

    // Check capability info
    let embedding_proxy = client.embedding_capability();
    println!(
        "      Capability supported: {}",
        embedding_proxy.is_reported_as_supported()
    );
    println!(
        "      Default dimension: {}",
        embedding_proxy.embedding_dimension()
    );

    Ok(())
}

/// Demonstrate capability detection and error handling
async fn demonstrate_capability_detection() -> Result<(), Box<dyn std::error::Error>> {
    // Try to create an Anthropic client (which doesn't support embeddings)
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        println!("   🧪 Testing Anthropic (unsupported provider):");

        let client = Siumai::builder()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-haiku-20241022")
            .build()
            .await?;

        // Check capability status
        let embedding_proxy = client.embedding_capability();
        println!("      Provider: {}", embedding_proxy.provider_name());
        println!(
            "      Reported as supported: {}",
            embedding_proxy.is_reported_as_supported()
        );
        println!("      Status: {}", embedding_proxy.support_status_message());

        // Try to embed (should fail gracefully)
        match client.embed(vec!["test".to_string()]).await {
            Ok(_) => println!("      ✅ Unexpectedly succeeded!"),
            Err(e) => println!("      ❌ Expected error: {e}"),
        }
    } else {
        println!("   ⏭️ Skipping Anthropic test (ANTHROPIC_API_KEY not set)");
    }

    Ok(())
}

/// Compare provider-specific vs unified interface
async fn demonstrate_interface_comparison(
    api_key: &str,
    texts: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   🔄 Unified Interface:");

    // Unified interface - simple and consistent
    let unified_client = Siumai::builder().openai().api_key(api_key).build().await?;

    let unified_response = unified_client.embed(texts.to_vec()).await?;
    println!(
        "      Unified: {} embeddings, {} dimensions",
        unified_response.embeddings.len(),
        unified_response.embeddings[0].len()
    );

    println!("   ⚙️ Provider-Specific Interface:");

    // Provider-specific interface - more control
    let specific_client = Provider::openai().api_key(api_key).build().await?;

    let specific_response = specific_client.embed(texts.to_vec()).await?;
    println!(
        "      Provider-specific: {} embeddings, {} dimensions",
        specific_response.embeddings.len(),
        specific_response.embeddings[0].len()
    );

    println!("   📊 Both interfaces produce identical results!");

    Ok(())
}

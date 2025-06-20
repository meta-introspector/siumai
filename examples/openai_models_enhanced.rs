//! OpenAI Enhanced Models API Example
//!
//! This example demonstrates the enhanced OpenAI Models API features:
//! - Comprehensive model listing with detailed capabilities
//! - Model filtering by capability type
//! - Enhanced model specifications and pricing information
//! - Model recommendation system
//! - Capability detection for all model types

use siumai::{
    providers::openai::{OpenAiConfig, OpenAiModels},
    traits::ModelListingCapability,
    types::HttpConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the OpenAI models client
    let config = OpenAiConfig::new("your-api-key-here");
    let http_config = HttpConfig::default();
    let models_client = OpenAiModels::new(
        config.api_key.clone(),
        config.base_url.clone(),
        reqwest::Client::new(),
        config.organization.clone(),
        config.project.clone(),
        http_config,
    );

    println!("🤖 OpenAI Enhanced Models API Demo");
    println!("===================================\n");

    // Example 1: Model capability filtering
    model_capability_filtering(&models_client).await?;

    // Example 2: Model recommendations
    model_recommendations(&models_client).await?;

    // Example 3: Model specifications and pricing
    model_specifications(&models_client).await?;

    // Example 4: Capability analysis
    capability_analysis(&models_client).await?;

    Ok(())
}

/// Example 1: Demonstrate model filtering by capability
async fn model_capability_filtering(
    client: &OpenAiModels,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Example 1: Model Capability Filtering");
    println!("----------------------------------------");

    // Demonstrate different model categories
    let capability_examples = vec![
        ("Chat Models", "chat"),
        ("Image Generation Models", "image_generation"),
        ("Audio Models", "audio"),
        ("Embedding Models", "embeddings"),
        ("Moderation Models", "moderation"),
        ("Vision-Capable Models", "vision"),
        ("Reasoning Models", "reasoning"),
        ("Tool-Capable Models", "tools"),
    ];

    for (description, capability) in capability_examples {
        println!("{}:", description);
        println!("  Capability: {}", capability);
        println!("  Status: Ready to filter models (requires valid API key)");

        // Note: Actual filtering would be:
        // let models = client.get_models_by_capability(capability).await?;
        // println!("  Found {} models with {} capability", models.len(), capability);
        // for model in models.iter().take(3) {
        //     println!("    - {} ({})", model.id, model.owned_by.as_deref().unwrap_or("OpenAI"));
        // }
        println!();
    }

    Ok(())
}

/// Example 2: Demonstrate model recommendations
async fn model_recommendations(client: &OpenAiModels) -> Result<(), Box<dyn std::error::Error>> {
    println!("💡 Example 2: Model Recommendations");
    println!("-----------------------------------");

    let use_cases = vec![
        (
            "General Chat",
            "chat",
            "Best for conversational AI and general tasks",
        ),
        ("Fast Chat", "chat_fast", "Optimized for quick responses"),
        (
            "Complex Reasoning",
            "reasoning",
            "Advanced problem-solving and analysis",
        ),
        ("Fast Reasoning", "reasoning_fast", "Quick reasoning tasks"),
        (
            "Image Analysis",
            "vision",
            "Understanding and analyzing images",
        ),
        ("Text-to-Speech", "tts", "Converting text to natural speech"),
        (
            "High-Quality TTS",
            "tts_hd",
            "Premium quality text-to-speech",
        ),
        (
            "Custom Voice TTS",
            "tts_custom",
            "TTS with voice customization",
        ),
        ("Speech-to-Text", "stt", "Converting speech to text"),
        (
            "Image Generation",
            "image_generation",
            "Creating images from text",
        ),
        (
            "Fast Image Generation",
            "image_generation_fast",
            "Quick image creation",
        ),
        (
            "HD Image Generation",
            "image_generation_hd",
            "High-resolution image creation",
        ),
        (
            "Text Embeddings",
            "embeddings",
            "Converting text to vector embeddings",
        ),
        (
            "Fast Embeddings",
            "embeddings_fast",
            "Quick embedding generation",
        ),
        (
            "Content Moderation",
            "moderation",
            "Detecting harmful content",
        ),
    ];

    println!("Model Recommendations by Use Case:");
    println!(
        "┌─────────────────────────┬─────────────────────┬─────────────────────────────────────┐"
    );
    println!(
        "│ Use Case                │ Recommended Model   │ Description                         │"
    );
    println!(
        "├─────────────────────────┼─────────────────────┼─────────────────────────────────────┤"
    );

    for (use_case, key, description) in use_cases {
        let recommended = client.get_recommended_model(key);
        println!(
            "│ {:<23} │ {:<19} │ {:<35} │",
            use_case, recommended, description
        );
    }
    println!(
        "└─────────────────────────┴─────────────────────┴─────────────────────────────────────┘"
    );
    println!();

    Ok(())
}

/// Example 3: Demonstrate model specifications and pricing
async fn model_specifications(client: &OpenAiModels) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 3: Model Specifications and Pricing");
    println!("----------------------------------------------");

    // Example model specifications (these would come from actual API calls)
    let model_specs = vec![
        (
            "gpt-4o",
            "128K",
            "16K",
            "$0.0025",
            "$0.01",
            "Chat, Vision, Audio",
        ),
        (
            "gpt-4o-mini",
            "128K",
            "16K",
            "$0.00015",
            "$0.0006",
            "Chat, Vision, Audio",
        ),
        (
            "gpt-4o-mini-tts",
            "N/A",
            "N/A",
            "$0.015/char",
            "N/A",
            "Text-to-Speech",
        ),
        ("o1-preview", "128K", "32K", "$0.015", "$0.06", "Reasoning"),
        (
            "o1-mini",
            "128K",
            "65K",
            "$0.003",
            "$0.012",
            "Fast Reasoning",
        ),
        (
            "dall-e-3",
            "N/A",
            "N/A",
            "$0.04/img",
            "N/A",
            "Image Generation",
        ),
        (
            "gpt-image-1",
            "N/A",
            "N/A",
            "$0.03/img",
            "N/A",
            "HD Image Generation",
        ),
        (
            "whisper-1",
            "N/A",
            "N/A",
            "$0.006/min",
            "N/A",
            "Speech-to-Text",
        ),
        (
            "text-embedding-3-large",
            "8K",
            "N/A",
            "$0.00013",
            "N/A",
            "Embeddings",
        ),
        (
            "text-moderation-latest",
            "32K",
            "N/A",
            "Free",
            "N/A",
            "Content Moderation",
        ),
    ];

    println!("Model Specifications:");
    println!(
        "┌─────────────────────────┬─────────┬─────────┬─────────────┬─────────────┬─────────────────────┐"
    );
    println!(
        "│ Model                   │ Context │ Output  │ Input Cost  │ Output Cost │ Capabilities        │"
    );
    println!(
        "├─────────────────────────┼─────────┼─────────┼─────────────┼─────────────┼─────────────────────┤"
    );

    for (model, context, output, input_cost, output_cost, capabilities) in model_specs {
        println!(
            "│ {:<23} │ {:<7} │ {:<7} │ {:<11} │ {:<11} │ {:<19} │",
            model, context, output, input_cost, output_cost, capabilities
        );
    }
    println!(
        "└─────────────────────────┴─────────┴─────────┴─────────────┴─────────────┴─────────────────────┘"
    );

    println!("\nPricing Notes:");
    println!("  - Chat models: Per 1K tokens");
    println!("  - TTS models: Per character");
    println!("  - Image models: Per image");
    println!("  - Audio models: Per minute");
    println!("  - Moderation: Free tier available");
    println!();

    Ok(())
}

/// Example 4: Demonstrate capability analysis
async fn capability_analysis(client: &OpenAiModels) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Example 4: Capability Analysis");
    println!("---------------------------------");

    println!("Enhanced Capability Detection:");

    // Chat capabilities
    println!("\n📝 Chat & Text Models:");
    println!("  - Basic chat: gpt-3.5-turbo, gpt-4, gpt-4o");
    println!("  - Tool support: Most modern chat models (except o1 series)");
    println!("  - Streaming: All chat models (except o1 series)");
    println!("  - Vision: gpt-4, gpt-4o series");
    println!("  - Audio: gpt-4o, gpt-4o-mini");
    println!("  - Reasoning: o1-preview, o1-mini");

    // Audio capabilities
    println!("\n🎵 Audio Models:");
    println!("  - Text-to-Speech: tts-1, tts-1-hd, gpt-4o-mini-tts");
    println!("  - Speech-to-Text: whisper-1");
    println!("  - Transcription: whisper-1");
    println!("  - Translation: whisper-1");
    println!("  - Custom voices: gpt-4o-mini-tts (with instructions)");

    // Image capabilities
    println!("\n🎨 Image Models:");
    println!("  - Generation: dall-e-2, dall-e-3, gpt-image-1");
    println!("  - Editing: dall-e-2 (basic), dall-e-3 (advanced)");
    println!("  - Variations: dall-e-2, dall-e-3");
    println!("  - High resolution: gpt-image-1 (up to 2048x2048)");

    // Embedding capabilities
    println!("\n🔢 Embedding Models:");
    println!("  - Text embeddings: text-embedding-3-small, text-embedding-3-large");
    println!("  - Legacy: text-embedding-ada-002");
    println!("  - Dimensions: Configurable for v3 models");

    // Moderation capabilities
    println!("\n🛡️ Moderation Models:");
    println!("  - Text moderation: text-moderation-stable, text-moderation-latest");
    println!("  - Categories: hate, harassment, self-harm, sexual, violence");
    println!("  - Confidence scores: 0.0 to 1.0 for each category");

    println!("\nCapability Matrix:");
    println!("  ✅ = Supported");
    println!("  ❌ = Not supported");
    println!("  🔄 = Limited support");

    let capability_matrix = vec![
        (
            "Model Type",
            "Chat",
            "Vision",
            "Audio",
            "Tools",
            "Stream",
            "Reason",
        ),
        ("gpt-4o", "✅", "✅", "✅", "✅", "✅", "❌"),
        ("gpt-4o-mini", "✅", "✅", "✅", "✅", "✅", "❌"),
        ("o1-preview", "✅", "❌", "❌", "❌", "❌", "✅"),
        ("o1-mini", "✅", "❌", "❌", "❌", "❌", "✅"),
        ("dall-e-3", "❌", "❌", "❌", "❌", "❌", "❌"),
        ("whisper-1", "❌", "❌", "✅", "❌", "❌", "❌"),
        ("tts-1", "❌", "❌", "✅", "❌", "❌", "❌"),
    ];

    println!("\n┌─────────────────┬──────┬────────┬───────┬───────┬────────┬────────┐");
    for (i, row) in capability_matrix.iter().enumerate() {
        if i == 0 {
            println!(
                "│ {:<15} │ {:<4} │ {:<6} │ {:<5} │ {:<5} │ {:<6} │ {:<6} │",
                row.0, row.1, row.2, row.3, row.4, row.5, row.6
            );
            println!("├─────────────────┼──────┼────────┼───────┼───────┼────────┼────────┤");
        } else {
            println!(
                "│ {:<15} │ {:<4} │ {:<6} │ {:<5} │ {:<5} │ {:<6} │ {:<6} │",
                row.0, row.1, row.2, row.3, row.4, row.5, row.6
            );
        }
    }
    println!("└─────────────────┴──────┴────────┴───────┴───────┴────────┴────────┘");
    println!();

    Ok(())
}

/// Example 5: Model selection workflow
#[allow(dead_code)]
async fn model_selection_workflow(client: &OpenAiModels) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Example 5: Model Selection Workflow");
    println!("--------------------------------------");

    // Step 1: Define requirements
    println!("Step 1: Define Requirements");
    let requirements = vec![
        "Need: Conversational AI with vision capabilities",
        "Budget: Medium ($0.01 per 1K tokens acceptable)",
        "Speed: Fast response time preferred",
        "Features: Tool calling, streaming, image analysis",
    ];

    for req in &requirements {
        println!("  - {}", req);
    }

    // Step 2: Filter models by capabilities
    println!("\nStep 2: Filter by Capabilities");
    println!("  - Filtering for: chat + vision + tools + streaming");
    // let chat_models = client.get_chat_models().await?;
    // let vision_models = chat_models.into_iter()
    //     .filter(|m| m.capabilities.contains(&"vision".to_string()))
    //     .collect::<Vec<_>>();

    // Step 3: Compare specifications
    println!("\nStep 3: Compare Specifications");
    let candidates = vec![
        (
            "gpt-4o",
            "✅ All requirements",
            "$0.0025 input, $0.01 output",
        ),
        (
            "gpt-4o-mini",
            "✅ All requirements",
            "$0.00015 input, $0.0006 output",
        ),
        (
            "gpt-4-turbo",
            "✅ All requirements",
            "$0.01 input, $0.03 output",
        ),
    ];

    for (model, features, cost) in candidates {
        println!("  - {}: {} ({})", model, features, cost);
    }

    // Step 4: Make recommendation
    println!("\nStep 4: Recommendation");
    let recommended = client.get_recommended_model("vision");
    println!("  - Recommended: {}", recommended);
    println!("  - Reason: Best balance of capabilities, performance, and cost");
    println!("  - Alternative: gpt-4o-mini for cost-sensitive applications");

    // Step 5: Validate choice
    println!("\nStep 5: Validation");
    // let supports_vision = client.model_supports_capability(&recommended, "vision").await?;
    // let supports_tools = client.model_supports_capability(&recommended, "tools").await?;
    // println!("  - Vision support: {}", if supports_vision { "✅" } else { "❌" });
    // println!("  - Tools support: {}", if supports_tools { "✅" } else { "❌" });
    println!("  - Status: Ready for validation (requires valid API key)");
    println!();

    Ok(())
}

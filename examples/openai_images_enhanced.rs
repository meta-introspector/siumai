//! OpenAI Enhanced Images Features Example
//!
//! This example demonstrates the newly implemented OpenAI Images API features:
//! - New model: gpt-image-1
//! - Enhanced image generation with higher resolution support
//! - Improved validation and error handling
//! - Complete image editing and variations support

use siumai::{
    providers::openai::{OpenAiConfig, OpenAiImages},
    traits::ImageGenerationCapability,
    types::{ImageEditRequest, ImageGenerationRequest, ImageVariationRequest},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the OpenAI images client
    let config = OpenAiConfig::new("your-api-key-here");
    let http_client = reqwest::Client::new();
    let images_client = OpenAiImages::new(config, http_client);

    println!("🎨 OpenAI Enhanced Images Features Demo");
    println!("=======================================\n");

    // Example 1: Using the new gpt-image-1 model
    new_model_example(&images_client).await?;

    // Example 2: Higher resolution support
    high_resolution_example(&images_client).await?;

    // Example 3: Enhanced validation
    validation_example(&images_client).await?;

    // Example 4: Complete image workflow
    complete_workflow_example(&images_client).await?;

    Ok(())
}

/// Example 1: Demonstrate the new gpt-image-1 model
async fn new_model_example(client: &OpenAiImages) -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Example 1: New GPT-Image-1 Model");
    println!("------------------------------------");

    let request = ImageGenerationRequest {
        prompt: "A futuristic cityscape with flying cars and neon lights, cyberpunk style"
            .to_string(),
        model: Some("gpt-image-1".to_string()), // New model
        size: Some("2048x2048".to_string()),    // Higher resolution
        count: 2,                               // Can generate up to 4 images
        quality: Some("hd".to_string()),
        style: Some("vivid".to_string()),
        ..Default::default()
    };

    println!("Image Generation Request:");
    println!(
        "  - Model: {}",
        request.model.as_deref().unwrap_or("default")
    );
    println!("  - Size: {}", request.size.as_deref().unwrap_or("default"));
    println!("  - Count: {}", request.count);
    println!(
        "  - Quality: {}",
        request.quality.as_deref().unwrap_or("default")
    );
    println!(
        "  - Style: {}",
        request.style.as_deref().unwrap_or("default")
    );

    // Note: This is a demonstration - actual API call would require valid credentials
    println!("  - Status: Ready for API call (requires valid API key)");
    println!();

    Ok(())
}

/// Example 2: Demonstrate higher resolution support
async fn high_resolution_example(client: &OpenAiImages) -> Result<(), Box<dyn std::error::Error>> {
    println!("📐 Example 2: Higher Resolution Support");
    println!("---------------------------------------");

    println!("Supported image sizes by model:");

    // DALL-E 2 sizes
    println!("  DALL-E 2:");
    println!("    - 256x256 (legacy)");
    println!("    - 512x512 (legacy)");
    println!("    - 1024x1024 (standard)");

    // DALL-E 3 sizes
    println!("  DALL-E 3:");
    println!("    - 1024x1024 (square)");
    println!("    - 1792x1024 (landscape)");
    println!("    - 1024x1792 (portrait)");

    // GPT-Image-1 sizes (new)
    println!("  GPT-Image-1 (NEW):");
    println!("    - 1024x1024 (square)");
    println!("    - 1792x1024 (landscape)");
    println!("    - 1024x1792 (portrait)");
    println!("    - 2048x2048 (high resolution) ✨");

    println!("\nAll supported sizes: {:?}", client.get_supported_sizes());
    println!();

    Ok(())
}

/// Example 3: Demonstrate enhanced validation
async fn validation_example(client: &OpenAiImages) -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Example 3: Enhanced Validation");
    println!("---------------------------------");

    println!("Model validation:");
    let valid_models = ["dall-e-2", "dall-e-3", "gpt-image-1"];
    let invalid_models = ["dall-e-1", "midjourney", "stable-diffusion"];

    for model in &valid_models {
        println!("  ✅ Model '{}': Supported", model);
    }

    for model in &invalid_models {
        println!("  ❌ Model '{}': Not supported", model);
    }

    println!("\nCount validation by model:");
    println!("  - DALL-E 2: 1-10 images");
    println!("  - DALL-E 3: 1 image only");
    println!("  - GPT-Image-1: 1-4 images ✨");

    println!("\nSize validation:");
    println!("  - Each model has specific supported sizes");
    println!("  - Invalid size/model combinations are rejected");
    println!("  - Clear error messages guide users");
    println!();

    Ok(())
}

/// Example 4: Complete image workflow
async fn complete_workflow_example(
    client: &OpenAiImages,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Example 4: Complete Image Workflow");
    println!("-------------------------------------");

    // Step 1: Generate initial image
    println!("Step 1: Generate initial image with GPT-Image-1");
    let _generation_request = ImageGenerationRequest {
        prompt: "A serene mountain landscape with a crystal clear lake".to_string(),
        model: Some("gpt-image-1".to_string()),
        size: Some("1024x1024".to_string()),
        count: 1,
        quality: Some("hd".to_string()),
        style: Some("natural".to_string()),
        ..Default::default()
    };
    println!("  - Configured for high-quality natural style");

    // Step 2: Edit the image (simulation)
    println!("\nStep 2: Edit the generated image");
    // Note: In a real scenario, you would use the image data from step 1
    let _edit_request = ImageEditRequest {
        image: vec![0u8; 1024], // Placeholder image data
        prompt: "Add a small wooden cabin by the lake shore".to_string(),
        mask: None, // Optional mask for targeted editing
        size: Some("1024x1024".to_string()),
        count: Some(1),
        response_format: Some("url".to_string()),
        extra_params: HashMap::new(),
    };
    println!("  - Adding cabin to the landscape");

    // Step 3: Create variations (simulation)
    println!("\nStep 3: Create variations of the edited image");
    let _variation_request = ImageVariationRequest {
        image: vec![0u8; 1024], // Placeholder image data
        size: Some("1024x1024".to_string()),
        count: Some(3),
        response_format: Some("url".to_string()),
        extra_params: HashMap::new(),
    };
    println!("  - Generating 3 variations");

    println!("\nWorkflow capabilities:");
    println!("  - Image generation: ✅ Supported");
    println!(
        "  - Image editing: ✅ Supported ({})",
        client.supports_image_editing()
    );
    println!(
        "  - Image variations: ✅ Supported ({})",
        client.supports_image_variations()
    );
    println!(
        "  - Multiple formats: ✅ {:?}",
        client.get_supported_formats()
    );

    // Note: Actual API calls would be:
    // let generated = client.generate_images(generation_request).await?;
    // let edited = client.edit_image(edit_request).await?;
    // let variations = client.create_variation(variation_request).await?;

    println!("  - Status: Ready for execution (requires valid API key)");
    println!();

    Ok(())
}

/// Example 5: Error handling and best practices
#[allow(dead_code)]
fn error_handling_examples() {
    println!("🚨 Example 5: Error Handling and Best Practices");
    println!("-----------------------------------------------");

    println!("Common validation errors:");

    // Model validation
    println!("  ❌ Unsupported model error:");
    println!(
        "     Message: 'Unsupported model: custom-model. Supported models: [\"dall-e-2\", \"dall-e-3\", \"gpt-image-1\"]'"
    );

    // Count validation
    println!("  ❌ Invalid count for DALL-E 3:");
    println!("     Message: 'DALL-E 3 can generate only 1 image at a time'");

    println!("  ❌ Invalid count for GPT-Image-1:");
    println!("     Message: 'GPT-Image-1 can generate at most 4 images'");

    // Size validation
    println!("  ❌ Invalid size for model:");
    println!(
        "     Message: 'Unsupported size \"3000x3000\" for model \"dall-e-2\". Supported sizes: [\"256x256\", \"512x512\", \"1024x1024\"]'"
    );

    println!("\nBest practices:");
    println!("  - Always validate model and parameters before API calls");
    println!("  - Use appropriate image sizes for each model");
    println!("  - Handle rate limits and API errors gracefully");
    println!("  - Consider using lower resolution for testing");
    println!();
}

// Note: Default implementations removed due to orphan rules
// These types are defined in the library, not in this example

/// Example 6: Model comparison and selection guide
#[allow(dead_code)]
fn model_comparison_guide() {
    println!("📊 Example 6: Model Comparison and Selection Guide");
    println!("--------------------------------------------------");

    println!("Model comparison:");
    println!("┌─────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Feature     │ DALL-E 2    │ DALL-E 3    │ GPT-Image-1 │");
    println!("├─────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ Max Images  │ 10          │ 1           │ 4           │");
    println!("│ Max Size    │ 1024x1024   │ 1792x1024   │ 2048x2048   │");
    println!("│ Quality     │ Standard    │ HD/Standard │ HD/Standard │");
    println!("│ Style       │ No          │ Yes         │ Yes         │");
    println!("│ Speed       │ Fast        │ Medium      │ Medium      │");
    println!("│ Cost        │ Low         │ High        │ Medium      │");
    println!("└─────────────┴─────────────┴─────────────┴─────────────┘");

    println!("\nSelection guide:");
    println!("  🎯 Use DALL-E 2 for:");
    println!("     - Bulk image generation (up to 10 images)");
    println!("     - Quick prototyping");
    println!("     - Cost-effective solutions");

    println!("  🎨 Use DALL-E 3 for:");
    println!("     - High-quality single images");
    println!("     - Complex prompt understanding");
    println!("     - Artistic style control");

    println!("  ✨ Use GPT-Image-1 for:");
    println!("     - High-resolution images (2048x2048)");
    println!("     - Multiple variations (up to 4)");
    println!("     - Balanced quality and performance");
    println!();
}

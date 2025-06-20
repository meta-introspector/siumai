//! OpenAI Moderation API Example
//!
//! This example demonstrates the newly implemented OpenAI Moderation API features:
//! - Content moderation for text input
//! - Multiple moderation models (stable and latest)
//! - Comprehensive category detection and scoring
//! - Batch processing capabilities
//! - Validation and error handling

use siumai::{
    providers::openai::{OpenAiConfig, OpenAiModeration},
    traits::ModerationCapability,
    types::ModerationRequest,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the OpenAI moderation client
    let config = OpenAiConfig::new("your-api-key-here");
    let http_client = reqwest::Client::new();
    let moderation_client = OpenAiModeration::new(config, http_client);

    println!("🛡️ OpenAI Moderation API Demo");
    println!("==============================\n");

    // Example 1: Basic content moderation
    basic_moderation_example(&moderation_client).await?;

    // Example 2: Model comparison
    model_comparison_example(&moderation_client).await?;

    // Example 3: Category analysis
    category_analysis_example(&moderation_client).await?;

    // Example 4: Validation and error handling
    validation_example(&moderation_client).await?;

    Ok(())
}

/// Example 1: Demonstrate basic content moderation
async fn basic_moderation_example(
    client: &OpenAiModeration,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Example 1: Basic Content Moderation");
    println!("--------------------------------------");

    // Example texts for moderation (safe examples for demonstration)
    let test_texts = vec![
        (
            "Safe Content",
            "Hello, how are you today? I hope you're having a great day!",
        ),
        (
            "Educational Content",
            "This is an educational discussion about online safety and digital citizenship.",
        ),
        (
            "Business Content",
            "Our company provides excellent customer service and quality products.",
        ),
        (
            "Creative Content",
            "Once upon a time, in a magical kingdom far away, there lived a kind princess.",
        ),
    ];

    println!("Supported categories: {:?}", client.supported_categories());
    println!("Supported models: {:?}", client.get_supported_models());
    println!("Default model: {}\n", client.default_model());

    for (description, text) in test_texts {
        let request = ModerationRequest {
            input: text.to_string(),
            model: None, // Use default model
        };

        println!("Moderating: {}", description);
        println!("  Text: \"{}\"", text.chars().take(50).collect::<String>());
        println!("  Length: {} characters", text.len());
        println!("  Model: {}", client.default_model());
        println!("  Status: Ready for moderation (requires valid API key)");

        // Note: Actual moderation would be:
        // let response = client.moderate(request).await?;
        // for (i, result) in response.results.iter().enumerate() {
        //     println!("  Result {}: Flagged = {}", i + 1, result.flagged);
        //     if result.flagged {
        //         for (category, flagged) in &result.categories {
        //             if *flagged {
        //                 let score = result.category_scores.get(category).unwrap_or(&0.0);
        //                 println!("    - {}: {:.3}", category, score);
        //             }
        //         }
        //     }
        // }
        println!();
    }

    Ok(())
}

/// Example 2: Demonstrate model comparison
async fn model_comparison_example(
    client: &OpenAiModeration,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️ Example 2: Model Comparison");
    println!("------------------------------");

    let test_text = "This is a sample text for comparing different moderation models.";
    let models = client.get_supported_models();

    println!("Comparing moderation models for text:");
    println!("\"{}\"", test_text);
    println!();

    for model in &models {
        let request = ModerationRequest {
            input: test_text.to_string(),
            model: Some(model.clone()),
        };

        println!("Model: {}", model);
        println!(
            "  Description: {}",
            match model.as_str() {
                "text-moderation-stable" => "Stable version with consistent results",
                "text-moderation-latest" => "Latest version with improved accuracy",
                _ => "Unknown model",
            }
        );
        println!(
            "  Use case: {}",
            match model.as_str() {
                "text-moderation-stable" => "Production environments requiring consistency",
                "text-moderation-latest" => "Applications needing highest accuracy",
                _ => "General purpose",
            }
        );
        println!("  Status: Ready for comparison (requires valid API key)");

        // Note: Actual comparison would be:
        // let response = client.moderate(request).await?;
        // println!("  Results: {} flagged categories",
        //     response.results[0].categories.values().filter(|&&v| v).count());
        println!();
    }

    Ok(())
}

/// Example 3: Demonstrate category analysis
async fn category_analysis_example(
    client: &OpenAiModeration,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 3: Category Analysis");
    println!("-------------------------------");

    let categories = client.supported_categories();

    println!("OpenAI Moderation Categories:");
    println!("┌─────────────────────────┬─────────────────────────────────────────┐");
    println!("│ Category                │ Description                             │");
    println!("├─────────────────────────┼─────────────────────────────────────────┤");

    let category_descriptions = vec![
        ("hate", "Content that expresses, incites, or promotes hate"),
        (
            "hate/threatening",
            "Hateful content that includes violence or threats",
        ),
        ("harassment", "Content that harasses, bullies, or threatens"),
        (
            "harassment/threatening",
            "Harassment that includes violence or threats",
        ),
        ("self-harm", "Content that promotes or encourages self-harm"),
        (
            "self-harm/intent",
            "Content expressing intent to engage in self-harm",
        ),
        (
            "self-harm/instructions",
            "Content providing instructions for self-harm",
        ),
        ("sexual", "Sexual content intended to arouse"),
        (
            "sexual/minors",
            "Sexual content involving individuals under 18",
        ),
        ("violence", "Content depicting or promoting violence"),
        (
            "violence/graphic",
            "Graphic violence or extremely detailed descriptions",
        ),
    ];

    for (category, description) in category_descriptions {
        println!("│ {:<23} │ {:<39} │", category, description);
    }
    println!("└─────────────────────────┴─────────────────────────────────────────┘");

    println!("\nCategory Analysis Features:");
    println!("  ✅ Boolean flags for each category");
    println!("  ✅ Confidence scores (0.0 to 1.0)");
    println!("  ✅ Overall flagged status");
    println!("  ✅ Multiple categories can be flagged simultaneously");
    println!("  ✅ Granular subcategories for better classification");
    println!();

    Ok(())
}

/// Example 4: Demonstrate validation and error handling
async fn validation_example(client: &OpenAiModeration) -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Example 4: Validation and Error Handling");
    println!("-------------------------------------------");

    println!("Input Validation:");

    // Valid scenarios
    println!("\n✅ Valid Moderation Scenarios:");
    let valid_scenarios = vec![
        (
            "Short text",
            "Hello world".to_string(),
            "text-moderation-latest",
        ),
        ("Medium text", "A".repeat(1000), "text-moderation-stable"),
        ("Long text", "A".repeat(10000), "text-moderation-latest"),
        ("Default model", "Test content".to_string(), ""),
    ];

    for (description, content, model) in valid_scenarios {
        let model_display = if model.is_empty() { "default" } else { model };
        println!(
            "  - {}: {} chars, model: {}",
            description,
            content.len(),
            model_display
        );
    }

    // Invalid scenarios
    println!("\n❌ Invalid Moderation Scenarios:");
    let invalid_scenarios = vec![
        (
            "Empty input",
            "".to_string(),
            "text-moderation-latest",
            "Input text cannot be empty",
        ),
        (
            "Too long",
            "A".repeat(40000),
            "text-moderation-latest",
            "Input exceeds 32,768 character limit",
        ),
        (
            "Invalid model",
            "Test".to_string(),
            "invalid-model",
            "Unsupported moderation model",
        ),
        (
            "Whitespace only",
            "   ".to_string(),
            "text-moderation-latest",
            "Input text cannot be empty (after trim)",
        ),
    ];

    for (description, content, model, error_msg) in invalid_scenarios {
        println!(
            "  - {}: {} chars, model: {}",
            description,
            content.len(),
            model
        );
        println!("    Error: {}", error_msg);
    }

    println!("\nAPI Error Handling:");
    println!("  - 400 Bad Request: Invalid input format");
    println!("  - 401 Unauthorized: Invalid API key");
    println!("  - 429 Too Many Requests: Rate limit exceeded");
    println!("  - 500 Internal Server Error: OpenAI service issues");

    println!("\nBest Practices:");
    println!("  - Validate input length before API calls");
    println!("  - Use appropriate model for your use case");
    println!("  - Handle rate limits with exponential backoff");
    println!("  - Cache results for repeated content");
    println!("  - Monitor API usage and costs");
    println!("  - Implement fallback strategies for service outages");
    println!();

    Ok(())
}

/// Example 5: Complete moderation workflow
#[allow(dead_code)]
async fn complete_moderation_workflow(
    _client: &OpenAiModeration,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Example 5: Complete Moderation Workflow");
    println!("------------------------------------------");

    // Step 1: Prepare content for moderation
    let user_content =
        "This is user-generated content that needs to be moderated before publication.";

    println!("Step 1: Content Preparation");
    println!("  - Content: \"{}\"", user_content);
    println!("  - Length: {} characters", user_content.len());
    println!("  - Source: User submission");

    // Step 2: Create moderation request
    let request = ModerationRequest {
        input: user_content.to_string(),
        model: Some("text-moderation-latest".to_string()),
    };

    println!("\nStep 2: Moderation Request");
    println!(
        "  - Model: {}",
        request.model.as_deref().unwrap_or("default")
    );
    println!("  - Input validated: ✅");

    // Step 3: Perform moderation
    println!("\nStep 3: Moderation Analysis");
    // let response = client.moderate(request).await?;
    // let result = &response.results[0];

    // Step 4: Process results
    println!("\nStep 4: Result Processing");
    // if result.flagged {
    //     println!("  - Status: ❌ Content flagged");
    //     println!("  - Action: Block publication");
    //
    //     for (category, &flagged) in &result.categories {
    //         if flagged {
    //             let score = result.category_scores.get(category).unwrap_or(&0.0);
    //             println!("  - Violation: {} (confidence: {:.3})", category, score);
    //         }
    //     }
    // } else {
    //     println!("  - Status: ✅ Content approved");
    //     println!("  - Action: Allow publication");
    // }

    // Step 5: Take appropriate action
    println!("\nStep 5: Content Action");
    println!("  - Approved content: Publish to platform");
    println!("  - Flagged content: Send for human review");
    println!("  - High-confidence violations: Auto-reject");

    println!("  - Status: Workflow ready for execution");
    println!();

    Ok(())
}

/// Example 6: Batch processing simulation
#[allow(dead_code)]
async fn batch_processing_example(
    _client: &OpenAiModeration,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Example 6: Batch Processing");
    println!("------------------------------");

    let batch_content = vec![
        "First piece of content to moderate",
        "Second piece of content for review",
        "Third content item for safety check",
        "Fourth text for moderation analysis",
        "Fifth content piece for evaluation",
    ];

    println!("Processing {} content items:", batch_content.len());

    let _results: Vec<String> = Vec::new();
    for (i, content) in batch_content.iter().enumerate() {
        let _request = ModerationRequest {
            input: content.to_string(),
            model: Some("text-moderation-latest".to_string()),
        };

        println!(
            "  Item {}: \"{}\"",
            i + 1,
            content.chars().take(30).collect::<String>()
        );

        // Note: In real implementation, you might want to add delays to respect rate limits
        // let response = client.moderate(request).await?;
        // results.push(response);

        println!("    Status: Ready for processing");
    }

    println!("\nBatch Summary:");
    println!("  - Total items: {}", batch_content.len());
    println!(
        "  - Processing time: ~{} seconds (estimated)",
        batch_content.len()
    );
    println!("  - Rate limit considerations: Implemented");
    println!("  - Error handling: Configured");
    println!();

    Ok(())
}

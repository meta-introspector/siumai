//! Anthropic Models API Example
//!
//! Demonstrates how to use Anthropic's model listing functionality, conforming to the official API specification:
//! https://docs.anthropic.com/en/api/models-list

use siumai::error::LlmError;
use siumai::providers::anthropic::AnthropicClient;
use siumai::traits::ModelListingCapability;
use siumai::types::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the API key from environment variables
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("Please set the ANTHROPIC_API_KEY environment variable");

    println!("🤖 Anthropic Models API Feature Demonstration\n");

    // Create an Anthropic client
    let client = AnthropicClient::new(
        api_key,
        "https://api.anthropic.com".to_string(),
        reqwest::Client::new(),
        CommonParams::default(),
        Default::default(),
        Default::default(),
    );

    // 1. List all available models
    demo_list_all_models(&client).await?;

    // 2. Get information for a specific model
    demo_get_specific_model(&client).await?;

    // 3. Analyze model capabilities
    demo_analyze_model_capabilities(&client).await?;

    // 4. Compare model specifications
    demo_compare_model_specs(&client).await?;

    println!("✅ All demonstrations complete!");
    Ok(())
}

/// Demonstrates listing all available models
async fn demo_list_all_models(client: &AnthropicClient) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 1. List all available models");

    match client.list_models().await {
        Ok(models) => {
            println!("   Found {} models:", models.len());

            // Only display the first 10
            for (i, model) in models.iter().enumerate().take(10) {
                println!(
                    "   {}. {} ({})",
                    i + 1,
                    model.name.as_ref().unwrap_or(&model.id),
                    model.id
                );

                if let Some(desc) = &model.description {
                    println!("       Description: {}", desc);
                }

                if let Some(context) = model.context_window {
                    println!("       Context Window: {} tokens", context);
                }

                if !model.capabilities.is_empty() {
                    println!("       Capabilities: {}", model.capabilities.join(", "));
                }

                println!();
            }

            if models.len() > 10 {
                println!("   ... and {} more models", models.len() - 10);
            }
        }
        Err(e) => {
            println!("   ❌ Failed to get model list: {}", e);

            // Analyze the error type
            match &e {
                LlmError::AuthenticationError(_) => {
                    println!("   💡 Tip: Please check if your API key is correct");
                }
                LlmError::RateLimitError(_) => {
                    println!("   💡 Tip: Please try again later, you have reached the rate limit");
                }
                LlmError::ApiError { code, .. } => {
                    println!("   💡 API Error Code: {}", code);
                }
                _ => {
                    println!("   💡 Other error type");
                }
            }
        }
    }

    println!();
    Ok(())
}

/// Demonstrates getting information for a specific model
async fn demo_get_specific_model(
    client: &AnthropicClient,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 2. Get information for a specific model");

    let model_ids = vec![
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
    ];

    for model_id in model_ids {
        println!("   Querying model: {}", model_id);

        match client.get_model(model_id.to_string()).await {
            Ok(model) => {
                println!("   ✅ Model Information:");
                println!("       ID: {}", model.id);
                println!(
                    "       Name: {}",
                    model.name.unwrap_or("Unknown".to_string())
                );
                println!("       Owned by: {}", model.owned_by);

                if let Some(created) = model.created {
                    let datetime =
                        chrono::DateTime::from_timestamp(created as i64, 0).unwrap_or_default();
                    println!("       Created: {}", datetime.format("%Y-%m-%d %H:%M:%S"));
                }

                if let Some(context) = model.context_window {
                    println!("       Context Window: {} tokens", context);
                }

                if let Some(max_output) = model.max_output_tokens {
                    println!("       Max Output: {} tokens", max_output);
                }

                if let Some(input_cost) = model.input_cost_per_token {
                    println!("       Input Cost: ${:.8} per token", input_cost);
                }

                if let Some(output_cost) = model.output_cost_per_token {
                    println!("       Output Cost: ${:.8} per token", output_cost);
                }

                println!("       Capabilities: {}", model.capabilities.join(", "));
            }
            Err(e) => {
                println!("   ❌ Failed to get model information: {}", e);
            }
        }

        println!();
    }

    Ok(())
}

/// Demonstrates analyzing model capabilities
async fn demo_analyze_model_capabilities(
    client: &AnthropicClient,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 3. Analyze model capabilities");

    match client.list_models().await {
        Ok(models) => {
            let mut thinking_models = Vec::new();
            let mut vision_models = Vec::new();
            let mut tool_models = Vec::new();

            for model in &models {
                if model.capabilities.contains(&"thinking".to_string()) {
                    thinking_models.push(&model.id);
                }
                if model.capabilities.contains(&"vision".to_string()) {
                    vision_models.push(&model.id);
                }
                if model.capabilities.contains(&"tools".to_string()) {
                    tool_models.push(&model.id);
                }
            }

            println!(
                "   🤔 Models supporting 'thinking' ({}):",
                thinking_models.len()
            );
            for model_id in thinking_models.iter().take(5) {
                println!("       - {}", model_id);
            }
            if thinking_models.len() > 5 {
                println!("       ... and {} more", thinking_models.len() - 5);
            }

            println!();
            println!(
                "   👁️  Models supporting 'vision' ({}):",
                vision_models.len()
            );
            for model_id in vision_models.iter().take(5) {
                println!("       - {}", model_id);
            }
            if vision_models.len() > 5 {
                println!("       ... and {} more", vision_models.len() - 5);
            }

            println!();
            println!("   🔧 Models supporting 'tools' ({}):", tool_models.len());
            for model_id in tool_models.iter().take(5) {
                println!("       - {}", model_id);
            }
            if tool_models.len() > 5 {
                println!("       ... and {} more", tool_models.len() - 5);
            }
        }
        Err(e) => {
            println!("   ❌ Failed to analyze model capabilities: {}", e);
        }
    }

    println!();
    Ok(())
}

/// Demonstrates comparing model specifications
async fn demo_compare_model_specs(
    client: &AnthropicClient,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 4. Compare model specifications");

    let comparison_models = vec![
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
    ];

    println!("   Model Specification Comparison:");
    println!(
        "   {:<30} {:<15} {:<15} {:<15} {:<15}",
        "Model", "Context", "Max Output", "Input Cost", "Output Cost"
    );
    println!("   {}", "-".repeat(90));

    for model_id in comparison_models {
        match client.get_model(model_id.to_string()).await {
            Ok(model) => {
                let context = model
                    .context_window
                    .map(|c| format!("{}K", c / 1000))
                    .unwrap_or_else(|| "N/A".to_string());

                let max_output = model
                    .max_output_tokens
                    .map(|m| format!("{}K", m / 1000))
                    .unwrap_or_else(|| "N/A".to_string());

                let input_cost = model
                    .input_cost_per_token
                    .map(|c| format!("${:.6}", c))
                    .unwrap_or_else(|| "N/A".to_string());

                let output_cost = model
                    .output_cost_per_token
                    .map(|c| format!("${:.6}", c))
                    .unwrap_or_else(|| "N/A".to_string());

                println!(
                    "   {:<30} {:<15} {:<15} {:<15} {:<15}",
                    model_id, context, max_output, input_cost, output_cost
                );
            }
            Err(e) => {
                println!("   {:<30} Failed to get: {}", model_id, e);
            }
        }
    }

    println!();
    println!("   💡 Cost Disclaimer:");
    println!("       - Costs are displayed in USD per token.");
    println!("       - Actual costs may vary depending on region and usage.");
    println!(
        "       - It is recommended to check the official documentation for the latest pricing."
    );

    println!();
    Ok(())
}

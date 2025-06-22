//! Parameter Mapping Example
//!
//! This example demonstrates how Siumai handles parameter conversion between different providers.
//! You'll learn about common vs provider-specific parameters, validation, and custom mappings.

use siumai::prelude::*;
use siumai::params::{
    ParameterMapperFactory, ParameterMappingUtils,
    AnthropicParams, AnthropicParamsBuilder, GeminiParams, GeminiParamsBuilder,
    OllamaProviderParams,
};
use siumai::types::{CommonParams, ProviderParams, ProviderType, CacheControl};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Parameter Mapping Example");
    println!("============================\n");

    // 1. Common Parameters - Work across all providers
    demonstrate_common_parameters().await?;
    
    // 2. Provider-Specific Parameters
    demonstrate_provider_specific_parameters().await?;
    
    // 3. Parameter Validation
    demonstrate_parameter_validation().await?;
    
    // 4. Custom Parameter Mapping
    demonstrate_custom_mapping().await?;
    
    // 5. Parameter Optimization
    demonstrate_parameter_optimization().await?;

    Ok(())
}

/// Demonstrates common parameters that work across all providers
async fn demonstrate_common_parameters() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 1. Common Parameters");
    println!("   These parameters work across all providers with automatic mapping\n");

    // Create common parameters
    let common_params = CommonParams {
        model: "gpt-4o-mini".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["END".to_string(), "STOP".to_string()]),
        seed: Some(42),
    };

    println!("   Common Parameters:");
    println!("   - Model: {}", common_params.model);
    println!("   - Temperature: {:?}", common_params.temperature);
    println!("   - Max Tokens: {:?}", common_params.max_tokens);
    println!("   - Top P: {:?}", common_params.top_p);
    println!("   - Stop Sequences: {:?}", common_params.stop_sequences);
    println!("   - Seed: {:?}", common_params.seed);

    // Show how these map to different providers
    let providers = vec![
        ProviderType::OpenAi,
        ProviderType::Anthropic,
        ProviderType::Gemini,
        ProviderType::Ollama,
    ];

    for provider in providers {
        println!("\n   → Mapping to {:?}:", provider);
        let mapper = ParameterMapperFactory::create_mapper(&provider);
        let mapped = mapper.map_common_params(&common_params);
        
        // Show key differences
        match provider {
            ProviderType::OpenAi => {
                println!("     • stop_sequences → 'stop': {:?}", mapped.get("stop"));
                println!("     • seed: {:?}", mapped.get("seed"));
            }
            ProviderType::Anthropic => {
                println!("     • stop_sequences → 'stop_sequences': {:?}", mapped.get("stop_sequences"));
                println!("     • seed: removed (not supported)");
            }
            ProviderType::Gemini => {
                println!("     • max_tokens → 'maxOutputTokens': {:?}", mapped.get("maxOutputTokens"));
                println!("     • top_p → 'topP': {:?}", mapped.get("topP"));
            }
            ProviderType::Ollama => {
                println!("     • All parameters preserved with Ollama format");
                println!("     • Additional Ollama-specific options available");
            }
            _ => {}
        }
    }

    println!("\n");
    Ok(())
}

/// Demonstrates provider-specific parameters
async fn demonstrate_provider_specific_parameters() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎛️  2. Provider-Specific Parameters");
    println!("   Each provider has unique capabilities and parameters\n");

    // OpenAI-specific parameters
    println!("   🤖 OpenAI-specific parameters:");

    // Create provider-specific parameters for OpenAI
    let mut openai_provider_params = ProviderParams::new();
    openai_provider_params = openai_provider_params
        .with_param("frequency_penalty", 0.1)
        .with_param("presence_penalty", 0.1)
        .with_param("logit_bias", serde_json::json!({"50256": -100}));

    println!("     • frequency_penalty: Controls repetition");
    println!("     • presence_penalty: Encourages new topics");
    println!("     • logit_bias: Biases token selection");

    // Anthropic-specific parameters
    println!("\n   🧠 Anthropic-specific parameters:");
    let mut metadata = HashMap::new();
    metadata.insert("user_id".to_string(), "demo".to_string());

    let anthropic_params = AnthropicParamsBuilder::new()
        .cache_control(CacheControl::ephemeral())
        .thinking_budget(1000)
        .system("You are a helpful assistant".to_string())
        .metadata(metadata)
        .build();

    println!("     • cache_control: Response caching");
    println!("     • thinking_budget: Reasoning token limit");
    println!("     • system: System message handling");
    println!("     • metadata: Request metadata");

    // Ollama-specific parameters
    println!("\n   🦙 Ollama-specific parameters:");
    let ollama_params = OllamaProviderParams {
        num_ctx: Some(4096),
        num_batch: Some(512),
        num_gpu: Some(1),
        main_gpu: Some(0),
        numa: Some(false),
        keep_alive: Some("10m".to_string()),
        ..Default::default()
    };

    println!("     • num_ctx: Context window size");
    println!("     • num_batch: Batch size for processing");
    println!("     • num_gpu: Number of GPUs to use");
    println!("     • keep_alive: Model persistence time");

    println!("\n");
    Ok(())
}

/// Demonstrates parameter validation
async fn demonstrate_parameter_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️  3. Parameter Validation");
    println!("   Siumai validates parameters for each provider\n");

    // Valid parameters
    let valid_params = serde_json::json!({
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9
    });

    println!("   ✅ Valid parameters:");
    for provider in [ProviderType::OpenAi, ProviderType::Anthropic, ProviderType::Gemini] {
        let result = ParameterMappingUtils::validate_for_provider(&valid_params, &provider);
        println!("     • {:?}: {}", provider, if result.is_ok() { "✓" } else { "✗" });
    }

    // Invalid parameters
    println!("\n   ❌ Invalid parameters (temperature too high):");
    let invalid_params = serde_json::json!({
        "model": "gpt-4o-mini",
        "temperature": 3.0,  // Too high for most providers
        "max_tokens": 1000
    });

    for provider in [ProviderType::OpenAi, ProviderType::Anthropic, ProviderType::Gemini] {
        let result = ParameterMappingUtils::validate_for_provider(&invalid_params, &provider);
        match result {
            Ok(_) => println!("     • {:?}: ✓", provider),
            Err(e) => println!("     • {:?}: ✗ ({})", provider, e),
        }
    }

    // Provider-specific validation
    println!("\n   🔍 Provider-specific validation:");
    
    // Anthropic thinking budget validation
    let anthropic_params = serde_json::json!({
        "thinking_budget": 70_000  // Too high for Anthropic
    });
    
    let result = ParameterMappingUtils::validate_for_provider(&anthropic_params, &ProviderType::Anthropic);
    match result {
        Ok(_) => println!("     • Anthropic thinking budget: ✓"),
        Err(e) => println!("     • Anthropic thinking budget: ✗ ({})", e),
    }

    println!("\n");
    Ok(())
}

/// Demonstrates custom parameter mapping
async fn demonstrate_custom_mapping() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 4. Custom Parameter Mapping");
    println!("   Create custom mappings for specific use cases\n");

    // Custom provider parameters
    let mut custom_params = ProviderParams::new();
    custom_params = custom_params
        .with_param("custom_setting", "value")
        .with_param("optimization_level", 2)
        .with_param("experimental_feature", true);

    println!("   Custom parameters added:");
    println!("   - custom_setting: value");
    println!("   - optimization_level: 2");
    println!("   - experimental_feature: true");

    // Convert and merge with common parameters
    let common_params = CommonParams {
        model: "gpt-4o-mini".to_string(),
        temperature: Some(0.8),
        max_tokens: Some(500),
        ..Default::default()
    };

    let final_params = ParameterMappingUtils::convert_params(
        &common_params,
        Some(&custom_params),
        &ProviderType::OpenAi,
    )?;

    println!("\n   Final merged parameters:");
    println!("   {}", serde_json::to_string_pretty(&final_params)?);

    println!("\n");
    Ok(())
}

/// Demonstrates parameter optimization
async fn demonstrate_parameter_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ 5. Parameter Optimization");
    println!("   Optimize parameters for different use cases\n");

    // Different optimization scenarios
    let scenarios = vec![
        ("Creative Writing", 0.9, 2000, Some(0.95)),
        ("Code Generation", 0.1, 1000, Some(0.1)),
        ("Data Analysis", 0.3, 1500, Some(0.8)),
        ("General Chat", 0.7, 1000, Some(0.9)),
    ];

    for (use_case, temp, max_tokens, top_p) in scenarios {
        println!("   📝 {}", use_case);
        println!("     • Temperature: {} ({})", temp, 
            if temp > 0.8 { "high creativity" } 
            else if temp < 0.3 { "high precision" } 
            else { "balanced" });
        println!("     • Max Tokens: {} ({})", max_tokens,
            if max_tokens > 1500 { "long responses" } else { "concise responses" });
        println!("     • Top P: {:?} ({})", top_p,
            if top_p.unwrap_or(1.0) > 0.9 { "diverse vocabulary" } else { "focused vocabulary" });
        println!();
    }

    // Show parameter constraints for different providers
    println!("   📊 Provider Constraints:");
    for provider in [ProviderType::OpenAi, ProviderType::Anthropic, ProviderType::Gemini] {
        let constraints = ParameterMappingUtils::get_constraints(&provider);
        println!("     • {:?}:", provider);
        println!("       - Temperature range: {:.1} - {:.1}", constraints.temperature_min, constraints.temperature_max);
        println!("       - Max tokens range: {} - {}", constraints.max_tokens_min, constraints.max_tokens_max);
        println!("       - Top P range: {:.1} - {:.1}", constraints.top_p_min, constraints.top_p_max);
    }

    println!("\n✨ Parameter mapping complete! You now understand how Siumai handles");
    println!("   parameter conversion, validation, and optimization across providers.");

    Ok(())
}

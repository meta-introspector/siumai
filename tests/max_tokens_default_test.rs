//! Max Tokens Default Value Tests
//!
//! Tests to ensure all providers handle max_tokens defaults correctly.

use siumai::prelude::*;
use siumai::types::CommonParams;
use siumai::params::mapper::ParameterMapperFactory;

#[test]
fn test_anthropic_max_tokens_default() {
    let mapper = ParameterMapperFactory::create_mapper(&ProviderType::Anthropic);
    
    // Test without max_tokens
    let params_without_max_tokens = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(0.7),
        max_tokens: None, // No max_tokens provided
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    let mapped = mapper.map_common_params(&params_without_max_tokens);
    
    // Anthropic should automatically set default max_tokens
    assert_eq!(mapped["max_tokens"], 4096);
    
    // Test with explicit max_tokens
    let params_with_max_tokens = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(2000), // Explicit max_tokens
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    let mapped_explicit = mapper.map_common_params(&params_with_max_tokens);
    
    // Should use the explicit value
    assert_eq!(mapped_explicit["max_tokens"], 2000);
}

#[test]
fn test_openai_max_tokens_optional() {
    let mapper = ParameterMapperFactory::create_mapper(&ProviderType::OpenAi);
    
    // Test without max_tokens
    let params_without_max_tokens = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        max_tokens: None, // No max_tokens provided
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    let mapped = mapper.map_common_params(&params_without_max_tokens);
    
    // OpenAI should not have max_tokens if not provided
    assert!(mapped.get("max_tokens").is_none());
    
    // Test with explicit max_tokens
    let params_with_max_tokens = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(2000), // Explicit max_tokens
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    let mapped_explicit = mapper.map_common_params(&params_with_max_tokens);
    
    // Should use the explicit value
    assert_eq!(mapped_explicit["max_tokens"], 2000);
}

#[test]
fn test_gemini_max_tokens_optional() {
    let mapper = ParameterMapperFactory::create_mapper(&ProviderType::Gemini);

    // Test without max_tokens
    let params_without_max_tokens = CommonParams {
        model: "gemini-1.5-pro".to_string(),
        temperature: Some(0.7),
        max_tokens: None, // No max_tokens provided
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };

    let mapped = mapper.map_common_params(&params_without_max_tokens);

    // Gemini should not have maxOutputTokens if not provided
    assert!(mapped.get("maxOutputTokens").is_none());

    // Test with explicit max_tokens
    let params_with_max_tokens = CommonParams {
        model: "gemini-1.5-pro".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(2000), // Explicit max_tokens
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };

    let mapped_explicit = mapper.map_common_params(&params_with_max_tokens);

    // Should use the explicit value as maxOutputTokens
    assert_eq!(mapped_explicit["maxOutputTokens"], 2000);
}

#[test]
fn test_ollama_max_tokens_optional() {
    let mapper = ParameterMapperFactory::create_mapper(&ProviderType::Ollama);
    
    // Test without max_tokens
    let params_without_max_tokens = CommonParams {
        model: "llama3.2".to_string(),
        temperature: Some(0.7),
        max_tokens: None, // No max_tokens provided
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    let mapped = mapper.map_common_params(&params_without_max_tokens);
    
    // Ollama should not have num_predict if not provided
    assert!(mapped.get("num_predict").is_none());
    
    // Test with explicit max_tokens
    let params_with_max_tokens = CommonParams {
        model: "llama3.2".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(2000), // Explicit max_tokens
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    let mapped_explicit = mapper.map_common_params(&params_with_max_tokens);
    
    // Should use the explicit value as num_predict
    assert_eq!(mapped_explicit["num_predict"], 2000);
}

#[test]
fn test_groq_max_tokens_optional() {
    let mapper = ParameterMapperFactory::create_mapper(&ProviderType::Groq);

    // Test without max_tokens (Groq uses OpenAI format)
    let params_without_max_tokens = CommonParams {
        model: "llama-3.3-70b-versatile".to_string(),
        temperature: Some(0.7),
        max_tokens: None, // No max_tokens provided
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };

    let mapped = mapper.map_common_params(&params_without_max_tokens);

    // Groq (OpenAI format) should not have max_tokens if not provided
    assert!(mapped.get("max_tokens").is_none());

    // Test with explicit max_tokens
    let params_with_max_tokens = CommonParams {
        model: "llama-3.3-70b-versatile".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(2000), // Explicit max_tokens
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };

    let mapped_explicit = mapper.map_common_params(&params_with_max_tokens);

    // Should use the explicit value
    assert_eq!(mapped_explicit["max_tokens"], 2000);
}

#[test]
fn test_xai_max_tokens_optional() {
    let mapper = ParameterMapperFactory::create_mapper(&ProviderType::XAI);
    
    // Test without max_tokens (XAI uses OpenAI format)
    let params_without_max_tokens = CommonParams {
        model: "grok-3-latest".to_string(),
        temperature: Some(0.7),
        max_tokens: None, // No max_tokens provided
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    let mapped = mapper.map_common_params(&params_without_max_tokens);
    
    // XAI (OpenAI format) should not have max_tokens if not provided
    assert!(mapped.get("max_tokens").is_none());
    
    // Test with explicit max_tokens
    let params_with_max_tokens = CommonParams {
        model: "grok-3-latest".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(2000), // Explicit max_tokens
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    let mapped_explicit = mapper.map_common_params(&params_with_max_tokens);
    
    // Should use the explicit value
    assert_eq!(mapped_explicit["max_tokens"], 2000);
}



#[tokio::test]
async fn test_anthropic_validation_requires_max_tokens() {
    // Test that Anthropic validation fails without max_tokens
    let mapper = ParameterMapperFactory::create_mapper(&ProviderType::Anthropic);
    
    let params_without_max_tokens = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(0.7),
        max_tokens: None,
        top_p: Some(0.9),
        stop_sequences: None,
        seed: None,
    };
    
    // Map the parameters (this should add default max_tokens)
    let mapped = mapper.map_common_params(&params_without_max_tokens);
    
    // Validation should pass because default max_tokens was added
    let validation_result = mapper.validate_params(&mapped);
    assert!(validation_result.is_ok());
    
    // Manually create params without max_tokens to test validation
    let mut manual_params = serde_json::json!({
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
        "top_p": 0.9
    });
    
    // Remove max_tokens if it exists
    manual_params.as_object_mut().unwrap().remove("max_tokens");
    
    // This should fail validation
    let validation_result = mapper.validate_params(&manual_params);
    assert!(validation_result.is_err());
    assert!(validation_result.unwrap_err().to_string().contains("max_tokens is required"));
}

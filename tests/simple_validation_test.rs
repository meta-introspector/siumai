//! Simple validation test to verify everything works without feature flags

use siumai::params::{AnthropicParams, OpenAiParams};
use siumai::prelude::*;

#[test]
fn test_core_validation_works() {
    // Test valid parameters - only core validations
    let valid_params = CommonParams::builder()
        .model("gpt-4".to_string())
        .temperature(0.7)
        .unwrap() // Core validation: 0.0-2.0
        .max_tokens(1000) // No validation needed
        .top_p(0.9)
        .unwrap() // Core validation: 0.0-1.0
        .build();

    assert!(valid_params.is_ok());

    // Test invalid temperature
    let invalid_temp = CommonParams::builder()
        .model("gpt-4".to_string())
        .temperature(5.0); // Invalid: > 2.0

    assert!(invalid_temp.is_err());

    // Test invalid top_p
    let invalid_top_p = CommonParams::builder()
        .model("gpt-4".to_string())
        .top_p(1.5); // Invalid: > 1.0

    assert!(invalid_top_p.is_err());
}

#[test]
fn test_openai_core_validation_works() {
    // Test valid OpenAI parameters - only core validations
    let valid_params = OpenAiParams::builder()
        .frequency_penalty(0.5)
        .unwrap() // Core validation: -2.0 to 2.0
        .presence_penalty(-0.2)
        .unwrap() // Core validation: -2.0 to 2.0
        .n(2) // No validation needed
        .user("test-user".to_string()) // No validation needed
        .build();

    assert!(valid_params.is_ok());

    // Test invalid frequency penalty
    let invalid_freq = OpenAiParams::builder().frequency_penalty(5.0); // Invalid: > 2.0

    assert!(invalid_freq.is_err());
}

#[test]
fn test_anthropic_params_work() {
    // Test Anthropic parameters - no core validations needed
    let params = AnthropicParams {
        thinking_budget: Some(1000),                       // No validation
        system: Some("You are helpful".to_string()),       // No validation
        beta_features: Some(vec!["thinking".to_string()]), // No validation
        ..Default::default()
    };

    // Should always work since we removed the validations
    assert!(params.validate_params().is_ok());
}

#[test]
fn test_basic_macros_work() {
    // Test basic macros - just verify they compile
    let _user_msg = user!("Hello!");
    let _system_msg = system!("You are helpful");

    // If we get here, the macros work - no assertion needed
}

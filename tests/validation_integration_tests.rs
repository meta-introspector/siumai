//! Basic Integration Tests
//!
//! Tests that verify basic functionality works correctly

use siumai::error::LlmError;
use siumai::params::OpenAiParams;
use siumai::prelude::*;

#[cfg(test)]
mod basic_tests {
    use super::*;

    #[test]
    fn test_common_params_builder() {
        // Test basic parameter building
        let common_params = CommonParams::builder()
            .model("gpt-4".to_string())
            .temperature(0.7)
            .unwrap()
            .max_tokens(1500)
            .top_p(0.9)
            .unwrap()
            .build();

        assert!(common_params.is_ok());
        let params = common_params.unwrap();

        // Verify the parameters are correctly set
        assert_eq!(params.model, "gpt-4");
        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.max_tokens, Some(1500));
        assert_eq!(params.top_p, Some(0.9));
    }

    #[test]
    fn test_openai_params_builder() {
        let openai_params = OpenAiParams::builder()
            .frequency_penalty(0.2)
            .unwrap()
            .presence_penalty(-0.1)
            .unwrap()
            .n(1)
            .user("test-user-123".to_string())
            .build();

        assert!(openai_params.is_ok());
        let params = openai_params.unwrap();

        assert_eq!(params.frequency_penalty, Some(0.2));
        assert_eq!(params.presence_penalty, Some(-0.1));
        assert_eq!(params.n, Some(1));
        assert_eq!(params.user, Some("test-user-123".to_string()));
    }

    #[test]
    fn test_temperature_validation() {
        // Test that invalid temperature is rejected
        let result = CommonParams::builder()
            .model("gpt-4".to_string())
            .temperature(3.0); // This should fail - temperature > 2.0

        assert!(result.is_err());
        if let Err(LlmError::InvalidParameter(msg)) = result {
            assert!(msg.contains("Temperature must be between 0.0 and 2.0"));
        } else {
            panic!("Expected InvalidParameter error");
        }
    }

    #[test]
    fn test_frequency_penalty_validation() {
        // Test that invalid frequency penalty is rejected
        let result = OpenAiParams::builder().frequency_penalty(5.0); // Invalid: > 2.0

        assert!(result.is_err());
        if let Err(LlmError::InvalidParameter(msg)) = result {
            assert!(msg.contains("Frequency penalty must be between -2.0 and 2.0"));
        } else {
            panic!("Expected InvalidParameter error");
        }
    }
}

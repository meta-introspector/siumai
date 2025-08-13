//! 🛡️ Error Handling - Production-ready error management
//!
//! This example demonstrates robust error handling patterns:
//! - Different error types and their handling
//! - Retry strategies for transient errors
//! - Rate limit handling and backoff
//! - Graceful degradation patterns
//! - Error logging and monitoring
//!
//! Before running, set your API keys:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example error_handling
//! ```

use siumai::models;
use siumai::prelude::*;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ Error Handling - Production-ready error management\n");

    // Demonstrate different error handling patterns
    demonstrate_error_types().await;
    demonstrate_retry_strategies().await;
    demonstrate_rate_limit_handling().await;
    demonstrate_graceful_degradation().await;
    demonstrate_error_classification().await;

    println!("\n✅ Error handling examples completed!");
    Ok(())
}

/// Demonstrate different types of errors and their characteristics
async fn demonstrate_error_types() {
    println!("🔍 Error Types and Classification:\n");

    // Test different error scenarios
    println!("   Testing Invalid API Key:");
    match test_invalid_api_key().await {
        Ok(_) => println!("      ❌ Expected error but got success"),
        Err(e) => {
            println!("      ✅ Got expected error: {e}");
            demonstrate_error_handling(&e);
        }
    }

    println!("\n   Testing Invalid Model:");
    match test_invalid_model().await {
        Ok(_) => println!("      ❌ Expected error but got success"),
        Err(e) => {
            println!("      ✅ Got expected error: {e}");
            demonstrate_error_handling(&e);
        }
    }

    println!("\n   Testing Network Timeout:");
    match test_network_timeout().await {
        Ok(_) => println!("      ❌ Expected error but got success"),
        Err(e) => {
            println!("      ✅ Got expected error: {e}");
            demonstrate_error_handling(&e);
        }
    }

    println!("\n   Testing Rate Limit:");
    match test_rate_limit().await {
        Ok(_) => println!("      ❌ Expected error but got success"),
        Err(e) => {
            println!("      ✅ Got expected error: {e}");
            demonstrate_error_handling(&e);
        }
    }

    println!("\n   Testing Invalid Request:");
    match test_invalid_request().await {
        Ok(_) => println!("      ❌ Expected error but got success"),
        Err(e) => {
            println!("      ✅ Got expected error: {e}");
            demonstrate_error_handling(&e);
        }
    }
}

/// Demonstrate retry strategies for transient errors
async fn demonstrate_retry_strategies() {
    println!("🔄 Retry Strategies:\n");

    let message = "Hello! This is a test message.";

    // Strategy 1: Simple retry with exponential backoff
    println!("   Strategy 1: Exponential Backoff");
    match retry_with_exponential_backoff(message, 3).await {
        Ok(response) => {
            println!("      ✅ Success after retries");
            if let Some(text) = response.content_text() {
                println!("      Response: {}", &text[..text.len().min(50)]);
            }
        }
        Err(e) => {
            println!("      ❌ Failed after all retries: {e}");
        }
    }

    // Strategy 2: Retry with jitter
    println!("\n   Strategy 2: Retry with Jitter");
    match retry_with_jitter(message, 3).await {
        Ok(response) => {
            println!("      ✅ Success with jitter strategy");
            if let Some(text) = response.content_text() {
                println!("      Response: {}", &text[..text.len().min(50)]);
            }
        }
        Err(e) => {
            println!("      ❌ Failed with jitter strategy: {e}");
        }
    }

    println!();
}

/// Demonstrate rate limit handling
async fn demonstrate_rate_limit_handling() {
    println!("⏱️ Rate Limit Handling:\n");

    // Simulate rate limit scenario
    println!("   Testing rate limit detection and handling...");

    match handle_rate_limits("Test rate limit handling").await {
        Ok(response) => {
            println!("   ✅ Successfully handled rate limits");
            if let Some(text) = response.content_text() {
                println!("   Response: {}", &text[..text.len().min(100)]);
            }
        }
        Err(e) => {
            println!("   ❌ Rate limit handling failed: {e}");
        }
    }

    println!();
}

/// Demonstrate graceful degradation
async fn demonstrate_graceful_degradation() {
    println!("🎭 Graceful Degradation:\n");

    let user_message = "Explain quantum computing";

    match chat_with_graceful_degradation(user_message).await {
        Ok((provider, response)) => {
            println!("   ✅ Successfully used provider: {provider}");
            if let Some(text) = response.content_text() {
                println!("   Response: {}", &text[..text.len().min(100)]);
            }
        }
        Err(e) => {
            println!("   ❌ All degradation strategies failed: {e}");
            println!("   💡 In production, you might return a cached response or error message");
        }
    }

    println!();
}

/// Demonstrate error classification for monitoring
async fn demonstrate_error_classification() {
    println!("📊 Error Classification for Monitoring:\n");

    // Simulate various errors and classify them
    let test_errors = vec![
        LlmError::AuthenticationError("Invalid API key".to_string()),
        LlmError::RateLimitError("Rate limit exceeded".to_string()),
        LlmError::TimeoutError("Request timed out".to_string()),
        LlmError::ModelNotSupported("gpt-5".to_string()),
        LlmError::InternalError("Network error".to_string()),
    ];

    for error in test_errors {
        println!("   Error: {error}");

        let classification = classify_error_for_monitoring(&error);
        println!("      Classification: {classification:?}");
        println!("      Action: {}", get_recommended_action(&classification));
        println!();
    }
}

/// Test invalid API key scenario
async fn test_invalid_api_key() -> Result<ChatResponse, LlmError> {
    let client = LlmBuilder::new()
        .openai()
        .api_key("invalid-key-12345")
        .model(models::openai::GPT_4O_MINI)
        .build()
        .await?;

    let messages = vec![user!("Hello")];
    client.chat(messages).await
}

/// Test invalid model scenario
async fn test_invalid_model() -> Result<ChatResponse, LlmError> {
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = LlmBuilder::new()
            .openai()
            .api_key(&api_key)
            .model("gpt-nonexistent-model")
            .build()
            .await?;

        let messages = vec![user!("Hello")];
        client.chat(messages).await
    } else {
        Err(LlmError::AuthenticationError("No API key".to_string()))
    }
}

/// Test network timeout scenario
async fn test_network_timeout() -> Result<ChatResponse, LlmError> {
    // This would require a client with very short timeout
    // For demo purposes, we'll simulate it
    Err(LlmError::TimeoutError("Simulated timeout".to_string()))
}

/// Test rate limit scenario
async fn test_rate_limit() -> Result<ChatResponse, LlmError> {
    // Simulate rate limit error
    Err(LlmError::RateLimitError("Rate limit exceeded".to_string()))
}

/// Test invalid request scenario
async fn test_invalid_request() -> Result<ChatResponse, LlmError> {
    // Simulate invalid request
    Err(LlmError::InternalError(
        "Invalid request format".to_string(),
    ))
}

/// Retry with exponential backoff
async fn retry_with_exponential_backoff(
    message: &str,
    max_retries: u32,
) -> Result<ChatResponse, LlmError> {
    let mut delay = Duration::from_millis(100);

    for attempt in 1..=max_retries {
        match try_chat_request(message).await {
            Ok(response) => {
                println!("      ✅ Success on attempt {attempt}");
                return Ok(response);
            }
            Err(e) if is_retryable_error(&e) && attempt < max_retries => {
                println!("      ⏳ Attempt {attempt} failed, retrying in {delay:?}");
                sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
            Err(e) => {
                println!("      ❌ Non-retryable error or max retries reached: {e}");
                return Err(e);
            }
        }
    }

    Err(LlmError::InternalError("Max retries exceeded".to_string()))
}

/// Retry with jitter to avoid thundering herd
async fn retry_with_jitter(message: &str, max_retries: u32) -> Result<ChatResponse, LlmError> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for attempt in 1..=max_retries {
        match try_chat_request(message).await {
            Ok(response) => {
                println!("      ✅ Success on attempt {attempt} with jitter");
                return Ok(response);
            }
            Err(e) if is_retryable_error(&e) && attempt < max_retries => {
                let base_delay = 100 * (1 << (attempt - 1)); // Exponential base
                let jitter = rng.gen_range(0..=base_delay / 2); // Add jitter
                let delay = Duration::from_millis(base_delay + jitter);

                println!("      ⏳ Attempt {attempt} failed, retrying in {delay:?} (with jitter)");
                sleep(delay).await;
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    Err(LlmError::InternalError("Max retries exceeded".to_string()))
}

/// Handle rate limits with appropriate backoff
async fn handle_rate_limits(message: &str) -> Result<ChatResponse, LlmError> {
    match try_chat_request(message).await {
        Ok(response) => Ok(response),
        Err(LlmError::RateLimitError(_)) => {
            println!("   ⏳ Rate limit detected, waiting 60 seconds...");
            sleep(Duration::from_secs(60)).await;

            // Retry after rate limit wait
            try_chat_request(message).await
        }
        Err(e) => Err(e),
    }
}

/// Chat with graceful degradation
async fn chat_with_graceful_degradation(message: &str) -> Result<(String, ChatResponse), LlmError> {
    // Try primary provider
    if let Ok(response) = try_chat_request(message).await {
        return Ok(("Primary".to_string(), response));
    }

    // Try Ollama as fallback
    if let Ok(client) = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .build()
        .await
    {
        let messages = vec![user!(message)];
        if let Ok(response) = client.chat(messages).await {
            return Ok(("Ollama Fallback".to_string(), response));
        }
    }

    // Final fallback: return a helpful error message
    Err(LlmError::InternalError(
        "All providers unavailable".to_string(),
    ))
}

/// Try a chat request with the best available provider
async fn try_chat_request(message: &str) -> Result<ChatResponse, LlmError> {
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = LlmBuilder::new()
            .openai()
            .api_key(&api_key)
            .model(models::openai::GPT_4O_MINI)
            .build()
            .await?;

        let messages = vec![user!(message)];
        client.chat(messages).await
    } else {
        Err(LlmError::AuthenticationError(
            "No API key available".to_string(),
        ))
    }
}

/// Check if an error is retryable
const fn is_retryable_error(error: &LlmError) -> bool {
    matches!(
        error,
        LlmError::TimeoutError(_) | LlmError::RateLimitError(_) | LlmError::InternalError(_)
    )
}

/// Check if an error is authentication-related
const fn is_auth_error(error: &LlmError) -> bool {
    matches!(error, LlmError::AuthenticationError(_))
}

/// Check if an error is rate limit-related
const fn is_rate_limit_error(error: &LlmError) -> bool {
    matches!(error, LlmError::RateLimitError(_))
}

/// Check if an error is a client error (4xx)
const fn is_client_error(error: &LlmError) -> bool {
    matches!(
        error,
        LlmError::AuthenticationError(_) | LlmError::ModelNotSupported(_)
    )
}

/// Error classification for monitoring
#[derive(Debug)]
enum ErrorClassification {
    Transient,      // Temporary issues, retry
    Authentication, // Auth problems, check credentials
    RateLimit,      // Rate limiting, backoff
    ClientError,    // Client-side issues, fix request
    ServerError,    // Server-side issues, contact support
}

/// Classify error for monitoring and alerting
const fn classify_error_for_monitoring(error: &LlmError) -> ErrorClassification {
    match error {
        LlmError::AuthenticationError(_) => ErrorClassification::Authentication,
        LlmError::RateLimitError(_) => ErrorClassification::RateLimit,
        LlmError::TimeoutError(_) => ErrorClassification::Transient,
        LlmError::ModelNotSupported(_) => ErrorClassification::ClientError,
        LlmError::InternalError(_) => ErrorClassification::ServerError,
        _ => ErrorClassification::ServerError, // Default for other error types
    }
}

/// Get recommended action for error classification
const fn get_recommended_action(classification: &ErrorClassification) -> &'static str {
    match classification {
        ErrorClassification::Transient => "Retry with exponential backoff",
        ErrorClassification::Authentication => "Check API credentials",
        ErrorClassification::RateLimit => "Implement rate limiting and backoff",
        ErrorClassification::ClientError => "Fix request parameters",
        ErrorClassification::ServerError => "Monitor and escalate if persistent",
    }
}

/// Demonstrate error handling for a specific error
fn demonstrate_error_handling(error: &LlmError) {
    println!("      📊 Error Analysis:");
    println!("         - Retryable: {}", is_retryable_error(error));
    println!("         - Auth error: {}", is_auth_error(error));
    println!("         - Rate limit: {}", is_rate_limit_error(error));
    println!("         - Client error: {}", is_client_error(error));

    let classification = classify_error_for_monitoring(error);
    println!("         - Classification: {classification:?}");
    println!(
        "         - Recommended action: {}",
        get_recommended_action(&classification)
    );
}

/*
🎯 Key Error Handling Concepts:

Error Types:
- Authentication: Invalid API keys, expired tokens
- Rate Limits: Too many requests, quota exceeded
- Timeouts: Network issues, slow responses
- Client Errors: Invalid requests, unsupported models
- Server Errors: Provider outages, internal errors

Retry Strategies:
- Exponential backoff: Increase delay between retries
- Jitter: Add randomness to prevent thundering herd
- Circuit breaker: Stop retrying after threshold
- Selective retry: Only retry transient errors

Best Practices:
1. Classify errors appropriately
2. Implement proper retry logic
3. Use graceful degradation
4. Log errors for monitoring
5. Provide meaningful user feedback
6. Set reasonable timeouts
7. Monitor error rates and patterns

Production Considerations:
- Error tracking and alerting
- Graceful degradation strategies
- User experience during failures
- Cost implications of retries
- Provider SLA monitoring

Next Steps:
- capability_detection.rs: Feature detection patterns
- ../03_advanced_features/: Advanced error handling
- ../05_use_cases/: Production error handling examples
*/

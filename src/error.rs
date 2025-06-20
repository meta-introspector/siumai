//! Error handling module
//!
//! Defines all error types used in the LLM library.

use thiserror::Error;

/// Error category for better error handling and recovery strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Network-related errors (connection, timeout, etc.)
    Network,
    /// Authentication and authorization errors
    Authentication,
    /// Rate limiting and quota errors
    RateLimit,
    /// Client-side errors (4xx HTTP status codes)
    Client,
    /// Server-side errors (5xx HTTP status codes)
    Server,
    /// Data parsing and serialization errors
    Parsing,
    /// Input validation errors
    Validation,
    /// Configuration errors
    Configuration,
    /// Unsupported operations or models
    Unsupported,
    /// Streaming-related errors
    Stream,
    /// Provider-specific errors
    Provider,
    /// Unknown or uncategorized errors
    Unknown,
}

/// The primary error type for the LLM library.
#[derive(Error, Debug, Clone)]
pub enum LlmError {
    /// HTTP request error
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(String),

    /// Response parsing error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Not found error
    #[error("Not found: {0}")]
    NotFound(String),

    /// Missing API key
    #[error("Missing API key: {0}")]
    MissingApiKey(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// API error response
    #[error("API error: {code} - {message}")]
    ApiError {
        code: u16,
        message: String,
        details: Option<serde_json::Value>,
    },

    /// Authentication error
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Rate limit error
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),

    /// Quota exceeded error
    #[error("Quota exceeded: {0}")]
    QuotaExceededError(String),

    /// Model not supported error
    #[error("Model not supported: {0}")]
    ModelNotSupported(String),

    /// Stream processing error
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Timeout error
    #[error("Request timeout: {0}")]
    TimeoutError(String),

    /// Network connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Provider-specific error
    #[error("Provider error ({provider}): {message}")]
    ProviderError {
        provider: String,
        message: String,
        error_code: Option<String>,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Processing error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Tool call error
    #[error("Tool call error: {0}")]
    ToolCallError(String),

    /// Tool validation error
    #[error("Tool validation error: {0}")]
    ToolValidationError(String),

    /// Unsupported tool type
    #[error("Unsupported tool type: {0}")]
    UnsupportedToolType(String),

    /// Other errors
    #[error("Other error: {0}")]
    Other(String),
}

impl LlmError {
    /// Creates a new API error.
    pub fn api_error(code: u16, message: impl Into<String>) -> Self {
        Self::ApiError {
            code,
            message: message.into(),
            details: None,
        }
    }

    /// Creates a new API error with details.
    pub fn api_error_with_details(
        code: u16,
        message: impl Into<String>,
        details: serde_json::Value,
    ) -> Self {
        Self::ApiError {
            code,
            message: message.into(),
            details: Some(details),
        }
    }

    /// Creates a new provider error.
    pub fn provider_error(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ProviderError {
            provider: provider.into(),
            message: message.into(),
            error_code: None,
        }
    }

    /// Creates a new provider error with an error code.
    pub fn provider_error_with_code(
        provider: impl Into<String>,
        message: impl Into<String>,
        error_code: impl Into<String>,
    ) -> Self {
        Self::ProviderError {
            provider: provider.into(),
            message: message.into(),
            error_code: Some(error_code.into()),
        }
    }

    /// Checks if the error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::HttpError(e) => {
                // Check for keywords in the error message to determine retryability.
                e.contains("timeout") || e.contains("connect") || e.contains("network")
            }
            Self::ApiError { code, .. } => {
                // 5xx and 429 errors are retryable.
                *code >= 500 || *code == 429
            }
            Self::RateLimitError(_) => true,
            Self::TimeoutError(_) => true,
            Self::ConnectionError(_) => true,
            _ => false,
        }
    }

    /// Checks if the error is an authentication-related error.
    pub fn is_auth_error(&self) -> bool {
        match self {
            Self::AuthenticationError(_) => true,
            Self::ApiError { code, .. } => *code == 401 || *code == 403,
            _ => false,
        }
    }

    /// Checks if the error is a rate limit error.
    pub fn is_rate_limit_error(&self) -> bool {
        match self {
            Self::RateLimitError(_) => true,
            Self::ApiError { code, .. } => *code == 429,
            _ => false,
        }
    }

    /// Gets the HTTP status code of the error, if available.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::ApiError { code, .. } => Some(*code),
            Self::HttpError(_) => None, // Cannot get status code directly from the string form.
            _ => None,
        }
    }

    /// Gets the error category for better error handling.
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::HttpError(_) | Self::ConnectionError(_) | Self::TimeoutError(_) => ErrorCategory::Network,
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => ErrorCategory::Authentication,
            Self::RateLimitError(_) | Self::QuotaExceededError(_) => ErrorCategory::RateLimit,
            Self::ApiError { code, .. } => {
                match *code {
                    400..=499 => ErrorCategory::Client,
                    500..=599 => ErrorCategory::Server,
                    _ => ErrorCategory::Unknown,
                }
            }
            Self::JsonError(_) | Self::ParseError(_) => ErrorCategory::Parsing,
            Self::InvalidInput(_) | Self::InvalidParameter(_) | Self::ToolValidationError(_) => ErrorCategory::Validation,
            Self::ConfigurationError(_) => ErrorCategory::Configuration,
            Self::ModelNotSupported(_) | Self::UnsupportedOperation(_) | Self::UnsupportedToolType(_) => ErrorCategory::Unsupported,
            Self::StreamError(_) => ErrorCategory::Stream,
            Self::ProviderError { .. } | Self::ToolCallError(_) => ErrorCategory::Provider,
            _ => ErrorCategory::Unknown,
        }
    }

    /// Gets a user-friendly error message.
    pub fn user_message(&self) -> String {
        match self {
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => {
                "Authentication failed. Please check your API key.".to_string()
            }
            Self::RateLimitError(_) => {
                "Rate limit exceeded. Please wait before making more requests.".to_string()
            }
            Self::QuotaExceededError(_) => {
                "API quota exceeded. Please check your usage limits.".to_string()
            }
            Self::ModelNotSupported(model) => {
                format!("The model '{}' is not supported by this provider.", model)
            }
            Self::ConnectionError(_) | Self::TimeoutError(_) => {
                "Network connection failed. Please check your internet connection and try again.".to_string()
            }
            Self::ApiError { code: 500..=599, .. } => {
                "The service is temporarily unavailable. Please try again later.".to_string()
            }
            _ => self.to_string(),
        }
    }

    /// Gets suggested recovery actions for the error.
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => {
                vec![
                    "Verify your API key is correct".to_string(),
                    "Check if your API key has the required permissions".to_string(),
                    "Ensure your API key is not expired".to_string(),
                ]
            }
            Self::RateLimitError(_) => {
                vec![
                    "Wait before making more requests".to_string(),
                    "Implement exponential backoff".to_string(),
                    "Consider upgrading your API plan".to_string(),
                ]
            }
            Self::QuotaExceededError(_) => {
                vec![
                    "Check your usage dashboard".to_string(),
                    "Upgrade your API plan".to_string(),
                    "Wait for quota reset".to_string(),
                ]
            }
            Self::ConnectionError(_) | Self::TimeoutError(_) => {
                vec![
                    "Check your internet connection".to_string(),
                    "Retry the request".to_string(),
                    "Increase timeout settings".to_string(),
                ]
            }
            Self::ModelNotSupported(_) => {
                vec![
                    "Use a supported model".to_string(),
                    "Check the provider's model list".to_string(),
                ]
            }
            Self::ApiError { code: 500..=599, .. } => {
                vec![
                    "Retry the request after a delay".to_string(),
                    "Check the service status page".to_string(),
                ]
            }
            _ => vec!["Check the error details and documentation".to_string()],
        }
    }
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, LlmError>;

// From implementations
impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        Self::HttpError(err.to_string())
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(err: serde_json::Error) -> Self {
        Self::JsonError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = LlmError::api_error(404, "Not found");
        assert!(matches!(error, LlmError::ApiError { code: 404, .. }));
    }

    #[test]
    fn test_retryable_errors() {
        let rate_limit = LlmError::RateLimitError("Too many requests".to_string());
        assert!(rate_limit.is_retryable());

        let server_error = LlmError::api_error(500, "Internal server error");
        assert!(server_error.is_retryable());

        let auth_error = LlmError::AuthenticationError("Invalid key".to_string());
        assert!(!auth_error.is_retryable());
    }

    #[test]
    fn test_auth_errors() {
        let auth_error = LlmError::AuthenticationError("Invalid key".to_string());
        assert!(auth_error.is_auth_error());

        let api_401 = LlmError::api_error(401, "Unauthorized");
        assert!(api_401.is_auth_error());
    }

    #[test]
    fn test_error_categories() {
        let auth_error = LlmError::AuthenticationError("Invalid key".to_string());
        assert_eq!(auth_error.category(), ErrorCategory::Authentication);

        let rate_limit = LlmError::RateLimitError("Too many requests".to_string());
        assert_eq!(rate_limit.category(), ErrorCategory::RateLimit);

        let server_error = LlmError::api_error(500, "Internal server error");
        assert_eq!(server_error.category(), ErrorCategory::Server);

        let client_error = LlmError::api_error(400, "Bad request");
        assert_eq!(client_error.category(), ErrorCategory::Client);

        let parse_error = LlmError::JsonError("Invalid JSON".to_string());
        assert_eq!(parse_error.category(), ErrorCategory::Parsing);
    }

    #[test]
    fn test_user_messages() {
        let auth_error = LlmError::AuthenticationError("Invalid key".to_string());
        let user_msg = auth_error.user_message();
        assert!(user_msg.contains("Authentication failed"));

        let rate_limit = LlmError::RateLimitError("Too many requests".to_string());
        let user_msg = rate_limit.user_message();
        assert!(user_msg.contains("Rate limit exceeded"));
    }

    #[test]
    fn test_recovery_suggestions() {
        let auth_error = LlmError::AuthenticationError("Invalid key".to_string());
        let suggestions = auth_error.recovery_suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("API key")));

        let rate_limit = LlmError::RateLimitError("Too many requests".to_string());
        let suggestions = rate_limit.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("Wait")));
    }
}

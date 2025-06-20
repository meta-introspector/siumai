//! Error handling module
//!
//! Defines all error types used in the LLM library.

use thiserror::Error;

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
}

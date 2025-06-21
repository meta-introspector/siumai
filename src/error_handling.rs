//! Enhanced Error Handling Module
//!
//! This module provides advanced error handling capabilities including
//! error classification, recovery strategies, and error reporting.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use crate::error::LlmError;
use crate::types::ProviderType;

/// Error classification for better handling
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Temporary errors that can be retried
    Transient,
    /// Authentication/authorization errors
    Authentication,
    /// Rate limiting errors
    RateLimit,
    /// Client-side errors (bad request, invalid parameters)
    ClientError,
    /// Server-side errors
    ServerError,
    /// Network connectivity issues
    Network,
    /// Configuration errors
    Configuration,
    /// Permanent errors that should not be retried
    Permanent,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - minor issues
    Low,
    /// Medium severity - notable issues
    Medium,
    /// High severity - significant issues
    High,
    /// Critical severity - system-threatening issues
    Critical,
}

/// Error context information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Provider that generated the error
    pub provider: Option<ProviderType>,
    /// Request ID if available
    pub request_id: Option<String>,
    /// Timestamp when error occurred
    pub timestamp: SystemTime,
    /// Additional context data
    pub metadata: HashMap<String, String>,
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            provider: None,
            request_id: None,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }
}

/// Enhanced error with classification and context
#[derive(Debug, Clone)]
pub struct ClassifiedError {
    /// The original error
    pub error: LlmError,
    /// Error category
    pub category: ErrorCategory,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Error context
    pub context: ErrorContext,
    /// Suggested recovery actions
    pub recovery_suggestions: Vec<RecoveryAction>,
}

/// Recovery action suggestions
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry the operation
    Retry { delay: Duration, max_attempts: u32 },
    /// Switch to a different provider
    SwitchProvider { suggested_provider: ProviderType },
    /// Reduce request complexity
    ReduceComplexity { suggestion: String },
    /// Check authentication
    CheckAuthentication,
    /// Wait for rate limit reset
    WaitForRateLimit { reset_time: Option<SystemTime> },
    /// Update configuration
    UpdateConfiguration {
        parameter: String,
        suggestion: String,
    },
    /// Contact support
    ContactSupport { message: String },
}

/// Error classifier that categorizes errors
pub struct ErrorClassifier;

impl ErrorClassifier {
    /// Classify an error with context
    pub fn classify(error: &LlmError, context: ErrorContext) -> ClassifiedError {
        let category = Self::categorize_error(error);
        let severity = Self::assess_severity(error, &category);
        let recovery_suggestions = Self::suggest_recovery(error, &category);

        ClassifiedError {
            error: error.clone(),
            category,
            severity,
            context,
            recovery_suggestions,
        }
    }

    /// Categorize an error
    fn categorize_error(error: &LlmError) -> ErrorCategory {
        match error {
            LlmError::HttpError(msg) => {
                if msg.contains("timeout") || msg.contains("connect") {
                    ErrorCategory::Network
                } else {
                    ErrorCategory::Transient
                }
            }
            LlmError::ApiError { code, .. } => match *code {
                401 | 403 => ErrorCategory::Authentication,
                429 => ErrorCategory::RateLimit,
                400 | 422 => ErrorCategory::ClientError,
                500..=599 => ErrorCategory::ServerError,
                _ => ErrorCategory::Transient,
            },
            LlmError::AuthenticationError(_) => ErrorCategory::Authentication,
            LlmError::RateLimitError(_) => ErrorCategory::RateLimit,
            LlmError::TimeoutError(_) => ErrorCategory::Network,
            LlmError::ConnectionError(_) => ErrorCategory::Network,
            LlmError::ConfigurationError(_) => ErrorCategory::Configuration,
            LlmError::InvalidParameter(_) => ErrorCategory::ClientError,
            LlmError::MissingApiKey(_) => ErrorCategory::Configuration,
            LlmError::ModelNotSupported(_) => ErrorCategory::ClientError,
            LlmError::UnsupportedOperation(_) => ErrorCategory::Permanent,
            _ => ErrorCategory::Transient,
        }
    }

    /// Assess error severity
    const fn assess_severity(error: &LlmError, category: &ErrorCategory) -> ErrorSeverity {
        match category {
            ErrorCategory::Authentication => ErrorSeverity::High,
            ErrorCategory::Configuration => ErrorSeverity::High,
            ErrorCategory::Permanent => ErrorSeverity::Medium,
            ErrorCategory::RateLimit => ErrorSeverity::Medium,
            ErrorCategory::ServerError => {
                if let LlmError::ApiError { code, .. } = error {
                    if *code >= 500 && *code < 510 {
                        ErrorSeverity::High
                    } else {
                        ErrorSeverity::Medium
                    }
                } else {
                    ErrorSeverity::Medium
                }
            }
            ErrorCategory::Network => ErrorSeverity::Medium,
            ErrorCategory::ClientError => ErrorSeverity::Low,
            ErrorCategory::Transient => ErrorSeverity::Low,
        }
    }

    /// Suggest recovery actions
    fn suggest_recovery(error: &LlmError, category: &ErrorCategory) -> Vec<RecoveryAction> {
        match category {
            ErrorCategory::Transient => vec![RecoveryAction::Retry {
                delay: Duration::from_millis(1000),
                max_attempts: 3,
            }],
            ErrorCategory::Authentication => vec![
                RecoveryAction::CheckAuthentication,
                RecoveryAction::UpdateConfiguration {
                    parameter: "api_key".to_string(),
                    suggestion: "Verify your API key is correct and has proper permissions"
                        .to_string(),
                },
            ],
            ErrorCategory::RateLimit => vec![
                RecoveryAction::WaitForRateLimit { reset_time: None },
                RecoveryAction::Retry {
                    delay: Duration::from_secs(60),
                    max_attempts: 2,
                },
            ],
            ErrorCategory::ClientError => {
                if let LlmError::InvalidParameter(param) = error {
                    vec![RecoveryAction::UpdateConfiguration {
                        parameter: "parameters".to_string(),
                        suggestion: format!("Check parameter: {param}"),
                    }]
                } else {
                    vec![RecoveryAction::ReduceComplexity {
                        suggestion: "Simplify your request or check input parameters".to_string(),
                    }]
                }
            }
            ErrorCategory::ServerError => vec![
                RecoveryAction::Retry {
                    delay: Duration::from_millis(2000),
                    max_attempts: 3,
                },
                RecoveryAction::SwitchProvider {
                    suggested_provider: ProviderType::OpenAi, // Default fallback
                },
            ],
            ErrorCategory::Network => vec![RecoveryAction::Retry {
                delay: Duration::from_millis(1500),
                max_attempts: 5,
            }],
            ErrorCategory::Configuration => vec![RecoveryAction::UpdateConfiguration {
                parameter: "configuration".to_string(),
                suggestion: "Check your configuration settings".to_string(),
            }],
            ErrorCategory::Permanent => vec![RecoveryAction::ContactSupport {
                message: "This operation is not supported".to_string(),
            }],
        }
    }
}

/// Error reporter for logging and monitoring
pub struct ErrorReporter {
    /// Whether to log errors
    pub enable_logging: bool,
    /// Error statistics
    pub stats: ErrorStats,
}

/// Error statistics tracking
#[derive(Debug, Clone, Default)]
pub struct ErrorStats {
    /// Total error count
    pub total_errors: u64,
    /// Errors by category
    pub errors_by_category: HashMap<String, u64>,
    /// Errors by provider
    pub errors_by_provider: HashMap<String, u64>,
    /// Errors by severity
    pub errors_by_severity: HashMap<String, u64>,
}

impl ErrorReporter {
    /// Create a new error reporter
    pub fn new() -> Self {
        Self {
            enable_logging: true,
            stats: ErrorStats::default(),
        }
    }

    /// Report a classified error
    pub fn report(&mut self, classified_error: &ClassifiedError) {
        self.update_stats(classified_error);

        if self.enable_logging {
            self.log_error(classified_error);
        }
    }

    /// Update error statistics
    fn update_stats(&mut self, classified_error: &ClassifiedError) {
        self.stats.total_errors += 1;

        // Update category stats
        let category_key = format!("{:?}", classified_error.category);
        *self
            .stats
            .errors_by_category
            .entry(category_key)
            .or_insert(0) += 1;

        // Update provider stats
        if let Some(provider) = &classified_error.context.provider {
            let provider_key = format!("{provider:?}");
            *self
                .stats
                .errors_by_provider
                .entry(provider_key)
                .or_insert(0) += 1;
        }

        // Update severity stats
        let severity_key = format!("{:?}", classified_error.severity);
        *self
            .stats
            .errors_by_severity
            .entry(severity_key)
            .or_insert(0) += 1;
    }

    /// Log an error
    fn log_error(&self, classified_error: &ClassifiedError) {
        match classified_error.severity {
            ErrorSeverity::Critical => {
                log::error!(
                    "CRITICAL ERROR [{:?}]: {} - Recovery: {:?}",
                    classified_error.category,
                    classified_error.error,
                    classified_error.recovery_suggestions
                );
            }
            ErrorSeverity::High => {
                log::error!(
                    "HIGH ERROR [{:?}]: {} - Recovery: {:?}",
                    classified_error.category,
                    classified_error.error,
                    classified_error.recovery_suggestions
                );
            }
            ErrorSeverity::Medium => {
                log::warn!(
                    "MEDIUM ERROR [{:?}]: {}",
                    classified_error.category,
                    classified_error.error
                );
            }
            ErrorSeverity::Low => {
                log::info!(
                    "LOW ERROR [{:?}]: {}",
                    classified_error.category,
                    classified_error.error
                );
            }
        }
    }

    /// Get error statistics
    pub const fn get_stats(&self) -> &ErrorStats {
        &self.stats
    }

    /// Reset error statistics
    pub fn reset_stats(&mut self) {
        self.stats = ErrorStats::default();
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to classify and report an error
pub fn handle_error(
    error: &LlmError,
    context: ErrorContext,
    reporter: &mut ErrorReporter,
) -> ClassifiedError {
    let classified = ErrorClassifier::classify(error, context);
    reporter.report(&classified);
    classified
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        let error = LlmError::ApiError {
            code: 429,
            message: "Rate limit exceeded".to_string(),
            details: None,
        };

        let context = ErrorContext::default();
        let classified = ErrorClassifier::classify(&error, context);

        assert_eq!(classified.category, ErrorCategory::RateLimit);
        assert_eq!(classified.severity, ErrorSeverity::Medium);
        assert!(!classified.recovery_suggestions.is_empty());
    }

    #[test]
    fn test_error_reporter() {
        let mut reporter = ErrorReporter::new();

        let error = LlmError::AuthenticationError("Invalid API key".to_string());
        let context = ErrorContext::default();
        let classified = ErrorClassifier::classify(&error, context);

        reporter.report(&classified);

        assert_eq!(reporter.stats.total_errors, 1);
        assert_eq!(
            reporter.stats.errors_by_category.get("Authentication"),
            Some(&1)
        );
    }

    #[test]
    fn test_recovery_suggestions() {
        let error = LlmError::RateLimitError("Too many requests".to_string());
        let category = ErrorCategory::RateLimit;
        let suggestions = ErrorClassifier::suggest_recovery(&error, &category);

        assert!(!suggestions.is_empty());
        assert!(matches!(
            suggestions[0],
            RecoveryAction::WaitForRateLimit { .. }
        ));
    }
}

//! Parameter Management Module
//!
//! Handles parameter mapping, validation, and provider-specific configurations.
//! This module provides a comprehensive parameter system that supports:
//! - Cross-provider parameter mapping
//! - Enhanced validation with detailed error reporting
//! - Parameter optimization for specific providers
//! - Compatibility checking between providers

pub mod common;
pub mod mapper;
pub mod openai;
pub mod anthropic;
pub mod gemini;
pub mod validator;

// Re-export main types and traits
pub use common::*;
pub use mapper::*;
pub use openai::*;
pub use anthropic::*;
pub use gemini::*;
pub use validator::*;

// Re-export for backward compatibility
pub use mapper::{ParameterMapper, ParameterMapperFactory};
pub use openai::OpenAiParameterMapper;
pub use anthropic::AnthropicParameterMapper;
pub use gemini::GeminiParameterMapper;

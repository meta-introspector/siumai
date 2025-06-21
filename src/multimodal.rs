//! Enhanced Multimodal Support
//!
//! This module provides comprehensive multimodal capabilities including
//! image processing, audio handling, document processing, and format conversion.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::types::{ContentPart, MessageContent};

/// Multimodal content processor
#[allow(dead_code)]
pub struct MultimodalProcessor {
    /// Supported image formats
    image_formats: Vec<ImageFormat>,
    /// Supported audio formats
    audio_formats: Vec<AudioFormat>,
    /// Supported document formats
    document_formats: Vec<DocumentFormat>,
    /// Processing configuration
    config: ProcessingConfig,
}

impl MultimodalProcessor {
    /// Create a new multimodal processor
    pub fn new() -> Self {
        Self {
            image_formats: ImageFormat::all_supported(),
            audio_formats: AudioFormat::all_supported(),
            document_formats: DocumentFormat::all_supported(),
            config: ProcessingConfig::default(),
        }
    }

    /// Process multimodal content
    pub fn process_content(&self, content: &MessageContent) -> Result<ProcessedContent, LlmError> {
        match content {
            MessageContent::Text(text) => Ok(ProcessedContent::Text(text.clone())),
            MessageContent::MultiModal(parts) => {
                let mut processed_parts = Vec::new();

                for part in parts {
                    let processed_part = self.process_content_part(part)?;
                    processed_parts.push(processed_part);
                }

                Ok(ProcessedContent::MultiModal(processed_parts))
            }
        }
    }

    /// Process a single content part
    fn process_content_part(&self, part: &ContentPart) -> Result<ProcessedContentPart, LlmError> {
        match part {
            ContentPart::Text { text } => Ok(ProcessedContentPart::Text {
                text: text.clone(),
                metadata: ContentMetadata::default(),
            }),
            ContentPart::Image { image_url, detail } => {
                let image_info = self.analyze_image(image_url)?;
                let format = match image_info.format {
                    MediaFormat::Image(fmt) => fmt,
                    _ => ImageFormat::Jpeg, // Fallback
                };
                Ok(ProcessedContentPart::Image {
                    data: image_url.clone(),
                    format,
                    detail: detail.clone(),
                    metadata: image_info.metadata,
                })
            }
            ContentPart::Audio { audio_url, format } => {
                let audio_info = self.analyze_audio(audio_url, Some(format.as_str()))?;
                let format = match audio_info.format {
                    MediaFormat::Audio(fmt) => fmt,
                    _ => AudioFormat::Wav, // Fallback
                };
                Ok(ProcessedContentPart::Audio {
                    data: audio_url.clone(),
                    format,
                    metadata: audio_info.metadata,
                })
            }
        }
    }

    /// Analyze image content
    fn analyze_image(&self, image_data: &str) -> Result<MediaInfo, LlmError> {
        // Detect format from data URL or base64 header
        let format = if image_data.starts_with("data:image/") {
            let mime_type = image_data
                .split(';')
                .next()
                .and_then(|s| s.strip_prefix("data:"))
                .unwrap_or("image/jpeg");

            ImageFormat::from_mime_type(mime_type)
        } else {
            // Try to detect from base64 header
            ImageFormat::detect_from_base64(image_data)
        };

        let mut metadata = ContentMetadata::default();
        metadata.insert(
            "original_format".to_string(),
            serde_json::Value::String(format.to_string()),
        );

        // Add size estimation if possible
        if let Ok(size) = self.estimate_data_size(image_data) {
            metadata.insert(
                "estimated_size_bytes".to_string(),
                serde_json::Value::Number(size.into()),
            );
        }

        Ok(MediaInfo {
            format: MediaFormat::Image(format),
            metadata,
        })
    }

    /// Analyze audio content
    fn analyze_audio(
        &self,
        audio_data: &str,
        format_hint: Option<&str>,
    ) -> Result<MediaInfo, LlmError> {
        let format = if let Some(hint) = format_hint {
            AudioFormat::from_extension(hint)
        } else if audio_data.starts_with("data:audio/") {
            let mime_type = audio_data
                .split(';')
                .next()
                .and_then(|s| s.strip_prefix("data:"))
                .unwrap_or("audio/wav");

            AudioFormat::from_mime_type(mime_type)
        } else {
            AudioFormat::Wav // Default
        };

        let mut metadata = ContentMetadata::default();
        metadata.insert(
            "original_format".to_string(),
            serde_json::Value::String(format.to_string()),
        );

        if let Ok(size) = self.estimate_data_size(audio_data) {
            metadata.insert(
                "estimated_size_bytes".to_string(),
                serde_json::Value::Number(size.into()),
            );
        }

        Ok(MediaInfo {
            format: MediaFormat::Audio(format),
            metadata,
        })
    }

    /// Estimate data size from base64 string
    fn estimate_data_size(&self, data: &str) -> Result<u64, LlmError> {
        // Remove data URL prefix if present
        let base64_data = if data.contains(',') {
            data.split(',').nth(1).unwrap_or(data)
        } else {
            data
        };

        // Estimate size: base64 is ~4/3 the size of original data
        let base64_len = base64_data.len() as u64;
        Ok((base64_len * 3) / 4)
    }
}

impl Default for MultimodalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Maximum image size in bytes
    pub max_image_size: Option<u64>,
    /// Maximum audio duration in seconds
    pub max_audio_duration: Option<u32>,
    /// Whether to compress large files
    pub auto_compress: bool,
    /// Target compression quality (0.0-1.0)
    pub compression_quality: f32,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            max_image_size: Some(20 * 1024 * 1024), // 20MB
            max_audio_duration: Some(300),          // 5 minutes
            auto_compress: true,
            compression_quality: 0.8,
        }
    }
}

/// Processed content
#[derive(Debug, Clone)]
pub enum ProcessedContent {
    /// Text content
    Text(String),
    /// Multimodal content
    MultiModal(Vec<ProcessedContentPart>),
}

/// Processed content part
#[derive(Debug, Clone)]
pub enum ProcessedContentPart {
    /// Text part
    Text {
        text: String,
        metadata: ContentMetadata,
    },
    /// Image part
    Image {
        data: String,
        format: ImageFormat,
        detail: Option<String>,
        metadata: ContentMetadata,
    },
    /// Audio part
    Audio {
        data: String,
        format: AudioFormat,
        metadata: ContentMetadata,
    },
    /// Document part
    Document {
        data: String,
        format: DocumentFormat,
        metadata: ContentMetadata,
    },
}

/// Content metadata
pub type ContentMetadata = HashMap<String, serde_json::Value>;

/// Media information
#[derive(Debug, Clone)]
pub struct MediaInfo {
    /// Media format
    pub format: MediaFormat,
    /// Metadata
    pub metadata: ContentMetadata,
}

/// Media format enumeration
#[derive(Debug, Clone)]
pub enum MediaFormat {
    /// Image format
    Image(ImageFormat),
    /// Audio format
    Audio(AudioFormat),
    /// Document format
    Document(DocumentFormat),
}

/// Supported image formats
#[derive(Debug, Clone, PartialEq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Gif,
    WebP,
    Bmp,
    Tiff,
    Svg,
}

impl ImageFormat {
    /// Get all supported image formats
    pub fn all_supported() -> Vec<Self> {
        vec![
            Self::Jpeg,
            Self::Png,
            Self::Gif,
            Self::WebP,
            Self::Bmp,
            Self::Tiff,
            Self::Svg,
        ]
    }

    /// Get format from MIME type
    pub fn from_mime_type(mime_type: &str) -> Self {
        match mime_type {
            "image/jpeg" | "image/jpg" => Self::Jpeg,
            "image/png" => Self::Png,
            "image/gif" => Self::Gif,
            "image/webp" => Self::WebP,
            "image/bmp" => Self::Bmp,
            "image/tiff" => Self::Tiff,
            "image/svg+xml" => Self::Svg,
            _ => Self::Jpeg, // Default
        }
    }

    /// Detect format from base64 data
    pub fn detect_from_base64(data: &str) -> Self {
        // Simple magic number detection
        if data.starts_with("/9j/") || data.starts_with("iVBOR") {
            Self::Jpeg
        } else if data.starts_with("iVBOR") {
            Self::Png
        } else if data.starts_with("R0lGOD") {
            Self::Gif
        } else if data.starts_with("UklGR") {
            Self::WebP
        } else {
            Self::Jpeg // Default
        }
    }

    /// Get MIME type
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::WebP => "image/webp",
            Self::Bmp => "image/bmp",
            Self::Tiff => "image/tiff",
            Self::Svg => "image/svg+xml",
        }
    }
}

impl std::fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Supported audio formats
#[derive(Debug, Clone, PartialEq)]
pub enum AudioFormat {
    Mp3,
    Wav,
    Flac,
    Ogg,
    M4a,
    Webm,
}

impl AudioFormat {
    /// Get all supported audio formats
    pub fn all_supported() -> Vec<Self> {
        vec![
            Self::Mp3,
            Self::Wav,
            Self::Flac,
            Self::Ogg,
            Self::M4a,
            Self::Webm,
        ]
    }

    /// Get format from MIME type
    pub fn from_mime_type(mime_type: &str) -> Self {
        match mime_type {
            "audio/mpeg" | "audio/mp3" => Self::Mp3,
            "audio/wav" | "audio/wave" => Self::Wav,
            "audio/flac" => Self::Flac,
            "audio/ogg" => Self::Ogg,
            "audio/m4a" => Self::M4a,
            "audio/webm" => Self::Webm,
            _ => Self::Wav, // Default
        }
    }

    /// Get format from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "mp3" => Self::Mp3,
            "wav" => Self::Wav,
            "flac" => Self::Flac,
            "ogg" => Self::Ogg,
            "m4a" => Self::M4a,
            "webm" => Self::Webm,
            _ => Self::Wav, // Default
        }
    }

    /// Get MIME type
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Wav => "audio/wav",
            Self::Flac => "audio/flac",
            Self::Ogg => "audio/ogg",
            Self::M4a => "audio/m4a",
            Self::Webm => "audio/webm",
        }
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Supported document formats
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentFormat {
    Pdf,
    Docx,
    Txt,
    Md,
    Html,
    Csv,
    Json,
    Xml,
}

impl DocumentFormat {
    /// Get all supported document formats
    pub fn all_supported() -> Vec<Self> {
        vec![
            Self::Pdf,
            Self::Docx,
            Self::Txt,
            Self::Md,
            Self::Html,
            Self::Csv,
            Self::Json,
            Self::Xml,
        ]
    }

    /// Get format from MIME type
    pub fn from_mime_type(mime_type: &str) -> Self {
        match mime_type {
            "application/pdf" => Self::Pdf,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Self::Docx,
            "text/plain" => Self::Txt,
            "text/markdown" => Self::Md,
            "text/html" => Self::Html,
            "text/csv" => Self::Csv,
            "application/json" => Self::Json,
            "application/xml" | "text/xml" => Self::Xml,
            _ => Self::Txt, // Default
        }
    }

    /// Get MIME type
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Pdf => "application/pdf",
            Self::Docx => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Self::Txt => "text/plain",
            Self::Md => "text/markdown",
            Self::Html => "text/html",
            Self::Csv => "text/csv",
            Self::Json => "application/json",
            Self::Xml => "application/xml",
        }
    }
}

impl std::fmt::Display for DocumentFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_detection() {
        assert_eq!(ImageFormat::from_mime_type("image/jpeg"), ImageFormat::Jpeg);
        assert_eq!(ImageFormat::from_mime_type("image/png"), ImageFormat::Png);
        assert_eq!(ImageFormat::from_mime_type("image/gif"), ImageFormat::Gif);
    }

    #[test]
    fn test_audio_format_detection() {
        assert_eq!(AudioFormat::from_mime_type("audio/mpeg"), AudioFormat::Mp3);
        assert_eq!(AudioFormat::from_extension("wav"), AudioFormat::Wav);
        assert_eq!(AudioFormat::from_extension("flac"), AudioFormat::Flac);
    }

    #[test]
    fn test_multimodal_processor() {
        let processor = MultimodalProcessor::new();

        let text_content = MessageContent::Text("Hello world".to_string());
        let processed = processor.process_content(&text_content).unwrap();

        match processed {
            ProcessedContent::Text(text) => assert_eq!(text, "Hello world"),
            _ => panic!("Expected text content"),
        }
    }
}

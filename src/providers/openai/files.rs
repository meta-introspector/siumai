//! OpenAI Files API Implementation
//!
//! This module provides the OpenAI implementation of the FileManagementCapability trait,
//! including file upload, listing, retrieval, and deletion operations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::FileManagementCapability;
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};

use super::config::OpenAiConfig;

/// OpenAI file upload API request structure
#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
struct OpenAiFileUploadForm {
    /// File purpose (e.g., "assistants", "fine-tune", "batch")
    purpose: String,
}

/// OpenAI file API response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiFileResponse {
    /// File ID
    id: String,
    /// Object type (should be "file")
    object: String,
    /// File size in bytes
    bytes: u64,
    /// Creation timestamp
    created_at: u64,
    /// Original filename
    filename: String,
    /// File purpose
    purpose: String,
    /// File status
    status: String,
    /// Status details (if any)
    status_details: Option<String>,
}

/// OpenAI file list API response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiFileListResponse {
    /// Object type (should be "list")
    #[allow(dead_code)]
    object: String,
    /// List of files
    data: Vec<OpenAiFileResponse>,
    /// Whether there are more results
    has_more: Option<bool>,
}

/// OpenAI file deletion API response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiFileDeleteResponse {
    /// File ID that was deleted
    id: String,
    /// Object type (should be "file")
    #[allow(dead_code)]
    object: String,
    /// Whether deletion was successful
    deleted: bool,
}

/// OpenAI file management capability implementation.
///
/// This struct provides the OpenAI-specific implementation of file management
/// operations using the OpenAI Files API.
///
/// # Supported Operations
/// - File upload with various purposes (assistants, fine-tune, batch, etc.)
/// - File listing with filtering and pagination
/// - File metadata retrieval
/// - File deletion
/// - File content download
///
/// # API Reference
/// https://platform.openai.com/docs/api-reference/files
#[derive(Debug, Clone)]
pub struct OpenAiFiles {
    /// OpenAI configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl OpenAiFiles {
    /// Create a new OpenAI files instance.
    ///
    /// # Arguments
    /// * `config` - OpenAI configuration
    /// * `http_client` - HTTP client for making requests
    pub fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Get supported file purposes.
    pub fn get_supported_purposes(&self) -> Vec<String> {
        vec![
            "assistants".to_string(),
            "batch".to_string(),
            "fine-tune".to_string(),
            "vision".to_string(),
        ]
    }

    /// Get maximum file size in bytes.
    pub fn get_max_file_size(&self) -> u64 {
        512 * 1024 * 1024 // 512 MB
    }

    /// Get supported file formats.
    pub fn get_supported_formats(&self) -> Vec<String> {
        vec![
            // Text formats
            "txt".to_string(),
            "json".to_string(),
            "jsonl".to_string(),
            "csv".to_string(),
            "tsv".to_string(),
            // Document formats
            "pdf".to_string(),
            "docx".to_string(),
            // Image formats
            "png".to_string(),
            "jpg".to_string(),
            "jpeg".to_string(),
            "gif".to_string(),
            "webp".to_string(),
            // Audio formats
            "mp3".to_string(),
            "mp4".to_string(),
            "mpeg".to_string(),
            "mpga".to_string(),
            "m4a".to_string(),
            "wav".to_string(),
            "webm".to_string(),
        ]
    }

    /// Validate file upload request.
    fn validate_upload_request(&self, request: &FileUploadRequest) -> Result<(), LlmError> {
        // Validate file size
        if request.content.len() as u64 > self.get_max_file_size() {
            return Err(LlmError::InvalidInput(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                request.content.len(),
                self.get_max_file_size()
            )));
        }

        // Validate purpose
        if !self.get_supported_purposes().contains(&request.purpose) {
            return Err(LlmError::InvalidInput(format!(
                "Unsupported file purpose: {}. Supported purposes: {:?}",
                request.purpose,
                self.get_supported_purposes()
            )));
        }

        // Validate filename
        if request.filename.is_empty() {
            return Err(LlmError::InvalidInput(
                "Filename cannot be empty".to_string(),
            ));
        }

        // Validate file extension if provided
        if let Some(extension) = request.filename.split('.').last() {
            let supported_formats = self.get_supported_formats();
            if !supported_formats.contains(&extension.to_lowercase()) {
                return Err(LlmError::InvalidInput(format!(
                    "Unsupported file format: {}. Supported formats: {:?}",
                    extension, supported_formats
                )));
            }
        }

        Ok(())
    }

    /// Convert OpenAI file response to our standard format.
    fn convert_file_response(&self, openai_file: OpenAiFileResponse) -> FileObject {
        let mut metadata = HashMap::new();
        metadata.insert(
            "object".to_string(),
            serde_json::Value::String(openai_file.object),
        );
        metadata.insert(
            "status".to_string(),
            serde_json::Value::String(openai_file.status),
        );

        if let Some(status_details) = openai_file.status_details {
            metadata.insert(
                "status_details".to_string(),
                serde_json::Value::String(status_details),
            );
        }

        FileObject {
            id: openai_file.id,
            filename: openai_file.filename,
            bytes: openai_file.bytes,
            created_at: openai_file.created_at,
            purpose: openai_file.purpose,
            status: "uploaded".to_string(), // Simplified status
            mime_type: None,                // OpenAI doesn't provide MIME type in response
            metadata,
        }
    }

    /// Make HTTP request with proper headers.
    async fn make_request(
        &self,
        method: reqwest::Method,
        endpoint: &str,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        let url = format!("{}/{}", self.config.base_url, endpoint);

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {}", e)))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {}", e)))?;
            headers.insert(header_name, header_value);
        }

        Ok(self.http_client.request(method, &url).headers(headers))
    }

    /// Handle API response errors.
    async fn handle_response_error(&self, response: reqwest::Response) -> LlmError {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());

        match status.as_u16() {
            404 => LlmError::NotFound(format!("File not found: {}", error_text)),
            413 => LlmError::InvalidInput("File too large".to_string()),
            415 => LlmError::InvalidInput("Unsupported file type".to_string()),
            _ => LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI Files API error {}: {}", status, error_text),
                details: None,
            },
        }
    }
}

#[async_trait]
impl FileManagementCapability for OpenAiFiles {
    /// Upload a file to OpenAI's storage.
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        // Validate request
        self.validate_upload_request(&request)?;

        // Create multipart form
        let form = reqwest::multipart::Form::new()
            .text("purpose", request.purpose.clone())
            .part(
                "file",
                reqwest::multipart::Part::bytes(request.content)
                    .file_name(request.filename.clone())
                    .mime_str(
                        request
                            .mime_type
                            .as_deref()
                            .unwrap_or("application/octet-stream"),
                    )
                    .map_err(|e| LlmError::HttpError(format!("Invalid MIME type: {}", e)))?,
            );

        let request_builder = self.make_request(reqwest::Method::POST, "files").await?;
        let response = request_builder
            .multipart(form)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let openai_response: OpenAiFileResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        Ok(self.convert_file_response(openai_response))
    }

    /// List files with optional filtering.
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let mut endpoint = "files".to_string();

        // Build query parameters
        if let Some(q) = query {
            let mut params = Vec::new();

            if let Some(purpose) = q.purpose {
                params.push(format!("purpose={}", urlencoding::encode(&purpose)));
            }
            if let Some(limit) = q.limit {
                params.push(format!("limit={}", limit));
            }
            if let Some(after) = q.after {
                params.push(format!("after={}", urlencoding::encode(&after)));
            }
            if let Some(order) = q.order {
                params.push(format!("order={}", urlencoding::encode(&order)));
            }

            if !params.is_empty() {
                endpoint.push('?');
                endpoint.push_str(&params.join("&"));
            }
        }

        let request_builder = self.make_request(reqwest::Method::GET, &endpoint).await?;
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let openai_response: OpenAiFileListResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        let files: Vec<FileObject> = openai_response
            .data
            .into_iter()
            .map(|f| self.convert_file_response(f))
            .collect();

        Ok(FileListResponse {
            files,
            has_more: openai_response.has_more.unwrap_or(false),
            next_cursor: None, // OpenAI uses different pagination
        })
    }

    /// Retrieve file metadata.
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        let endpoint = format!("files/{}", file_id);

        let request_builder = self.make_request(reqwest::Method::GET, &endpoint).await?;
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let openai_response: OpenAiFileResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        Ok(self.convert_file_response(openai_response))
    }

    /// Delete a file permanently.
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let endpoint = format!("files/{}", file_id);

        let request_builder = self
            .make_request(reqwest::Method::DELETE, &endpoint)
            .await?;
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let openai_response: OpenAiFileDeleteResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        Ok(FileDeleteResponse {
            id: openai_response.id,
            deleted: openai_response.deleted,
        })
    }

    /// Get file content as bytes.
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let endpoint = format!("files/{}/content", file_id);

        let request_builder = self.make_request(reqwest::Method::GET, &endpoint).await?;
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let content = response
            .bytes()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to read response body: {}", e)))?;

        Ok(content.to_vec())
    }
}

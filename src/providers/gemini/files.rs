//! Gemini Files API Implementation
//!
//! This module provides the Gemini implementation of the `FileManagementCapability` trait,
//! including file upload, listing, retrieval, and deletion operations.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::json;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::FileManagementCapability;
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};

use super::types::{
    CreateFileRequest, CreateFileResponse, GeminiConfig, GeminiFile, GeminiFileState,
    ListFilesResponse,
};

/// Gemini file management capability implementation.
///
/// This struct provides the Gemini-specific implementation of file management
/// operations using the Gemini Files API.
///
/// # Supported Operations
/// - File upload with metadata
/// - File listing with pagination
/// - File metadata retrieval
/// - File deletion
/// - File content download
///
/// # API Reference
/// <https://ai.google.dev/api/files>
#[derive(Debug, Clone)]
pub struct GeminiFiles {
    /// Gemini configuration
    config: GeminiConfig,
    /// HTTP client
    http_client: HttpClient,
}

impl GeminiFiles {
    /// Create a new Gemini files capability
    pub const fn new(config: GeminiConfig, http_client: HttpClient) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Convert `GeminiFile` to `FileObject`
    fn convert_gemini_file_to_file_object(&self, gemini_file: GeminiFile) -> FileObject {
        // Extract file ID from the full name (e.g., "files/abc123" -> "abc123")
        let id = gemini_file
            .name
            .as_ref()
            .and_then(|name| name.strip_prefix("files/"))
            .unwrap_or("")
            .to_string();

        // Parse size from string to u64
        let bytes = gemini_file
            .size_bytes
            .as_ref()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        // Parse creation timestamp
        let created_at = gemini_file
            .create_time
            .as_ref()
            .and_then(|time| chrono::DateTime::parse_from_rfc3339(time).ok())
            .map(|dt| dt.timestamp() as u64)
            .unwrap_or(0);

        // Determine status
        let status = match gemini_file.state {
            Some(GeminiFileState::Active) => "active".to_string(),
            Some(GeminiFileState::Processing) => "processing".to_string(),
            Some(GeminiFileState::Failed) => "failed".to_string(),
            _ => "unknown".to_string(),
        };

        // Extract filename from display_name or use ID as fallback
        let filename = gemini_file
            .display_name
            .unwrap_or_else(|| format!("file_{id}"));

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), json!("gemini"));
        if let Some(uri) = &gemini_file.uri {
            metadata.insert("uri".to_string(), json!(uri));
        }
        if let Some(hash) = &gemini_file.sha256_hash {
            metadata.insert("sha256_hash".to_string(), json!(hash));
        }
        if let Some(expiration) = &gemini_file.expiration_time {
            metadata.insert("expiration_time".to_string(), json!(expiration));
        }

        FileObject {
            id,
            filename,
            bytes,
            created_at,
            purpose: "general".to_string(), // Gemini doesn't have explicit purposes
            status,
            mime_type: gemini_file.mime_type,
            metadata,
        }
    }

    /// Convert `FileUploadRequest` to `CreateFileRequest`
    #[allow(dead_code)]
    fn convert_upload_request_to_gemini(&self, request: &FileUploadRequest) -> CreateFileRequest {
        let gemini_file = GeminiFile {
            name: None, // Will be auto-generated
            display_name: Some(request.filename.clone()),
            mime_type: request.mime_type.clone(),
            size_bytes: None, // Will be set by the API
            create_time: None,
            update_time: None,
            expiration_time: None,
            sha256_hash: None,
            uri: None,
            state: None,
            error: None,
            video_metadata: None,
        };

        CreateFileRequest {
            file: Some(gemini_file),
        }
    }

    /// Make a request to the Gemini API
    async fn make_request(
        &self,
        method: reqwest::Method,
        endpoint: &str,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        let url = format!("{}/{}", self.config.base_url, endpoint);

        let request_builder = self
            .http_client
            .request(method, &url)
            .header("x-goog-api-key", &self.config.api_key)
            .header("Content-Type", "application/json");

        Ok(request_builder)
    }

    /// Handle API response errors
    async fn handle_response_error(&self, response: reqwest::Response) -> LlmError {
        let status_code = response.status().as_u16();
        let error_text = response.text().await.unwrap_or_default();

        LlmError::api_error(
            status_code,
            format!("Gemini API error: {status_code} - {error_text}"),
        )
    }

    /// Validate file upload request
    fn validate_upload_request(&self, request: &FileUploadRequest) -> Result<(), LlmError> {
        if request.content.is_empty() {
            return Err(LlmError::InvalidInput("File content cannot be empty".to_string()));
        }

        if request.filename.is_empty() {
            return Err(LlmError::InvalidInput("Filename cannot be empty".to_string()));
        }

        // Check file size limits (Gemini has specific limits)
        const MAX_FILE_SIZE: usize = 20 * 1024 * 1024; // 20MB for most files
        if request.content.len() > MAX_FILE_SIZE {
            return Err(LlmError::InvalidInput(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                request.content.len(),
                MAX_FILE_SIZE
            )));
        }

        Ok(())
    }
}

#[async_trait]
impl FileManagementCapability for GeminiFiles {
    /// Upload a file to Gemini's storage.
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        // Validate request
        self.validate_upload_request(&request)?;

        // Note: Gemini's file upload is typically done via multipart/form-data
        // but the exact implementation may vary. For now, we'll implement a basic version.
        
        // Create multipart form
        let form = reqwest::multipart::Form::new()
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
                    .map_err(|e| LlmError::HttpError(format!("Invalid MIME type: {e}")))?,
            );

        // Add metadata if provided
        let mut form = form;
        if let Some(display_name) = request.metadata.get("display_name") {
            form = form.text("display_name", display_name.clone());
        }

        let url = format!("{}/files", self.config.base_url);
        let response = self
            .http_client
            .post(&url)
            .header("x-goog-api-key", &self.config.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let create_response: CreateFileResponse = response.json().await.map_err(|e| {
            LlmError::ParseError(format!("Failed to parse upload response: {e}"))
        })?;

        let gemini_file = create_response
            .file
            .ok_or_else(|| LlmError::ParseError("No file in upload response".to_string()))?;

        Ok(self.convert_gemini_file_to_file_object(gemini_file))
    }

    /// List files with optional filtering.
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let mut endpoint = "files".to_string();
        
        // Add query parameters
        let mut params = Vec::new();
        if let Some(q) = &query {
            if let Some(limit) = q.limit {
                params.push(format!("pageSize={limit}"));
            }
            if let Some(after) = &q.after {
                params.push(format!("pageToken={after}"));
            }
        }

        if !params.is_empty() {
            endpoint.push('?');
            endpoint.push_str(&params.join("&"));
        }

        let request_builder = self.make_request(reqwest::Method::GET, &endpoint).await?;
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let list_response: ListFilesResponse = response.json().await.map_err(|e| {
            LlmError::ParseError(format!("Failed to parse list response: {e}"))
        })?;

        let files: Vec<FileObject> = list_response
            .files
            .into_iter()
            .map(|f| self.convert_gemini_file_to_file_object(f))
            .collect();

        Ok(FileListResponse {
            files,
            has_more: list_response.next_page_token.is_some(),
            next_cursor: list_response.next_page_token,
        })
    }

    /// Retrieve file metadata.
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        // Ensure the file ID has the proper prefix
        let full_file_name = if file_id.starts_with("files/") {
            file_id
        } else {
            format!("files/{file_id}")
        };

        let endpoint = &full_file_name;
        let request_builder = self.make_request(reqwest::Method::GET, endpoint).await?;
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let gemini_file: GeminiFile = response.json().await.map_err(|e| {
            LlmError::ParseError(format!("Failed to parse file response: {e}"))
        })?;

        Ok(self.convert_gemini_file_to_file_object(gemini_file))
    }

    /// Delete a file permanently.
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        // Ensure the file ID has the proper prefix
        let full_file_name = if file_id.starts_with("files/") {
            file_id.clone()
        } else {
            format!("files/{file_id}")
        };

        let endpoint = &full_file_name;
        let request_builder = self.make_request(reqwest::Method::DELETE, endpoint).await?;
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        // Extract file ID without prefix for response
        let clean_file_id = full_file_name
            .strip_prefix("files/")
            .unwrap_or(&full_file_name)
            .to_string();

        Ok(FileDeleteResponse {
            id: clean_file_id,
            deleted: true,
        })
    }

    /// Get file content as bytes.
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        // Delegate to the internal implementation
        self.download_file_content(file_id).await
    }
}

impl GeminiFiles {
    /// Download file content as bytes.
    ///
    /// Note: This downloads the file content from Gemini's storage.
    pub async fn download_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        // Ensure the file ID has the proper prefix
        let full_file_name = if file_id.starts_with("files/") {
            file_id
        } else {
            format!("files/{file_id}")
        };

        // First get the file metadata to get the download URI
        let file_metadata = self.retrieve_file(full_file_name.clone()).await?;

        // Check if we have a download URI in metadata
        let download_uri = file_metadata
            .metadata
            .get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                LlmError::UnsupportedOperation("File download URI not available".to_string())
            })?;

        // Download the file content
        let response = self
            .http_client
            .get(download_uri)
            .header("x-goog-api-key", &self.config.api_key)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Download request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let content = response
            .bytes()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to read response body: {e}")))?;

        Ok(content.to_vec())
    }

    /// Get file content as string.
    pub async fn get_file_content_as_string(&self, file_id: String) -> Result<String, LlmError> {
        let bytes = self.download_file_content(file_id).await?;
        String::from_utf8(bytes)
            .map_err(|e| LlmError::ParseError(format!("File content is not valid UTF-8: {e}")))
    }

    /// Check if a file exists.
    pub async fn file_exists(&self, file_id: String) -> bool {
        self.retrieve_file(file_id).await.is_ok()
    }

    /// Wait for file processing to complete.
    ///
    /// This method polls the file status until it's either active or failed.
    pub async fn wait_for_file_processing(
        &self,
        file_id: String,
        max_wait_seconds: u64,
    ) -> Result<FileObject, LlmError> {
        let start_time = std::time::Instant::now();
        let max_duration = std::time::Duration::from_secs(max_wait_seconds);

        loop {
            let file = self.retrieve_file(file_id.clone()).await?;

            match file.status.as_str() {
                "active" => return Ok(file),
                "failed" => {
                    return Err(LlmError::ProcessingError(
                        "File processing failed".to_string(),
                    ))
                }
                "processing" => {
                    // Continue waiting
                    if start_time.elapsed() >= max_duration {
                        return Err(LlmError::TimeoutError(format!(
                            "File processing timeout after {max_wait_seconds} seconds"
                        )));
                    }

                    // Wait before next check
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
                _ => {
                    return Err(LlmError::ProcessingError(format!(
                        "Unknown file status: {}",
                        file.status
                    )))
                }
            }
        }
    }
}

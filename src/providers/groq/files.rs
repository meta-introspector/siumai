//! `Groq` Files Capability Implementation
//!
//! Implements file management capabilities for Groq.

use reqwest::multipart::{Form, Part};

use crate::error::LlmError;
use crate::types::HttpConfig;

// File response type for Groq
#[derive(Debug, Clone)]
pub struct FileResponse {
    pub id: String,
    pub filename: String,
    pub size: u64,
    pub purpose: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub status: Option<String>,
    pub status_details: Option<String>,
}

use super::types::*;
use super::utils::*;

/// `Groq` Files Capability Implementation
pub struct GroqFiles {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
}

impl GroqFiles {
    /// Create a new `Groq` files capability instance
    pub const fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
        }
    }

    /// Convert Groq file to our FileResponse
    #[allow(dead_code)]
    fn convert_groq_file(&self, groq_file: GroqFile) -> FileResponse {
        FileResponse {
            id: groq_file.id,
            filename: groq_file.filename,
            size: groq_file.bytes,
            purpose: groq_file.purpose,
            created_at: chrono::DateTime::from_timestamp(groq_file.created_at as i64, 0)
                .unwrap_or_else(chrono::Utc::now),
            status: None, // Groq doesn't provide status
            status_details: None,
        }
    }
}

#[allow(dead_code)]
impl GroqFiles {
    async fn upload_file(
        &self,
        file_data: Vec<u8>,
        filename: String,
        purpose: String,
    ) -> Result<FileResponse, LlmError> {
        let url = format!("{}/files", self.base_url);

        let form = Form::new()
            .part("file", Part::bytes(file_data).file_name(filename.clone()))
            .text("purpose", purpose);

        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq file upload error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_file: GroqFile = response.json().await?;
        Ok(self.convert_groq_file(groq_file))
    }

    async fn list_files(&self) -> Result<Vec<FileResponse>, LlmError> {
        let url = format!("{}/files", self.base_url);
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq list files error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_response: GroqFilesResponse = response.json().await?;
        let files = groq_response
            .data
            .into_iter()
            .map(|f| self.convert_groq_file(f))
            .collect();

        Ok(files)
    }

    async fn get_file(&self, file_id: String) -> Result<FileResponse, LlmError> {
        let url = format!("{}/files/{}", self.base_url, file_id);
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq get file error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_file: GroqFile = response.json().await?;
        Ok(self.convert_groq_file(groq_file))
    }

    async fn delete_file(&self, file_id: String) -> Result<bool, LlmError> {
        let url = format!("{}/files/{}", self.base_url, file_id);
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self
            .http_client
            .delete(&url)
            .headers(headers)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq delete file error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let delete_response: GroqDeleteFileResponse = response.json().await?;
        Ok(delete_response.deleted)
    }

    async fn download_file(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let url = format!("{}/files/{}/content", self.base_url, file_id);
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq download file error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let file_data = response.bytes().await?;
        Ok(file_data.to_vec())
    }

    fn supports_file_upload(&self) -> bool {
        true
    }

    fn max_file_size(&self) -> Option<usize> {
        Some(100 * 1024 * 1024) // 100 MB for batch files
    }

    fn supported_file_types(&self) -> Vec<String> {
        vec!["jsonl".to_string()] // Only JSONL for batch processing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HttpConfig;

    fn create_test_files() -> GroqFiles {
        GroqFiles::new(
            "test-api-key".to_string(),
            "https://api.groq.com/openai/v1".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        )
    }

    #[test]
    fn test_convert_groq_file() {
        let files = create_test_files();
        let groq_file = GroqFile {
            id: "file_123".to_string(),
            object: "file".to_string(),
            bytes: 1024,
            created_at: 1640995200, // 2022-01-01 00:00:00 UTC
            filename: "test.jsonl".to_string(),
            purpose: "batch".to_string(),
        };

        let file_response = files.convert_groq_file(groq_file);

        assert_eq!(file_response.id, "file_123");
        assert_eq!(file_response.filename, "test.jsonl");
        assert_eq!(file_response.size, 1024);
        assert_eq!(file_response.purpose, "batch");
    }

    #[test]
    fn test_capability_support() {
        let files = create_test_files();

        assert!(files.supports_file_upload());
        assert_eq!(files.max_file_size(), Some(100 * 1024 * 1024));
        assert_eq!(files.supported_file_types(), vec!["jsonl".to_string()]);
    }
}

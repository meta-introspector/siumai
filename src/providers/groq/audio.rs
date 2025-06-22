//! `Groq` Audio Capability Implementation
//!
//! Implements audio processing capabilities for Groq.

// Audio response types for Groq
#[derive(Debug, Clone)]
pub struct AudioTranscriptionResponse {
    pub text: String,
    pub language: Option<String>,
    pub duration: Option<f32>,
    pub segments: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct AudioTranslationResponse {
    pub text: String,
    pub language: Option<String>,
    pub duration: Option<f32>,
    pub segments: Option<Vec<serde_json::Value>>,
}
use reqwest::multipart::{Form, Part};

use crate::error::LlmError;
use crate::types::HttpConfig;

use super::types::*;
use super::utils::*;

/// `Groq` Audio Capability Implementation
pub struct GroqAudio {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
}

impl GroqAudio {
    /// Create a new `Groq` audio capability instance
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

    /// Create multipart form for audio upload
    fn create_audio_form(
        &self,
        audio_data: Vec<u8>,
        model: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        response_format: Option<&str>,
    ) -> Result<Form, LlmError> {
        let mut form = Form::new()
            .part("file", Part::bytes(audio_data).file_name("audio.wav"))
            .text("model", model.to_string());

        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }

        if let Some(p) = prompt {
            form = form.text("prompt", p.to_string());
        }

        if let Some(format) = response_format {
            form = form.text("response_format", format.to_string());
        }

        Ok(form)
    }
}

impl GroqAudio {
    /// Transcribe audio to text
    pub async fn transcribe(
        &self,
        audio_data: Vec<u8>,
        model: Option<String>,
        language: Option<String>,
        prompt: Option<String>,
    ) -> Result<AudioTranscriptionResponse, LlmError> {
        let model = model.unwrap_or_else(|| "whisper-large-v3".to_string());
        let url = format!("{}/audio/transcriptions", self.base_url);

        let form = self.create_audio_form(
            audio_data,
            &model,
            language.as_deref(),
            prompt.as_deref(),
            Some("json"),
        )?;

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
                message: format!("Groq transcription error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_response: GroqTranscriptionResponse = response.json().await?;

        Ok(AudioTranscriptionResponse {
            text: groq_response.text,
            language: None, // Groq doesn't return detected language
            duration: None, // Groq doesn't return duration
            segments: None, // Groq doesn't return segments in basic response
        })
    }

    /// Translate audio to English text
    pub async fn translate(
        &self,
        audio_data: Vec<u8>,
        model: Option<String>,
        prompt: Option<String>,
    ) -> Result<AudioTranslationResponse, LlmError> {
        let model = model.unwrap_or_else(|| "whisper-large-v3".to_string());
        let url = format!("{}/audio/translations", self.base_url);

        let form = self.create_audio_form(
            audio_data,
            &model,
            None, // Translation doesn't use language parameter
            prompt.as_deref(),
            Some("json"),
        )?;

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
                message: format!("Groq translation error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_response: GroqTranslationResponse = response.json().await?;

        Ok(AudioTranslationResponse {
            text: groq_response.text,
            language: Some("en".to_string()), // Groq translates to English
            duration: None,                   // Groq doesn't return duration
            segments: None,                   // Groq doesn't return segments in basic response
        })
    }

    /// Generate speech from text
    pub async fn speech(
        &self,
        text: String,
        model: Option<String>,
        voice: Option<String>,
        response_format: Option<String>,
        speed: Option<f32>,
    ) -> Result<Vec<u8>, LlmError> {
        let model = model.unwrap_or_else(|| "playai-tts".to_string());
        let voice = voice.unwrap_or_else(|| "Fritz-PlayAI".to_string());
        let response_format = response_format.unwrap_or_else(|| "wav".to_string());
        let speed = speed.unwrap_or(1.0);

        let url = format!("{}/audio/speech", self.base_url);

        let request_body = serde_json::json!({
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed
        });

        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq speech synthesis error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let audio_data = response.bytes().await?;
        Ok(audio_data.to_vec())
    }

    /// Check if transcription is supported
    pub fn supports_transcription(&self) -> bool {
        true
    }

    /// Check if translation is supported
    pub fn supports_translation(&self) -> bool {
        true
    }

    /// Check if speech synthesis is supported
    pub fn supports_speech_synthesis(&self) -> bool {
        true
    }

    /// Get supported audio models
    pub fn supported_audio_models(&self) -> Vec<String> {
        vec![
            "whisper-large-v3".to_string(),
            "whisper-large-v3-turbo".to_string(),
            "distil-whisper-large-v3-en".to_string(),
            "playai-tts".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HttpConfig;

    fn create_test_audio() -> GroqAudio {
        GroqAudio::new(
            "test-api-key".to_string(),
            "https://api.groq.com/openai/v1".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        )
    }

    #[test]
    fn test_create_audio_form() {
        let audio = create_test_audio();
        let audio_data = vec![1, 2, 3, 4]; // Dummy audio data

        let form = audio.create_audio_form(
            audio_data,
            "whisper-large-v3",
            Some("en"),
            Some("Test prompt"),
            Some("json"),
        );

        assert!(form.is_ok());
    }

    #[test]
    fn test_supported_audio_models() {
        let audio = create_test_audio();
        let models = audio.supported_audio_models();

        assert!(models.contains(&"whisper-large-v3".to_string()));
        assert!(models.contains(&"whisper-large-v3-turbo".to_string()));
        assert!(models.contains(&"playai-tts".to_string()));
    }

    #[test]
    fn test_capability_support() {
        let audio = create_test_audio();

        assert!(audio.supports_transcription());
        assert!(audio.supports_translation());
        assert!(audio.supports_speech_synthesis());
    }
}

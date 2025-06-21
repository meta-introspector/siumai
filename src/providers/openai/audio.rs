//! `OpenAI` Audio Implementation
//!
//! This module provides the `OpenAI` implementation of the `AudioCapability` trait,
//! including text-to-speech and speech-to-text functionality.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::AudioCapability;
use crate::types::{
    AudioFeature, AudioTranslationRequest, LanguageInfo, SttRequest, SttResponse, TtsRequest,
    TtsResponse, VoiceInfo, WordTimestamp,
};

use super::config::OpenAiConfig;

/// `OpenAI` TTS API request structure
#[derive(Debug, Clone, Serialize)]
struct OpenAiTtsRequest {
    /// Model to use
    model: String,
    /// Input text
    input: String,
    /// Voice to use
    voice: String,
    /// Response format
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    /// Speed of speech
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    /// Voice instructions (only for gpt-4o-mini-tts)
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
}

/// `OpenAI` STT API request structure (multipart form data)
#[derive(Debug, Clone)]
struct OpenAiSttRequest {
    /// Audio file data
    file_data: Vec<u8>,
    /// Filename
    filename: String,
    /// Model to use
    model: String,
    /// Language (optional)
    language: Option<String>,
    /// Prompt (optional)
    prompt: Option<String>,
    /// Response format
    response_format: Option<String>,
    /// Temperature
    temperature: Option<f32>,
    /// Timestamp granularities
    timestamp_granularities: Option<Vec<String>>,
}

/// `OpenAI` STT API response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiSttResponse {
    /// Transcribed text
    text: String,
    /// Language (if detected)
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    /// Duration in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f32>,
    /// Word-level timestamps
    #[serde(skip_serializing_if = "Option::is_none")]
    words: Option<Vec<OpenAiWordTimestamp>>,
}

/// `OpenAI` word timestamp structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiWordTimestamp {
    /// The word
    word: String,
    /// Start time in seconds
    start: f32,
    /// End time in seconds
    end: f32,
}

/// `OpenAI` audio capability implementation.
///
/// This struct provides the OpenAI-specific implementation of audio processing
/// capabilities including text-to-speech and speech-to-text.
///
/// # Supported Features
/// - Text-to-speech with multiple voices
/// - Speech-to-text transcription
/// - Audio translation (speech to English text)
/// - Multiple audio formats
///
/// # API References
/// - TTS: <https://platform.openai.com/docs/api-reference/audio/createSpeech>
/// - STT: <https://platform.openai.com/docs/api-reference/audio/createTranscription>
#[derive(Debug, Clone)]
pub struct OpenAiAudio {
    /// `OpenAI` configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
    /// Supported features
    features: Vec<AudioFeature>,
}

impl OpenAiAudio {
    /// Create a new `OpenAI` audio instance.
    ///
    /// # Arguments
    /// * `config` - `OpenAI` configuration
    /// * `http_client` - HTTP client for making requests
    pub fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        let features = vec![
            AudioFeature::TextToSpeech,
            AudioFeature::SpeechToText,
            AudioFeature::AudioTranslation,
            AudioFeature::CharacterTiming,
        ];

        Self {
            config,
            http_client,
            features,
        }
    }

    /// Get available TTS voices.
    fn get_tts_voices(&self) -> Vec<VoiceInfo> {
        vec![
            VoiceInfo {
                id: "alloy".to_string(),
                name: "Alloy".to_string(),
                description: Some("Neutral, balanced voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("neutral".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "ash".to_string(),
                name: "Ash".to_string(),
                description: Some("Warm, expressive voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("neutral".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "ballad".to_string(),
                name: "Ballad".to_string(),
                description: Some("Melodic, storytelling voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("neutral".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "coral".to_string(),
                name: "Coral".to_string(),
                description: Some("Bright, cheerful voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("female".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "echo".to_string(),
                name: "Echo".to_string(),
                description: Some("Male voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("male".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "fable".to_string(),
                name: "Fable".to_string(),
                description: Some("British accent".to_string()),
                language: Some("en".to_string()),
                gender: Some("male".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "nova".to_string(),
                name: "Nova".to_string(),
                description: Some("Female voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("female".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "onyx".to_string(),
                name: "Onyx".to_string(),
                description: Some("Deep male voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("male".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "sage".to_string(),
                name: "Sage".to_string(),
                description: Some("Wise, thoughtful voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("neutral".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "shimmer".to_string(),
                name: "Shimmer".to_string(),
                description: Some("Soft female voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("female".to_string()),
                category: Some("standard".to_string()),
            },
            VoiceInfo {
                id: "verse".to_string(),
                name: "Verse".to_string(),
                description: Some("Poetic, rhythmic voice".to_string()),
                language: Some("en".to_string()),
                gender: Some("neutral".to_string()),
                category: Some("standard".to_string()),
            },
        ]
    }

    /// Make a TTS API request.
    async fn make_tts_request(&self, request: OpenAiTtsRequest) -> Result<Vec<u8>, LlmError> {
        let url = format!("{}/audio/speech", self.config.base_url);

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("TTS request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI TTS API error {status}: {error_text}"),
                details: None,
            });
        }

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to read audio data: {e}")))?;

        Ok(audio_data.to_vec())
    }

    /// Make an STT API request.
    async fn make_stt_request(
        &self,
        request: OpenAiSttRequest,
    ) -> Result<OpenAiSttResponse, LlmError> {
        let url = format!("{}/audio/transcriptions", self.config.base_url);

        // Create multipart form
        let mut form = reqwest::multipart::Form::new();

        // Add file
        let file_part = reqwest::multipart::Part::bytes(request.file_data)
            .file_name(request.filename)
            .mime_str("audio/mpeg")
            .map_err(|e| LlmError::HttpError(format!("Failed to create file part: {e}")))?;
        form = form.part("file", file_part);

        // Add other fields
        form = form.text("model", request.model);

        if let Some(language) = request.language {
            form = form.text("language", language);
        }

        if let Some(prompt) = request.prompt {
            form = form.text("prompt", prompt);
        }

        if let Some(format) = request.response_format {
            form = form.text("response_format", format);
        }

        if let Some(temp) = request.temperature {
            form = form.text("temperature", temp.to_string());
        }

        if let Some(granularities) = request.timestamp_granularities {
            for granularity in granularities {
                form = form.text("timestamp_granularities[]", granularity);
            }
        }

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            // Skip Content-Type for multipart
            if key == "Content-Type" {
                continue;
            }
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .multipart(form)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("STT request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI STT API error {status}: {error_text}"),
                details: None,
            });
        }

        let openai_response: OpenAiSttResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse STT response: {e}")))?;

        Ok(openai_response)
    }

    /// Convert `OpenAI` STT response to our standard format.
    fn convert_stt_response(&self, openai_response: OpenAiSttResponse) -> SttResponse {
        let words = openai_response.words.map(|words| {
            words
                .into_iter()
                .map(|w| WordTimestamp {
                    word: w.word,
                    start: w.start,
                    end: w.end,
                    confidence: None, // OpenAI doesn't provide confidence scores
                })
                .collect()
        });

        SttResponse {
            text: openai_response.text,
            language: openai_response.language,
            confidence: None, // OpenAI doesn't provide overall confidence
            words,
            duration: openai_response.duration,
            metadata: HashMap::new(),
        }
    }

    /// Get supported TTS models.
    fn get_supported_tts_models(&self) -> Vec<String> {
        vec![
            "tts-1".to_string(),
            "tts-1-hd".to_string(),
            "gpt-4o-mini-tts".to_string(), // New model
        ]
    }

    /// Validate TTS request parameters.
    fn validate_tts_request(
        &self,
        model: &str,
        instructions: &Option<String>,
    ) -> Result<(), LlmError> {
        // Validate model
        if !self.get_supported_tts_models().contains(&model.to_string()) {
            return Err(LlmError::InvalidInput(format!(
                "Unsupported TTS model: {}. Supported models: {}",
                model,
                self.get_supported_tts_models().join(", ")
            )));
        }

        // Validate instructions parameter compatibility
        if let Some(instructions) = instructions {
            if model == "tts-1" || model == "tts-1-hd" {
                return Err(LlmError::InvalidInput(
                    "Instructions parameter is not supported for tts-1 and tts-1-hd models"
                        .to_string(),
                ));
            }

            // Validate instructions length
            if instructions.len() > 4096 {
                return Err(LlmError::InvalidInput(
                    "Instructions cannot exceed 4096 characters".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Check if a voice is supported.
    fn is_voice_supported(&self, voice: &str) -> bool {
        self.get_tts_voices().iter().any(|v| v.id == voice)
    }
}

#[async_trait]
impl AudioCapability for OpenAiAudio {
    /// Get all audio features supported by OpenAI.
    fn supported_features(&self) -> &[AudioFeature] {
        &self.features
    }

    /// Convert text to speech.
    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let voice = request.voice.unwrap_or_else(|| "alloy".to_string());
        let format = request.format.unwrap_or_else(|| "mp3".to_string());
        let model = request.model.unwrap_or_else(|| "tts-1".to_string());

        // Extract instructions from extra_params
        let instructions = request
            .extra_params
            .get("instructions")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string);

        // Validate voice
        if !self.is_voice_supported(&voice) {
            return Err(LlmError::InvalidInput(format!(
                "Unsupported voice: {}. Supported voices: {}",
                voice,
                self.get_tts_voices()
                    .iter()
                    .map(|v| v.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        // Validate request parameters
        self.validate_tts_request(&model, &instructions)?;

        let openai_request = OpenAiTtsRequest {
            model,
            input: request.text,
            voice,
            response_format: Some(format.clone()),
            speed: request.speed,
            instructions,
        };

        let audio_data = self.make_tts_request(openai_request).await?;

        Ok(TtsResponse {
            audio_data,
            format,
            duration: None, // OpenAI doesn't provide duration info
            sample_rate: None,
            metadata: HashMap::new(),
        })
    }

    /// Convert speech to text.
    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        let (file_data, filename) = if let Some(data) = request.audio_data {
            (data, "audio.mp3".to_string())
        } else if let Some(path) = request.file_path {
            let data = std::fs::read(&path)
                .map_err(|e| LlmError::IoError(format!("Failed to read audio file: {e}")))?;
            let filename = std::path::Path::new(&path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("audio.mp3")
                .to_string();
            (data, filename)
        } else {
            return Err(LlmError::InvalidInput(
                "Either audio_data or file_path must be provided".to_string(),
            ));
        };

        let model = request.model.unwrap_or_else(|| "whisper-1".to_string());

        let openai_request = OpenAiSttRequest {
            file_data,
            filename,
            model,
            language: request.language,
            prompt: None, // Could be added to SttRequest if needed
            response_format: Some("verbose_json".to_string()),
            temperature: None,
            timestamp_granularities: request.timestamp_granularities,
        };

        let openai_response = self.make_stt_request(openai_request).await?;
        Ok(self.convert_stt_response(openai_response))
    }

    /// Translate audio to English text.
    async fn translate_audio(
        &self,
        request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        let url = format!("{}/audio/translations", self.config.base_url);

        let (file_data, filename) = if let Some(data) = request.audio_data {
            (data, "audio.mp3".to_string())
        } else if let Some(path) = request.file_path {
            let data = std::fs::read(&path)
                .map_err(|e| LlmError::IoError(format!("Failed to read audio file: {e}")))?;
            let filename = std::path::Path::new(&path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("audio.mp3")
                .to_string();
            (data, filename)
        } else {
            return Err(LlmError::InvalidInput(
                "Either audio_data or file_path must be provided".to_string(),
            ));
        };

        // Create multipart form for translation
        let mut form = reqwest::multipart::Form::new();

        let file_part = reqwest::multipart::Part::bytes(file_data)
            .file_name(filename)
            .mime_str("audio/mpeg")
            .map_err(|e| LlmError::HttpError(format!("Failed to create file part: {e}")))?;
        form = form.part("file", file_part);

        let model = request.model.unwrap_or_else(|| "whisper-1".to_string());
        form = form.text("model", model);
        form = form.text("response_format", "json");

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            if key == "Content-Type" {
                continue;
            }
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .multipart(form)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Translation request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI Translation API error {status}: {error_text}"),
                details: None,
            });
        }

        let openai_response: OpenAiSttResponse = response.json().await.map_err(|e| {
            LlmError::ParseError(format!("Failed to parse translation response: {e}"))
        })?;

        Ok(self.convert_stt_response(openai_response))
    }

    /// Get available voices for TTS.
    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        Ok(self.get_tts_voices())
    }

    /// Get supported languages for transcription and translation.
    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        // OpenAI Whisper supports many languages
        let languages = vec![
            LanguageInfo {
                code: "en".to_string(),
                name: "English".to_string(),
                supports_transcription: true,
                supports_translation: true,
            },
            LanguageInfo {
                code: "zh".to_string(),
                name: "Chinese".to_string(),
                supports_transcription: true,
                supports_translation: true,
            },
            LanguageInfo {
                code: "es".to_string(),
                name: "Spanish".to_string(),
                supports_transcription: true,
                supports_translation: true,
            },
            LanguageInfo {
                code: "fr".to_string(),
                name: "French".to_string(),
                supports_transcription: true,
                supports_translation: true,
            },
            LanguageInfo {
                code: "de".to_string(),
                name: "German".to_string(),
                supports_transcription: true,
                supports_translation: true,
            },
            LanguageInfo {
                code: "ja".to_string(),
                name: "Japanese".to_string(),
                supports_transcription: true,
                supports_translation: true,
            },
            LanguageInfo {
                code: "ko".to_string(),
                name: "Korean".to_string(),
                supports_transcription: true,
                supports_translation: true,
            },
            // Add more languages as needed
        ];

        Ok(languages)
    }

    /// Get supported audio formats.
    fn get_supported_audio_formats(&self) -> Vec<String> {
        vec![
            "mp3".to_string(),
            "mp4".to_string(),
            "mpeg".to_string(),
            "mpga".to_string(),
            "m4a".to_string(),
            "wav".to_string(),
            "webm".to_string(),
        ]
    }
}

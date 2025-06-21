//! Common Streaming Utilities
//!
//! This module provides common utilities for handling streaming responses
//! across different providers, including line buffering and UTF-8 handling.

use std::sync::Arc;
use tokio::sync::Mutex;

/// Generic line buffer for handling incomplete lines in streaming responses
#[derive(Debug, Clone)]
pub struct LineBuffer {
    /// Buffer for incomplete lines
    buffer: Arc<Mutex<String>>,
}

impl LineBuffer {
    /// Create a new line buffer
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Add chunk to buffer and extract complete lines
    pub async fn add_chunk(&self, chunk: &str) -> Vec<String> {
        let mut buffer = self.buffer.lock().await;
        buffer.push_str(chunk);

        // Find complete lines (ending with \n)
        let mut lines = Vec::new();
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].to_string();
            *buffer = buffer[newline_pos + 1..].to_string();
            if !line.trim().is_empty() {
                lines.push(line);
            }
        }

        lines
    }

    /// Flush any remaining content in the buffer
    pub async fn flush(&self) -> Option<String> {
        let mut buffer = self.buffer.lock().await;
        if buffer.trim().is_empty() {
            None
        } else {
            let content = buffer.clone();
            buffer.clear();
            Some(content)
        }
    }
}

impl Default for LineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON object buffer for handling incomplete JSON objects in streaming responses
#[derive(Debug, Clone)]
pub struct JsonBuffer {
    /// Buffer for incomplete JSON objects
    buffer: Arc<Mutex<String>>,
}

impl JsonBuffer {
    /// Create a new JSON buffer
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Add chunk to buffer and extract complete JSON objects
    pub async fn add_chunk(&self, chunk: &str) -> Vec<String> {
        let mut buffer = self.buffer.lock().await;
        buffer.push_str(chunk);

        let mut objects = Vec::new();
        let mut brace_count = 0;
        let mut start_pos = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, ch) in buffer.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '"' if !escape_next => in_string = !in_string,
                '\\' if in_string => escape_next = true,
                '{' if !in_string => {
                    if brace_count == 0 {
                        start_pos = i;
                    }
                    brace_count += 1;
                }
                '}' if !in_string => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        // Found a complete JSON object
                        let json_str = buffer[start_pos..=i].to_string();
                        objects.push(json_str);

                        // Update buffer to remove processed object
                        let remaining = buffer[i + 1..].to_string();
                        *buffer = remaining;

                        // Reset for next object search
                        return objects; // Return early to avoid index issues
                    }
                }
                _ => {}
            }
        }

        objects
    }

    /// Flush any remaining content in the buffer
    pub async fn flush(&self) -> Option<String> {
        let mut buffer = self.buffer.lock().await;
        if buffer.trim().is_empty() {
            None
        } else {
            let content = buffer.clone();
            buffer.clear();
            Some(content)
        }
    }
}

impl Default for JsonBuffer {
    fn default() -> Self {
        Self::new()
    }
}

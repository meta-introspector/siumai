//! 🤖 Simple Chatbot - Interactive AI assistant

#![allow(clippy::single_match)]
//!
//! This example demonstrates a complete chatbot implementation:
//! - Interactive command-line interface
//! - Conversation memory and context management
//! - Multiple provider support with fallback
//! - Streaming responses for better UX
//! - Command system for bot control
//!
//! Before running, set your API keys:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! # Or start Ollama: ollama serve && ollama pull llama3.2
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example simple_chatbot
//! ```

use futures_util::StreamExt;
use siumai::prelude::*;
use std::collections::VecDeque;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Simple Chatbot - Interactive AI Assistant");
    println!("============================================\n");

    // Initialize the chatbot
    let mut chatbot = match Chatbot::new().await {
        Ok(bot) => bot,
        Err(e) => {
            println!("❌ Failed to initialize chatbot: {e}");
            println!("💡 Make sure you have set API keys or Ollama is running");
            return Ok(());
        }
    };

    // Show welcome message and instructions
    chatbot.show_welcome();

    // Main chat loop
    chatbot.run().await?;

    println!("\n👋 Goodbye! Thanks for chatting!");
    Ok(())
}

/// Simple chatbot implementation
struct Chatbot {
    client: Box<dyn ChatCapability + Send + Sync>,
    provider_name: String,
    conversation: VecDeque<ChatMessage>,
    max_history: usize,
    use_streaming: bool,
}

impl Chatbot {
    /// Create a new chatbot with the best available provider
    async fn new() -> Result<Self, LlmError> {
        let (client, provider_name) = Self::create_best_client().await?;

        let mut conversation = VecDeque::new();
        conversation.push_back(system!(
            "You are a helpful, friendly AI assistant. Keep responses concise but informative. \
             If asked about yourself, mention that you're powered by Siumai, a Rust library for LLM integration."
        ));

        Ok(Self {
            client,
            provider_name,
            conversation,
            max_history: 20, // Keep last 20 messages
            use_streaming: true,
        })
    }

    /// Try to create a client with the best available provider
    async fn create_best_client()
    -> Result<(Box<dyn ChatCapability + Send + Sync>, String), LlmError> {
        // Try OpenAI first
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if !api_key.is_empty() {
                match LlmBuilder::new()
                    .openai()
                    .api_key(&api_key)
                    .model("gpt-4o-mini")
                    .temperature(0.7)
                    .max_tokens(1000)
                    .build()
                    .await
                {
                    Ok(client) => return Ok((Box::new(client), "OpenAI".to_string())),
                    Err(_) => {} // Try next provider
                }
            }
        }

        // Try Anthropic
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            if !api_key.is_empty() {
                match LlmBuilder::new()
                    .anthropic()
                    .api_key(&api_key)
                    .model("claude-3-5-haiku-20241022")
                    .temperature(0.7)
                    .max_tokens(1000)
                    .build()
                    .await
                {
                    Ok(client) => return Ok((Box::new(client), "Anthropic".to_string())),
                    Err(_) => {} // Try next provider
                }
            }
        }

        // Try Ollama as fallback
        match LlmBuilder::new()
            .ollama()
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .temperature(0.7)
            .max_tokens(1000)
            .build()
            .await
        {
            Ok(client) => {
                // Test if Ollama is actually working
                let test_messages = vec![user!("Hi")];
                match client.chat(test_messages).await {
                    Ok(_) => return Ok((Box::new(client), "Ollama".to_string())),
                    Err(_) => {} // Ollama not working
                }
            }
            Err(_) => {} // Ollama not available
        }

        Err(LlmError::InternalError(
            "No AI providers available. Please set API keys or start Ollama.".to_string(),
        ))
    }

    /// Show welcome message and instructions
    fn show_welcome(&self) {
        println!("🎉 Chatbot initialized successfully!");
        println!("📡 Using provider: {}", self.provider_name);
        println!(
            "💬 Streaming: {}",
            if self.use_streaming {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!();
        println!("💡 Commands:");
        println!("   /help     - Show this help message");
        println!("   /clear    - Clear conversation history");
        println!("   /history  - Show conversation history");
        println!("   /stream   - Toggle streaming mode");
        println!("   /quit     - Exit the chatbot");
        println!();
        println!("Type your message and press Enter to chat!");
        println!("{}", "=".repeat(50));
    }

    /// Main chat loop
    async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            // Get user input
            print!("\n🧑 You: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            // Handle empty input
            if input.is_empty() {
                continue;
            }

            // Handle commands
            if input.starts_with('/') {
                if self.handle_command(input).await? {
                    break; // Exit if quit command
                }
                continue;
            }

            // Process user message
            self.add_user_message(input);

            // Get AI response
            print!("🤖 {}: ", self.provider_name);
            io::stdout().flush()?;

            match self.get_ai_response().await {
                Ok(response) => {
                    self.add_assistant_message(&response);
                    if !self.use_streaming {
                        println!("{response}");
                    }
                }
                Err(e) => {
                    println!("❌ Error: {e}");
                    println!("💡 Try again or use /help for commands");
                }
            }
        }

        Ok(())
    }

    /// Handle bot commands
    async fn handle_command(&mut self, command: &str) -> Result<bool, Box<dyn std::error::Error>> {
        match command {
            "/help" => {
                self.show_welcome();
            }
            "/clear" => {
                self.clear_history();
                println!("🧹 Conversation history cleared!");
            }
            "/history" => {
                self.show_history();
            }
            "/stream" => {
                self.use_streaming = !self.use_streaming;
                println!(
                    "🌊 Streaming {}",
                    if self.use_streaming {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            "/quit" | "/exit" => {
                return Ok(true); // Signal to exit
            }
            _ => {
                println!("❓ Unknown command: {command}");
                println!("💡 Type /help to see available commands");
            }
        }
        Ok(false)
    }

    /// Add user message to conversation
    fn add_user_message(&mut self, message: &str) {
        self.conversation.push_back(user!(message));
        self.trim_history();
    }

    /// Add assistant message to conversation
    fn add_assistant_message(&mut self, message: &str) {
        self.conversation.push_back(assistant!(message));
        self.trim_history();
    }

    /// Trim conversation history to max length
    fn trim_history(&mut self) {
        while self.conversation.len() > self.max_history {
            // Always keep the system message (first message)
            if self.conversation.len() > 1 {
                self.conversation.remove(1);
            } else {
                break;
            }
        }
    }

    /// Clear conversation history (except system message)
    fn clear_history(&mut self) {
        let system_message = self.conversation.front().cloned();
        self.conversation.clear();
        if let Some(msg) = system_message {
            self.conversation.push_back(msg);
        }
    }

    /// Show conversation history
    fn show_history(&self) {
        println!("\n📚 Conversation History:");
        println!("{}", "=".repeat(50));

        for (i, message) in self.conversation.iter().enumerate() {
            if i == 0 {
                continue; // Skip system message
            }

            match message.role {
                MessageRole::User => {
                    if let Some(text) = message.content.text() {
                        println!("🧑 You: {text}");
                    }
                }
                MessageRole::Assistant => {
                    if let Some(text) = message.content.text() {
                        println!("🤖 AI: {text}");
                    }
                }
                _ => {}
            }
        }

        if self.conversation.len() <= 1 {
            println!("(No conversation history yet)");
        }

        println!("{}", "=".repeat(50));
    }

    /// Get AI response (streaming or regular)
    async fn get_ai_response(&self) -> Result<String, LlmError> {
        let messages: Vec<_> = self.conversation.iter().cloned().collect();

        if self.use_streaming {
            self.get_streaming_response(messages).await
        } else {
            self.get_regular_response(messages).await
        }
    }

    /// Get streaming response
    async fn get_streaming_response(&self, messages: Vec<ChatMessage>) -> Result<String, LlmError> {
        let mut stream = self.client.chat_stream(messages, None).await?;
        let mut full_response = String::new();

        while let Some(event) = stream.next().await {
            match event? {
                ChatStreamEvent::ContentDelta { delta, .. } => {
                    print!("{delta}");
                    io::stdout().flush().unwrap();
                    full_response.push_str(&delta);
                }
                ChatStreamEvent::Done { .. } => {
                    println!(); // New line after streaming
                    break;
                }
                _ => {}
            }
        }

        Ok(full_response)
    }

    /// Get regular (non-streaming) response
    async fn get_regular_response(&self, messages: Vec<ChatMessage>) -> Result<String, LlmError> {
        let response = self.client.chat(messages).await?;
        Ok(response.content_text().unwrap_or_default().to_string())
    }
}

/*
🎯 Key Chatbot Implementation Concepts:

Architecture:
- Provider abstraction for flexibility
- Conversation memory management
- Command system for user control
- Streaming support for better UX

Features:
- Multi-provider support with fallback
- Conversation history management
- Interactive command system
- Real-time streaming responses
- Error handling and recovery

User Experience:
- Clear welcome and instructions
- Responsive streaming output
- Helpful error messages
- Command shortcuts for control
- Conversation history access

Production Considerations:
- Memory management for long conversations
- Error recovery and graceful degradation
- Rate limiting and cost control
- User session management
- Logging and monitoring

Extensibility:
- Easy to add new providers
- Pluggable command system
- Configurable conversation limits
- Customizable AI personality
- Integration with external services

Best Practices:
1. Provide clear user instructions
2. Handle errors gracefully
3. Manage conversation memory efficiently
4. Use streaming for better perceived performance
5. Implement fallback providers for reliability
6. Keep the interface simple and intuitive

Next Steps:
- code_assistant.rs: Specialized coding chatbot
- content_generator.rs: Content creation tool
- api_integration.rs: REST API with AI capabilities
*/

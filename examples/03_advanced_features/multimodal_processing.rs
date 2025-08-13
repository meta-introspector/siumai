//! Multimodal Processing Example
//!
//! This example demonstrates how to work with multiple types of content (text, images, audio)
//! in a single conversation using Siumai's multimodal capabilities.

use siumai::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 Multimodal Processing Example");
    println!("=================================\n");

    // 1. Text and Image Processing
    demonstrate_image_analysis().await?;

    // 2. Audio Processing
    demonstrate_audio_processing().await?;

    // 3. Combined Multimodal Content
    demonstrate_combined_modalities().await?;

    // 4. Multimodal Conversation
    demonstrate_multimodal_conversation().await?;

    Ok(())
}

/// Demonstrates image analysis capabilities
async fn demonstrate_image_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("🖼️  1. Image Analysis");
    println!("   Analyzing images with AI vision models\n");

    // Check if we have an API key for a vision-capable provider
    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .unwrap_or_else(|_| {
            println!("   ⚠️  No API key found. Using demo mode.");
            "demo-key".to_string()
        });

    // Create a vision-capable client
    let client = if std::env::var("OPENAI_API_KEY").is_ok() {
        Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o") // Vision-capable model
            .temperature(0.3)
            .build()
            .await?
    } else if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        Siumai::builder()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-sonnet-20241022") // Vision-capable model
            .temperature(0.3)
            .build()
            .await?
    } else {
        println!("   📝 Demo: Would analyze image with vision model");
        println!("   💡 Set OPENAI_API_KEY or ANTHROPIC_API_KEY to try real image analysis\n");
        return Ok(());
    };

    // Example image analysis scenarios
    let scenarios = vec![
        (
            "Chart Analysis",
            "Analyze this chart and explain the trends you see.",
        ),
        (
            "Code Screenshot",
            "What programming language is this? Explain what the code does.",
        ),
        (
            "Document OCR",
            "Extract and summarize the text from this document.",
        ),
        (
            "Scene Description",
            "Describe this scene in detail, including objects, people, and setting.",
        ),
    ];

    for (scenario, prompt) in scenarios {
        println!("   📊 {}", scenario);
        println!("     Prompt: {}", prompt);

        // In a real scenario, you would create a message with an image
        // let message = ChatMessage::user(prompt)
        //     .with_image("path/to/image.jpg", Some("high"))
        //     .build();

        println!("     📝 Demo: Would process image with vision model");
        println!("     🔍 Expected: Detailed analysis based on image content\n");
    }

    Ok(())
}

/// Demonstrates audio processing capabilities
async fn demonstrate_audio_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 2. Audio Processing");
    println!("   Processing audio content with AI models\n");

    // Audio processing scenarios
    let scenarios = vec![
        (
            "Speech Transcription",
            "Convert speech to text",
            "audio/speech.mp3",
        ),
        (
            "Music Analysis",
            "Analyze musical content and style",
            "audio/music.wav",
        ),
        (
            "Sound Classification",
            "Identify and classify sounds",
            "audio/environment.wav",
        ),
        (
            "Language Detection",
            "Detect the language being spoken",
            "audio/multilingual.mp3",
        ),
    ];

    for (scenario, description, file_path) in scenarios {
        println!("   🎧 {}", scenario);
        println!("     Description: {}", description);
        println!("     File: {}", file_path);

        // In a real scenario, you would process audio
        // let message = ChatMessage::user("Transcribe this audio")
        //     .with_audio(file_path, "mp3")
        //     .build();

        println!("     📝 Demo: Would process audio file");
        println!("     🔍 Expected: Transcription or analysis results\n");
    }

    Ok(())
}

/// Demonstrates combining multiple modalities
async fn demonstrate_combined_modalities() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 3. Combined Multimodal Content");
    println!("   Using text, images, and audio together\n");

    // Combined scenarios
    let scenarios = vec![
        (
            "Video Analysis",
            "Analyze this video frame and its audio track",
            vec!["image/frame.jpg", "audio/soundtrack.mp3"],
        ),
        (
            "Presentation Review",
            "Review this slide and its speaker notes",
            vec!["image/slide.png", "audio/narration.wav"],
        ),
        (
            "Document Analysis",
            "Analyze this document image and related audio explanation",
            vec!["image/document.jpg", "audio/explanation.mp3"],
        ),
    ];

    for (scenario, prompt, files) in scenarios {
        println!("   🎬 {}", scenario);
        println!("     Prompt: {}", prompt);
        println!("     Files: {:?}", files);

        // In a real scenario, you would combine multiple content types
        // let mut message_builder = ChatMessage::user(prompt);
        // for file in files {
        //     if file.contains("image/") {
        //         message_builder = message_builder.with_image(file, Some("high"));
        //     } else if file.contains("audio/") {
        //         message_builder = message_builder.with_audio(file, "mp3");
        //     }
        // }
        // let message = message_builder.build();

        println!("     📝 Demo: Would process multiple content types together");
        println!("     🔍 Expected: Comprehensive analysis across modalities\n");
    }

    Ok(())
}

/// Demonstrates a multimodal conversation
async fn demonstrate_multimodal_conversation() -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 4. Multimodal Conversation");
    println!("   Building a conversation with mixed content types\n");

    // Simulate a multimodal conversation flow
    let conversation_steps = vec![
        ("User", "Text", "I have a chart I'd like you to analyze"),
        (
            "Assistant",
            "Text",
            "I'd be happy to help! Please share the chart.",
        ),
        ("User", "Image", "[Uploads chart image]"),
        (
            "Assistant",
            "Text",
            "I can see this is a sales performance chart showing...",
        ),
        ("User", "Text", "Can you explain the trends in more detail?"),
        (
            "Assistant",
            "Text",
            "Certainly! The chart shows three key trends...",
        ),
        (
            "User",
            "Audio",
            "[Uploads audio question about specific data point]",
        ),
        (
            "Assistant",
            "Text",
            "Based on your audio question about Q3 data...",
        ),
    ];

    println!("   📱 Conversation Flow:");
    for (i, (speaker, content_type, content)) in conversation_steps.iter().enumerate() {
        let icon = match *content_type {
            "Text" => "💬",
            "Image" => "🖼️",
            "Audio" => "🎵",
            _ => "📄",
        };

        println!(
            "   {}. {} {}: {} {}",
            i + 1,
            if *speaker == "User" { "👤" } else { "🤖" },
            speaker,
            icon,
            content
        );
    }

    println!("\n   💡 Key Benefits of Multimodal Conversations:");
    println!("     • Rich context from multiple content types");
    println!("     • Natural interaction patterns");
    println!("     • Comprehensive understanding");
    println!("     • Flexible communication methods");

    println!("\n   🔧 Implementation Tips:");
    println!("     • Use appropriate models for each content type");
    println!("     • Consider file size and format limitations");
    println!("     • Handle different processing times gracefully");
    println!("     • Provide fallbacks for unsupported content");

    println!("\n   📊 Performance Considerations:");
    println!("     • Larger request sizes with multimodal content");
    println!("     • Different pricing for different modalities");
    println!("     • Provider-specific capabilities and limits");
    println!("     • Network bandwidth requirements");

    println!("\n✨ Multimodal processing complete! You now understand how to work");
    println!("   with text, images, and audio in AI conversations.");

    Ok(())
}

/// Helper function to check if a file exists (for real implementations)
#[allow(dead_code)]
fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Helper function to get file extension (for real implementations)
#[allow(dead_code)]
fn get_file_extension(path: &str) -> Option<&str> {
    Path::new(path).extension()?.to_str()
}

/// Helper function to validate image format (for real implementations)
#[allow(dead_code)]
fn is_supported_image_format(path: &str) -> bool {
    match get_file_extension(path) {
        Some("jpg") | Some("jpeg") | Some("png") | Some("gif") | Some("webp") => true,
        _ => false,
    }
}

/// Helper function to validate audio format (for real implementations)
#[allow(dead_code)]
fn is_supported_audio_format(path: &str) -> bool {
    match get_file_extension(path) {
        Some("mp3") | Some("wav") | Some("m4a") | Some("ogg") => true,
        _ => false,
    }
}

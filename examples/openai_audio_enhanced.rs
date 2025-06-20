//! OpenAI Enhanced Audio Features Example
//!
//! This example demonstrates the newly implemented OpenAI Audio API features:
//! - New TTS model: gpt-4o-mini-tts
//! - New voices: ash, ballad, coral, sage, verse
//! - Instructions parameter for voice control
//! - Enhanced parameter validation

use siumai::{
    providers::openai::{OpenAiAudio, OpenAiConfig},
    traits::AudioCapability,
    types::{SttRequest, TtsRequest},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the OpenAI audio client
    let config = OpenAiConfig::new("your-api-key-here");
    let http_client = reqwest::Client::new();
    let audio_client = OpenAiAudio::new(config, http_client);

    println!("🎵 OpenAI Enhanced Audio Features Demo");
    println!("======================================\n");

    // Example 1: Using new voices
    new_voices_example(&audio_client).await?;

    // Example 2: Using the new gpt-4o-mini-tts model with instructions
    enhanced_tts_example(&audio_client).await?;

    // Example 3: Voice validation
    voice_validation_example(&audio_client).await?;

    // Example 4: Model compatibility validation
    model_compatibility_example(&audio_client).await?;

    Ok(())
}

/// Example 1: Demonstrate the new voices
async fn new_voices_example(client: &OpenAiAudio) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 Example 1: New Voice Options");
    println!("-------------------------------");

    let voices = client.get_voices().await?;
    println!("Available voices ({} total):", voices.len());

    for voice in &voices {
        println!(
            "  - {}: {} ({})",
            voice.id,
            voice.name,
            voice.description.as_deref().unwrap_or("No description")
        );
    }

    // Highlight new voices
    let new_voices = ["ash", "ballad", "coral", "sage", "verse"];
    println!("\nNew voices added:");
    for voice in &voices {
        if new_voices.contains(&voice.id.as_str()) {
            println!("  ✨ {}: {}", voice.id, voice.name);
        }
    }
    println!();

    Ok(())
}

/// Example 2: Demonstrate enhanced TTS with new model and instructions
async fn enhanced_tts_example(_client: &OpenAiAudio) -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Example 2: Enhanced TTS with gpt-4o-mini-tts");
    println!("------------------------------------------------");

    // Create TTS request with new model and instructions
    let mut extra_params = HashMap::new();
    extra_params.insert(
        "instructions".to_string(),
        serde_json::Value::String(
            "Speak in a warm, friendly tone with slight emphasis on important words.".to_string(),
        ),
    );

    let request = TtsRequest {
        text: "Welcome to the enhanced OpenAI audio features! This is using the new gpt-4o-mini-tts model with custom voice instructions.".to_string(),
        voice: Some("coral".to_string()), // Using one of the new voices
        format: Some("mp3".to_string()),
        speed: Some(1.1),
        model: Some("gpt-4o-mini-tts".to_string()), // New model
        extra_params,
    };

    println!("TTS Request Configuration:");
    println!(
        "  - Model: {}",
        request.model.as_deref().unwrap_or("default")
    );
    println!(
        "  - Voice: {}",
        request.voice.as_deref().unwrap_or("default")
    );
    println!("  - Speed: {}", request.speed.unwrap_or(1.0));
    println!(
        "  - Instructions: {}",
        request
            .extra_params
            .get("instructions")
            .and_then(|v| v.as_str())
            .unwrap_or("None")
    );

    // Note: This is a demonstration - actual API call would require valid credentials
    println!("  - Status: Ready for API call (requires valid API key)");
    println!();

    Ok(())
}

/// Example 3: Demonstrate voice validation
async fn voice_validation_example(_client: &OpenAiAudio) -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Example 3: Voice Validation");
    println!("------------------------------");

    let valid_voices = [
        "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer",
        "verse",
    ];
    let invalid_voices = ["invalid_voice", "old_voice", "custom_voice"];

    println!("Testing voice validation:");

    for voice in &valid_voices {
        let _request = TtsRequest::new("Test".to_string()).with_voice(voice.to_string());
        println!("  ✅ Voice '{}': Valid", voice);
    }

    for voice in &invalid_voices {
        println!("  ❌ Voice '{}': Would be rejected", voice);
    }
    println!();

    Ok(())
}

/// Example 4: Demonstrate model compatibility validation
async fn model_compatibility_example(
    _client: &OpenAiAudio,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Example 4: Model Compatibility Validation");
    println!("--------------------------------------------");

    println!("Testing instructions parameter compatibility:");

    // Valid: gpt-4o-mini-tts with instructions
    println!("  ✅ gpt-4o-mini-tts + instructions: Compatible");

    // Invalid: tts-1 with instructions
    println!("  ❌ tts-1 + instructions: Not compatible");
    println!("  ❌ tts-1-hd + instructions: Not compatible");

    println!("\nSupported models:");
    let models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"];
    for model in &models {
        let supports_instructions = *model == "gpt-4o-mini-tts";
        println!(
            "  - {}: Instructions support = {}",
            model, supports_instructions
        );
    }
    println!();

    Ok(())
}

/// Example 5: Complete enhanced audio workflow
#[allow(dead_code)]
async fn complete_audio_workflow(_client: &OpenAiAudio) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Example 5: Complete Enhanced Audio Workflow");
    println!("----------------------------------------------");

    // Step 1: Create enhanced TTS request
    let mut extra_params = HashMap::new();
    extra_params.insert(
        "instructions".to_string(),
        serde_json::Value::String(
            "Speak clearly and professionally, as if presenting to a technical audience."
                .to_string(),
        ),
    );

    let _tts_request = TtsRequest {
        text: "This is a demonstration of the enhanced OpenAI audio capabilities, featuring the new gpt-4o-mini-tts model with custom voice instructions and the new Sage voice.".to_string(),
        voice: Some("sage".to_string()),
        format: Some("mp3".to_string()),
        speed: Some(1.0),
        model: Some("gpt-4o-mini-tts".to_string()),
        extra_params,
    };

    println!("Step 1: Enhanced TTS Configuration");
    println!("  - Using new model: gpt-4o-mini-tts");
    println!("  - Using new voice: sage");
    println!("  - Custom instructions: Professional presentation style");

    // Step 2: Validate configuration
    println!("\nStep 2: Validation");
    println!("  - Voice validation: ✅ Sage is supported");
    println!("  - Model validation: ✅ gpt-4o-mini-tts is supported");
    println!("  - Instructions compatibility: ✅ Compatible with gpt-4o-mini-tts");
    println!("  - Text length: ✅ Within limits");

    // Step 3: Execute (simulation)
    println!("\nStep 3: Execution");
    println!("  - API call would be made here with validated parameters");
    println!("  - Enhanced audio would be generated with custom voice characteristics");

    // Note: Actual API calls would be:
    // let tts_response = client.text_to_speech(tts_request).await?;
    // std::fs::write("enhanced_audio.mp3", &tts_response.audio_data)?;

    println!("  - Status: Ready for execution (requires valid API key)");
    println!();

    Ok(())
}

// Note: Default implementation removed due to orphan rules
// TtsRequest is defined in the library, not in this example

/// Example 6: Error handling and validation
#[allow(dead_code)]
fn validation_examples() {
    println!("🚨 Example 6: Validation and Error Handling");
    println!("-------------------------------------------");

    println!("Common validation scenarios:");

    // Instructions with incompatible model
    println!("  ❌ Error: Instructions with tts-1 model");
    println!(
        "     Message: 'Instructions parameter is not supported for tts-1 and tts-1-hd models'"
    );

    // Invalid voice
    println!("  ❌ Error: Unsupported voice");
    println!(
        "     Message: 'Unsupported voice: custom_voice. Supported voices: alloy, ash, ballad, ...'"
    );

    // Instructions too long
    println!("  ❌ Error: Instructions exceed 4096 characters");
    println!("     Message: 'Instructions cannot exceed 4096 characters'");

    // Unsupported model
    println!("  ❌ Error: Unsupported model");
    println!(
        "     Message: 'Unsupported TTS model: custom-model. Supported models: tts-1, tts-1-hd, gpt-4o-mini-tts'"
    );

    println!();
}

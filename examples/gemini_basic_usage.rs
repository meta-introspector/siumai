//! Basic Gemini Usage Example
//!
//! This example demonstrates how to use the Gemini provider for basic chat functionality.

use siumai::{ChatCapability, ModelListingCapability, Tool, ToolFunction, system, user, Provider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    // Get API key from environment
    let api_key =
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY environment variable is required");

    println!("🚀 Creating Gemini client...");

    // Create a Gemini client
    let client = Provider::gemini()
        .api_key(api_key)
        .model("gemini-1.5-flash")
        .temperature(0.7)
        .max_tokens(1000)
        .build()
        .await?;

    println!("✅ Gemini client created successfully!");

    // Test basic chat
    println!("\n💬 Testing basic chat...");
    let messages = vec![user!(
        "Hello! Can you tell me a short joke about programming?"
    )];

    let response = client.chat_with_tools(messages, None).await?;

    if let Some(text) = response.text() {
        println!("🤖 Gemini: {}", text);
    } else {
        println!("❌ No text response received");
    }

    // Test with system message
    println!("\n🎭 Testing with system message...");
    let messages = vec![
        system!("You are a helpful assistant that responds in a pirate accent."),
        user!("What's the weather like today?"),
    ];

    let response = client.chat_with_tools(messages, None).await?;

    if let Some(text) = response.text() {
        println!("🏴‍☠️ Pirate Gemini: {}", text);
    }

    // Test function calling
    println!("\n🔧 Testing function calling...");

    // Define a simple function
    let get_weather_tool = Tool {
        r#type: "function".to_string(),
        function: ToolFunction {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }),
        },
    };

    let messages = vec![user!("What's the weather like in Tokyo?")];

    let response = client
        .chat_with_tools(messages, Some(vec![get_weather_tool]))
        .await?;

    if let Some(tool_calls) = &response.tool_calls {
        println!("🛠️ Tool calls requested:");
        for tool_call in tool_calls {
            if let Some(function) = &tool_call.function {
                println!("  - Function: {}", function.name);
                println!("    Arguments: {}", function.arguments);
            }
        }
    } else if let Some(text) = response.text() {
        println!("🤖 Gemini: {}", text);
    }

    // Test model listing
    println!("\n📋 Testing model listing...");
    match client.list_models().await {
        Ok(models) => {
            println!("Available models:");
            for model in models.iter().take(5) {
                // Show first 5 models
                println!(
                    "  - {}: {}",
                    model.id,
                    model.name.as_deref().unwrap_or("Unknown")
                );
                if let Some(desc) = &model.description {
                    println!("    Description: {}", desc);
                }
                println!(
                    "    Context window: {} tokens",
                    model.context_window.unwrap_or(0)
                );
            }
            if models.len() > 5 {
                println!("  ... and {} more models", models.len() - 5);
            }
        }
        Err(e) => {
            println!("❌ Failed to list models: {}", e);
        }
    }

    // Test with JSON schema (structured output)
    println!("\n📊 Testing structured output with JSON schema...");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "occupation": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age", "occupation"]
    });

    let structured_client = Provider::gemini()
        .api_key(std::env::var("GEMINI_API_KEY")?)
        .model("gemini-1.5-flash")
        .json_schema(schema)
        .build()
        .await?;

    let messages = vec![user!(
        "Create a profile for a fictional software engineer named Alex who is 28 years old."
    )];

    let response = structured_client.chat_with_tools(messages, None).await?;

    if let Some(text) = response.text() {
        println!("📋 Structured response: {}", text);

        // Try to parse as JSON
        match serde_json::from_str::<serde_json::Value>(&*text) {
            Ok(json) => {
                println!("✅ Successfully parsed as JSON:");
                println!("{}", serde_json::to_string_pretty(&json)?);
            }
            Err(e) => {
                println!("⚠️ Response is not valid JSON: {}", e);
            }
        }
    }

    println!("\n🎉 All tests completed!");

    Ok(())
}

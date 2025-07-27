//! OpenAI Responses API Example
//!
//! This example demonstrates how to use OpenAI's Responses API which provides:
//! - Stateful conversations with automatic context management
//! - Background processing for long-running tasks
//! - Built-in tools (web search, file search, computer use)
//! - Response lifecycle management (create, get, cancel, list)
//!
//! Note: This is an OpenAI-specific API that is not available on other providers.
//! You need a valid OpenAI API key to run this example.
//!
//! Run with: OPENAI_API_KEY=your_key cargo run --example openai_responses_api

use siumai::prelude::*;
use siumai::providers::openai::responses::{
    ListResponsesQuery, ResponseStatus, ResponsesApiCapability,
};
use siumai::types::OpenAiBuiltInTool;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    siumai::tracing::init_default_tracing().ok();

    // Get API key from environment
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    println!("🚀 OpenAI Responses API Example");
    println!("================================\n");

    // Create OpenAI client with Responses API enabled
    let client = LlmBuilder::new()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o")
        .build()
        .await?;

    // Get the OpenAI client to access Responses API
    let _openai_client = client
        .as_any()
        .downcast_ref::<siumai::providers::openai::client::OpenAiClient>()
        .expect("Expected OpenAI client");

    // Create Responses API client
    let responses_client = siumai::providers::openai::responses::OpenAiResponses::new(
        reqwest::Client::new(),
        siumai::providers::openai::config::OpenAiConfig::new(&api_key)
            .with_model("gpt-4o")
            .with_responses_api(true)
            .with_built_in_tool(OpenAiBuiltInTool::WebSearch),
    );

    // Example 1: Basic chat with Responses API
    println!("1️⃣ Basic Chat with Responses API");
    println!("--------------------------------");

    let messages = vec![user!("What's the weather like today in San Francisco?")];

    let response = responses_client
        .chat_with_tools(messages.clone(), None)
        .await?;

    println!("Response: {}", response.content.all_text());
    println!();

    // Example 2: Background processing
    println!("2️⃣ Background Processing");
    println!("------------------------");

    let complex_messages = vec![user!(
        "Please research the latest developments in quantum computing and write a comprehensive summary."
    )];

    // Start background processing
    let background_response = responses_client
        .create_response_background(
            complex_messages,
            None,
            Some(vec![OpenAiBuiltInTool::WebSearch]),
            None,
        )
        .await?;

    println!(
        "Started background task with ID: {}",
        background_response.id
    );
    println!("Status: {:?}", background_response.status);

    // Poll for completion
    let mut attempts = 0;
    let max_attempts = 30; // 30 seconds max

    loop {
        sleep(Duration::from_secs(1)).await;
        attempts += 1;

        let is_ready = responses_client
            .is_response_ready(&background_response.id)
            .await?;

        if is_ready {
            println!("✅ Background task completed!");

            // Get the final response
            let final_response = responses_client
                .get_response(&background_response.id)
                .await?;

            println!("Final response: {}", final_response.content.all_text());
            break;
        }

        if attempts >= max_attempts {
            println!("⏰ Timeout waiting for background task");

            // Cancel the background task
            let cancelled = responses_client
                .cancel_response(&background_response.id)
                .await?;

            println!("Cancelled task: {:?}", cancelled.status);
            break;
        }

        print!(".");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
    }
    println!();

    // Example 3: Conversation chaining
    println!("3️⃣ Conversation Chaining");
    println!("------------------------");

    // First message
    let first_messages = vec![user!(
        "Tell me about the history of artificial intelligence."
    )];

    let first_response = responses_client
        .chat_with_tools(first_messages, None)
        .await?;

    println!("First response: {}", first_response.content.all_text());

    // Continue the conversation using the response ID
    if let Some(response_id) = &first_response.id {
        let follow_up_messages = vec![user!("What are the current challenges in AI development?")];

        let continued_response = ResponsesApiCapability::continue_conversation(
            &responses_client,
            response_id.clone(),
            follow_up_messages,
            None,
            false,
        )
        .await?;

        println!(
            "Continued response: {}",
            continued_response.content.all_text()
        );
    }
    println!();

    // Example 4: List responses
    println!("4️⃣ List Recent Responses");
    println!("------------------------");

    let query = ListResponsesQuery {
        limit: Some(5),
        status: Some(ResponseStatus::Completed),
        ..Default::default()
    };

    let responses_list = responses_client.list_responses(Some(query)).await?;

    println!("Recent completed responses:");
    for (i, response_meta) in responses_list.iter().enumerate() {
        println!(
            "  {}. ID: {} | Model: {} | Created: {}",
            i + 1,
            response_meta.id,
            response_meta.model,
            response_meta.created_at
        );
    }
    println!();

    // Example 5: Built-in tools
    println!("5️⃣ Built-in Tools (Web Search)");
    println!("------------------------------");

    let search_messages = vec![user!("What are the latest news about SpaceX launches?")];

    let search_response = responses_client
        .chat_with_tools(search_messages, None)
        .await?;

    println!(
        "Search-enhanced response: {}",
        search_response.content.all_text()
    );

    if let Some(usage) = &search_response.usage {
        println!(
            "Token usage - Prompt: {}, Completion: {}, Total: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    println!("\n✨ OpenAI Responses API example completed!");

    Ok(())
}

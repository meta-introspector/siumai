//! Concurrent Usage Example
//!
//! This example demonstrates how to use siumai clients in concurrent scenarios
//! using the Clone trait support.

use siumai::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Siumai Concurrent Usage Example");
    println!("=====================================\n");

    // Example 1: Concurrent requests with Arc<Client>
    println!("📋 Example 1: Concurrent requests with Arc<Client>");
    concurrent_with_arc().await?;

    println!("\n" + "=".repeat(50) + "\n");

    // Example 2: Direct client cloning
    println!("📋 Example 2: Direct client cloning");
    direct_clone_usage().await?;

    println!("\n" + "=".repeat(50) + "\n");

    // Example 3: Multiple providers concurrently
    println!("📋 Example 3: Multiple providers concurrently");
    multiple_providers_concurrent().await?;

    println!("\n" + "=".repeat(50) + "\n");

    // Example 4: Unified interface concurrent usage
    println!("📋 Example 4: Unified interface concurrent usage");
    unified_interface_concurrent().await?;

    Ok(())
}

/// Example 1: Using Arc<Client> for concurrent requests
async fn concurrent_with_arc() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client (using OpenAI as example, but works with any provider)
    let client = if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        Provider::openai()
            .api_key(api_key)
            .model("gpt-4o-mini")
            .temperature(0.7)
            .build()
            .await?
    } else {
        println!("⚠️  OPENAI_API_KEY not set, using mock client");
        return Ok(());
    };

    println!("✅ Created OpenAI client");

    // Wrap in Arc for sharing across tasks
    let client_arc = Arc::new(client);
    let mut handles = vec![];
    let start_time = Instant::now();

    // Spawn multiple concurrent tasks
    for i in 0..5 {
        let client_clone = Arc::clone(&client_arc);
        let handle = task::spawn(async move {
            let task_start = Instant::now();
            let messages = vec![user!(format!(
                "Task {}: Explain what {} is in one sentence.",
                i,
                match i {
                    0 => "artificial intelligence",
                    1 => "machine learning",
                    2 => "neural networks",
                    3 => "deep learning",
                    _ => "natural language processing",
                }
            ))];

            match client_clone.chat(messages).await {
                Ok(response) => {
                    let duration = task_start.elapsed();
                    (i, Ok(response.text().unwrap_or_default()), duration)
                }
                Err(e) => (i, Err(e), task_start.elapsed()),
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    println!("⏳ Waiting for {} concurrent requests...", handles.len());
    for handle in handles {
        let (task_id, result, duration) = handle.await?;
        match result {
            Ok(text) => {
                println!(
                    "✅ Task {} completed in {:?}: {}",
                    task_id,
                    duration,
                    text.chars().take(100).collect::<String>()
                        + if text.len() > 100 { "..." } else { "" }
                );
            }
            Err(e) => {
                println!("❌ Task {} failed in {:?}: {}", task_id, duration, e);
            }
        }
    }

    let total_duration = start_time.elapsed();
    println!("🏁 All tasks completed in {:?}", total_duration);

    Ok(())
}

/// Example 2: Direct client cloning
async fn direct_clone_usage() -> Result<(), Box<dyn std::error::Error>> {
    let client = if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        Provider::anthropic()
            .api_key(api_key)
            .model("claude-3-haiku-20240307")
            .temperature(0.5)
            .build()
            .await?
    } else {
        println!("⚠️  ANTHROPIC_API_KEY not set, skipping this example");
        return Ok(());
    };

    println!("✅ Created Anthropic client");

    // Clone the client directly (lightweight operation)
    let client1 = client.clone();
    let client2 = client.clone();
    let client3 = client;

    println!("📋 Created 3 client clones");

    // Use clones in different contexts
    let handle1 = task::spawn(async move {
        let messages = vec![user!("What is the capital of France?")];
        client1.chat(messages).await
    });

    let handle2 = task::spawn(async move {
        let messages = vec![user!("What is the capital of Germany?")];
        client2.chat(messages).await
    });

    let handle3 = task::spawn(async move {
        let messages = vec![user!("What is the capital of Italy?")];
        client3.chat(messages).await
    });

    // Wait for results
    let results = tokio::try_join!(handle1, handle2, handle3)?;

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(response) => {
                println!(
                    "✅ Clone {} response: {}",
                    i + 1,
                    response.text().unwrap_or_default()
                );
            }
            Err(e) => {
                println!("❌ Clone {} error: {}", i + 1, e);
            }
        }
    }

    Ok(())
}

/// Example 3: Multiple providers concurrently
async fn multiple_providers_concurrent() -> Result<(), Box<dyn std::error::Error>> {
    let mut clients = Vec::new();

    // Try to create clients for available providers
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = Provider::openai()
            .api_key(api_key)
            .model("gpt-4o-mini")
            .build()
            .await?;
        clients.push(("OpenAI", client));
    }

    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        let client = Provider::anthropic()
            .api_key(api_key)
            .model("claude-3-haiku-20240307")
            .build()
            .await?;
        clients.push(("Anthropic", client));
    }

    if clients.is_empty() {
        println!("⚠️  No API keys available, skipping this example");
        return Ok(());
    }

    println!("✅ Created {} provider clients", clients.len());

    let mut handles = vec![];

    for (provider_name, client) in clients {
        let handle = task::spawn(async move {
            let messages = vec![user!("What is your name and who created you?")];
            let result = client.chat(messages).await;
            (provider_name, result)
        });
        handles.push(handle);
    }

    println!("⏳ Querying multiple providers concurrently...");

    for handle in handles {
        let (provider_name, result) = handle.await?;
        match result {
            Ok(response) => {
                println!(
                    "✅ {} response: {}",
                    provider_name,
                    response.text().unwrap_or_default()
                );
            }
            Err(e) => {
                println!("❌ {} error: {}", provider_name, e);
            }
        }
    }

    Ok(())
}

/// Example 4: Unified interface concurrent usage
async fn unified_interface_concurrent() -> Result<(), Box<dyn std::error::Error>> {
    let client = if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        Siumai::builder()
            .openai()
            .api_key(api_key)
            .model("gpt-4o-mini")
            .temperature(0.8)
            .build()
            .await?
    } else {
        println!("⚠️  OPENAI_API_KEY not set, skipping this example");
        return Ok(());
    };

    println!("✅ Created unified Siumai client");

    // Clone the unified client
    let client_arc = Arc::new(client);
    let mut handles = vec![];

    let questions = vec![
        "What is the meaning of life?",
        "How does photosynthesis work?",
        "What is quantum computing?",
        "Explain blockchain technology",
        "What is artificial general intelligence?",
    ];

    for (i, question) in questions.into_iter().enumerate() {
        let client_clone = Arc::clone(&client_arc);
        let handle = task::spawn(async move {
            let messages = vec![user!(question)];
            let result = client_clone.chat(messages).await;
            (i, question, result)
        });
        handles.push(handle);
    }

    println!("⏳ Processing {} questions concurrently...", handles.len());

    for handle in handles {
        let (i, question, result) = handle.await?;
        match result {
            Ok(response) => {
                println!(
                    "✅ Q{}: {} -> {}",
                    i + 1,
                    question,
                    response
                        .text()
                        .unwrap_or_default()
                        .chars()
                        .take(100)
                        .collect::<String>()
                        + "..."
                );
            }
            Err(e) => {
                println!("❌ Q{}: {} -> Error: {}", i + 1, question, e);
            }
        }
    }

    Ok(())
}

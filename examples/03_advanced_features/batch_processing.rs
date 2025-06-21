//! ⚡ Batch Processing - High-volume concurrent processing
//!
//! This example demonstrates how to process large numbers of requests efficiently:
//! - Concurrent request handling with proper rate limiting
//! - Progress tracking and monitoring

#![allow(unused_variables)]
#![allow(clippy::let_and_return)]
//! - Error handling and retry strategies at scale
//! - Memory management for large batches
//! - Performance optimization techniques
//!
//! Before running, set your API keys:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example batch_processing
//! ```

use siumai::prelude::*;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Batch Processing - High-volume concurrent processing\n");

    // Demonstrate different batch processing patterns
    demonstrate_basic_batch_processing().await;
    demonstrate_rate_limited_processing().await;
    demonstrate_progress_tracking().await;
    demonstrate_error_handling_at_scale().await;
    demonstrate_memory_efficient_processing().await;

    println!("\n✅ Batch processing examples completed!");
    Ok(())
}

/// Demonstrate basic concurrent batch processing
async fn demonstrate_basic_batch_processing() {
    println!("🔄 Basic Batch Processing:\n");

    if let Ok(client) = create_test_client().await {
        // Create a batch of tasks
        let tasks = create_sample_tasks(10);
        
        println!("   Processing {} tasks concurrently...", tasks.len());
        let start_time = Instant::now();

        // Process all tasks concurrently
        let futures: Vec<_> = tasks.into_iter()
            .enumerate()
            .map(|(i, task)| {
                let client = client.clone();
                async move {
                    let result = process_single_task(client.as_ref(), &task).await;
                    (i, result)
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        let duration = start_time.elapsed();

        // Analyze results
        let successful = results.iter().filter(|(_, r)| r.is_ok()).count();
        let failed = results.len() - successful;

        println!("   📊 Results:");
        println!("      ✅ Successful: {}", successful);
        println!("      ❌ Failed: {}", failed);
        println!("      ⏱️  Total time: {:?}", duration);
        println!("      📈 Throughput: {:.2} tasks/second", 
            results.len() as f64 / duration.as_secs_f64());

        println!("   ✅ Basic batch processing completed");
    } else {
        println!("   ⚠️  No client available for batch processing");
    }
    
    println!();
}

/// Demonstrate rate-limited batch processing
async fn demonstrate_rate_limited_processing() {
    println!("⏱️ Rate-Limited Processing:\n");

    if let Ok(client) = create_test_client().await {
        let tasks = create_sample_tasks(15);
        
        println!("   Processing {} tasks with rate limiting...", tasks.len());
        let start_time = Instant::now();

        // Process with rate limiting (max 3 concurrent, 500ms delay between batches)
        let results = process_with_rate_limiting(
            client,
            tasks,
            3,                              // max concurrent
            Duration::from_millis(500),     // delay between requests
        ).await;

        let duration = start_time.elapsed();
        let successful = results.iter().filter(|r| r.is_ok()).count();
        let failed = results.len() - successful;

        println!("   📊 Rate-Limited Results:");
        println!("      ✅ Successful: {}", successful);
        println!("      ❌ Failed: {}", failed);
        println!("      ⏱️  Total time: {:?}", duration);
        println!("      🐌 Controlled throughput: {:.2} tasks/second", 
            results.len() as f64 / duration.as_secs_f64());

        println!("   ✅ Rate-limited processing completed");
    } else {
        println!("   ⚠️  No client available for rate-limited processing");
    }
    
    println!();
}

/// Demonstrate progress tracking
async fn demonstrate_progress_tracking() {
    println!("📊 Progress Tracking:\n");

    if let Ok(client) = create_test_client().await {
        let tasks = create_sample_tasks(20);
        
        println!("   Processing {} tasks with progress tracking...", tasks.len());

        let results = process_with_progress_tracking(client, tasks).await;

        let successful = results.iter().filter(|r| r.is_ok()).count();
        let failed = results.len() - successful;

        println!("\n   📊 Final Results:");
        println!("      ✅ Successful: {}", successful);
        println!("      ❌ Failed: {}", failed);
        println!("      📈 Success rate: {:.1}%", 
            (successful as f64 / results.len() as f64) * 100.0);

        println!("   ✅ Progress tracking completed");
    } else {
        println!("   ⚠️  No client available for progress tracking");
    }
    
    println!();
}

/// Demonstrate error handling at scale
async fn demonstrate_error_handling_at_scale() {
    println!("🛡️ Error Handling at Scale:\n");

    if let Ok(client) = create_test_client().await {
        // Create tasks with some that will likely fail
        let mut tasks = create_sample_tasks(10);
        tasks.extend(create_error_prone_tasks(5));

        println!("   Processing {} tasks with robust error handling...", tasks.len());

        let results = process_with_error_handling(client, tasks).await;

        // Categorize results
        let mut successful = 0;
        let mut retried = 0;
        let mut failed = 0;

        for result in &results {
            match result {
                Ok(_) => successful += 1,
                Err(e) if e.to_string().contains("retry") => retried += 1,
                Err(_) => failed += 1,
            }
        }

        println!("   📊 Error Handling Results:");
        println!("      ✅ Successful: {}", successful);
        println!("      🔄 Retried and succeeded: {}", retried);
        println!("      ❌ Permanently failed: {}", failed);

        println!("   ✅ Error handling at scale completed");
    } else {
        println!("   ⚠️  No client available for error handling demo");
    }
    
    println!();
}

/// Demonstrate memory-efficient processing for large batches
async fn demonstrate_memory_efficient_processing() {
    println!("💾 Memory-Efficient Processing:\n");

    if let Ok(client) = create_test_client().await {
        let total_tasks = 50;
        let chunk_size = 10;

        println!("   Processing {} tasks in chunks of {}...", total_tasks, chunk_size);

        let mut total_successful = 0;
        let mut total_failed = 0;
        let start_time = Instant::now();

        // Process in chunks to manage memory
        for chunk_start in (0..total_tasks).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_tasks);
            let chunk_tasks = create_sample_tasks(chunk_end - chunk_start);

            println!("   Processing chunk {}-{}...", chunk_start + 1, chunk_end);

            let chunk_results = process_chunk(client.clone(), chunk_tasks).await;
            
            let chunk_successful = chunk_results.iter().filter(|r| r.is_ok()).count();
            let chunk_failed = chunk_results.len() - chunk_successful;

            total_successful += chunk_successful;
            total_failed += chunk_failed;

            println!("      Chunk results: {} ✅, {} ❌", chunk_successful, chunk_failed);

            // Small delay between chunks to be respectful
            sleep(Duration::from_millis(100)).await;
        }

        let duration = start_time.elapsed();

        println!("\n   📊 Memory-Efficient Results:");
        println!("      ✅ Total successful: {}", total_successful);
        println!("      ❌ Total failed: {}", total_failed);
        println!("      ⏱️  Total time: {:?}", duration);
        println!("      💾 Memory usage: Constant (chunked processing)");

        println!("   ✅ Memory-efficient processing completed");
    } else {
        println!("   ⚠️  No client available for memory-efficient processing");
    }
    
    println!();
}

/// Create a test client
async fn create_test_client() -> Result<Arc<dyn ChatCapability>, LlmError> {
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = LlmBuilder::new()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .temperature(0.7)
            .build()
            .await?;
        Ok(Arc::new(client) as Arc<dyn ChatCapability>)
    } else if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        let client = LlmBuilder::new()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-haiku-20241022")
            .temperature(0.7)
            .build()
            .await?;
        Ok(Arc::new(client) as Arc<dyn ChatCapability>)
    } else {
        Err(LlmError::AuthenticationError("No API key available".to_string()))
    }
}

/// Create sample tasks for processing
fn create_sample_tasks(count: usize) -> Vec<String> {
    (1..=count)
        .map(|i| format!("Task {}: What is {}+{}? Answer briefly.", i, i, i+1))
        .collect()
}

/// Create tasks that are likely to cause errors
fn create_error_prone_tasks(count: usize) -> Vec<String> {
    (1..=count)
        .map(|i| format!("Error task {}: [This might cause an error] Process this: {}", i, "x".repeat(1000)))
        .collect()
}

/// Process a single task
async fn process_single_task(client: &dyn ChatCapability, task: &str) -> Result<String, LlmError> {
    let messages = vec![user!(task)];
    let response = client.chat(messages).await?;
    Ok(response.content_text().unwrap_or_default().to_string())
}

/// Process tasks with rate limiting
async fn process_with_rate_limiting(
    client: Arc<dyn ChatCapability>,
    tasks: Vec<String>,
    max_concurrent: usize,
    delay_between_requests: Duration,
) -> Vec<Result<String, LlmError>> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent));

    stream::iter(tasks)
        .map(|task| {
            let client = client.clone();
            let semaphore = semaphore.clone();
            async move {
                let _permit = semaphore.acquire().await.unwrap();
                sleep(delay_between_requests).await;
                process_single_task(client.as_ref(), &task).await
            }
        })
        .buffer_unordered(max_concurrent)
        .collect()
        .await
}

/// Process tasks with progress tracking
async fn process_with_progress_tracking(
    client: Arc<dyn ChatCapability>,
    tasks: Vec<String>,
) -> Vec<Result<String, LlmError>> {
    let total = tasks.len();
    let mut completed = 0;

    let futures: Vec<_> = tasks.into_iter()
        .map(|task| {
            let client = client.clone();
            async move {
                let result = process_single_task(client.as_ref(), &task).await;
                result
            }
        })
        .collect();

    let mut results = Vec::new();
    
    for future in futures {
        let result = future.await;
        completed += 1;
        
        let progress = (completed as f64 / total as f64) * 100.0;
        print!("\r   Progress: {:.1}% ({}/{})", progress, completed, total);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        results.push(result);
    }
    
    println!(); // New line after progress
    results
}

/// Process tasks with comprehensive error handling
async fn process_with_error_handling(
    client: Arc<dyn ChatCapability>,
    tasks: Vec<String>,
) -> Vec<Result<String, LlmError>> {
    let futures: Vec<_> = tasks.into_iter()
        .map(|task| {
            let client = client.clone();
            async move {
                // Retry logic for each task
                for attempt in 1..=3 {
                    match process_single_task(client.as_ref(), &task).await {
                        Ok(result) => return Ok(result),
                        Err(e) if attempt < 3 => {
                            // Wait before retry
                            sleep(Duration::from_millis(100 * attempt as u64)).await;
                            continue;
                        }
                        Err(e) => return Err(e),
                    }
                }
                unreachable!()
            }
        })
        .collect();

    futures::future::join_all(futures).await
}

/// Process a chunk of tasks
async fn process_chunk(
    client: Arc<dyn ChatCapability>,
    tasks: Vec<String>,
) -> Vec<Result<String, LlmError>> {
    let futures: Vec<_> = tasks.into_iter()
        .map(|task| {
            let client = client.clone();
            async move {
                process_single_task(client.as_ref(), &task).await
            }
        })
        .collect();

    futures::future::join_all(futures).await
}

/*
🎯 Key Batch Processing Concepts:

Concurrency Control:
- Use Semaphore to limit concurrent requests
- Buffer streams to control memory usage
- Implement proper backpressure handling

Rate Limiting:
- Respect provider rate limits
- Add delays between requests
- Use exponential backoff for retries
- Monitor and adjust based on errors

Error Handling:
- Retry transient errors with backoff
- Categorize and handle different error types
- Implement circuit breakers for persistent failures
- Log errors for monitoring and debugging

Memory Management:
- Process in chunks for large datasets
- Use streaming instead of collecting all results
- Clean up resources promptly
- Monitor memory usage in production

Performance Optimization:
- Balance concurrency vs. rate limits
- Use connection pooling when available
- Implement caching for repeated requests
- Monitor and optimize based on metrics

Best Practices:
1. Start with conservative rate limits
2. Implement comprehensive error handling
3. Monitor progress and performance
4. Use chunking for very large batches
5. Respect provider terms of service
6. Implement proper logging and monitoring

Production Considerations:
- Cost monitoring and budgets
- Error alerting and escalation
- Performance metrics and SLAs
- Graceful shutdown and cleanup
- Resource scaling and management

Next Steps:
- custom_configurations.rs: Optimize for specific use cases
- ../04_providers/: Provider-specific batch optimizations
- ../05_use_cases/: Real-world batch processing applications
*/

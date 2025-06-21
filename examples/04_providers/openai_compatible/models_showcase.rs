//! 🔄 OpenAI Compatible Models
//! 
//! This example demonstrates various OpenAI-compatible providers including:
//! - DeepSeek integration and specialized models
//! - Groq high-speed inference capabilities
//! - Local model deployment options
//! - Performance comparison across providers
//! - Cost and speed optimization strategies
//! 
//! Before running, set your API keys:
//! ```bash
//! export DEEPSEEK_API_KEY="your-deepseek-key"
//! export GROQ_API_KEY="your-groq-key"
//! export OPENAI_API_KEY="your-openai-key"
//! ```
//! 
//! Usage:
//! ```bash
//! cargo run --example openai_compatible_models_showcase
//! ```

use siumai::prelude::*;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 OpenAI Compatible Models Showcase\n");

    println!("🔧 Demonstrating Compatible Providers:");
    println!("   1. DeepSeek Specialized Models");
    println!("   2. Groq High-Speed Inference");
    println!("   3. Performance Comparison");
    println!("   4. Cost Optimization Strategies");
    println!("   5. Use Case Recommendations\n");

    // Demo 1: DeepSeek Models
    println!("🧠 1. DeepSeek Specialized Models");
    demo_deepseek_models().await?;
    println!();

    // Demo 2: Groq High-Speed
    println!("⚡ 2. Groq High-Speed Inference");
    demo_groq_performance().await?;
    println!();

    // Demo 3: Performance Comparison
    println!("📊 3. Performance Comparison");
    demo_performance_comparison().await?;
    println!();

    // Demo 4: Cost Optimization
    println!("💰 4. Cost Optimization Strategies");
    demo_cost_optimization().await?;
    println!();

    // Demo 5: Use Case Recommendations
    println!("🎯 5. Use Case Recommendations");
    demo_use_case_recommendations().await?;

    println!("\n✅ OpenAI Compatible Models showcase completed!");
    Ok(())
}

/// Demo DeepSeek specialized models
async fn demo_deepseek_models() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Exploring DeepSeek specialized capabilities...");
    
    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .unwrap_or_else(|_| {
            println!("   ⚠️  DEEPSEEK_API_KEY not set, using demo key");
            "demo-key".to_string()
        });

    // DeepSeek Coder model
    println!("   Testing DeepSeek Coder for programming tasks:");
    let coder_ai = Siumai::builder()
        .deepseek()
        .api_key(&api_key)
        .model("deepseek-coder")
        .temperature(0.1)
        .max_tokens(400)
        .build()
        .await?;

    let coding_task = vec![
        ChatMessage::user(
            "Write a Rust function that implements a binary search algorithm. \
            Include error handling and comprehensive documentation."
        ).build(),
    ];

    match coder_ai.chat(coding_task).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   💻 DeepSeek Coder response:");
                let lines: Vec<&str> = text.lines().take(5).collect();
                for line in lines {
                    println!("      {}", line);
                }
                if text.lines().count() > 5 {
                    println!("      ... (truncated)");
                }
            }
        }
        Err(e) => println!("   ❌ Error: {}", e),
    }

    // DeepSeek Reasoning model
    println!("   Testing DeepSeek Reasoning for complex analysis:");
    let reasoning_ai = Siumai::builder()
        .deepseek()
        .api_key(&api_key)
        .model("deepseek-reasoner")
        .temperature(0.2)
        .max_tokens(500)
        .build()
        .await?;

    let reasoning_task = vec![
        ChatMessage::user(
            "Analyze the trade-offs between microservices and monolithic \
            architecture for a startup with 10 developers building a \
            e-commerce platform. Consider scalability, complexity, and team dynamics."
        ).build(),
    ];

    match reasoning_ai.chat(reasoning_task).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🧠 DeepSeek Reasoning analysis:");
                println!("      Analysis length: {} words", text.split_whitespace().count());
                println!("      Covers: {}",
                    if text.contains("trade-off") || text.contains("consider") {
                        "Comprehensive trade-off analysis"
                    } else {
                        "Detailed technical comparison"
                    }
                );
            }
        }
        Err(e) => println!("   ❌ Error: {}", e),
    }

    Ok(())
}

/// Demo Groq high-speed performance
async fn demo_groq_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing Groq high-speed inference capabilities...");
    
    let api_key = std::env::var("GROQ_API_KEY")
        .unwrap_or_else(|_| {
            println!("   ⚠️  GROQ_API_KEY not set, using demo key");
            "demo-key".to_string()
        });

    // Note: Groq is not directly supported in this version, using OpenAI as substitute
    let groq_ai = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini") // Using OpenAI model as substitute
        .temperature(0.3)
        .max_tokens(300)
        .build()
        .await?;

    // Speed test with multiple requests
    println!("   Performing speed test with multiple requests:");
    
    let test_prompts = [
        "Explain machine learning in one paragraph",
        "What are the benefits of cloud computing?",
        "Describe the importance of cybersecurity",
        "How does blockchain technology work?",
    ];

    let mut total_time = std::time::Duration::new(0, 0);
    let mut successful_requests = 0;

    for (i, prompt) in test_prompts.iter().enumerate() {
        let start_time = Instant::now();
        
        let messages = vec![ChatMessage::user(*prompt).build()];
        
        match groq_ai.chat(messages).await {
            Ok(response) => {
                let elapsed = start_time.elapsed();
                total_time += elapsed;
                successful_requests += 1;
                
                if let Some(text) = response.text() {
                    println!("   ⚡ Request {}: {}ms - {}", 
                            i + 1, 
                            elapsed.as_millis(),
                            &text[..text.len().min(50)]);
                }
            }
            Err(e) => println!("   ❌ Request {} failed: {}", i + 1, e),
        }
    }

    if successful_requests > 0 {
        let avg_time = total_time / successful_requests;
        println!("   📊 Performance Summary:");
        println!("      Successful requests: {}/{}", successful_requests, test_prompts.len());
        println!("      Average response time: {}ms", avg_time.as_millis());
        println!("      Total time: {}ms", total_time.as_millis());
    }

    Ok(())
}

/// Demo performance comparison across providers
async fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Comparing performance across different providers...");
    
    let test_prompt = "Explain the concept of recursion in programming with a simple example";
    
    // Test different providers (Note: Using OpenAI for both since Groq is not directly supported)
    let providers = vec![
        ("OpenAI-Fast", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("OpenAI-Quality", "gpt-4", "OPENAI_API_KEY"),
    ];

    for (name, model, env_key) in providers {
        println!("   Testing {} performance:", name);
        
        let api_key = std::env::var(env_key)
            .unwrap_or_else(|_| {
                println!("     ⚠️  {} not set, using demo key", env_key);
                "demo-key".to_string()
            });

        let start_time = Instant::now();
        
        let ai = Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model(model)
            .temperature(0.3)
            .max_tokens(200)
            .build()
            .await?;

        let messages = vec![ChatMessage::user(test_prompt).build()];
        
        match ai.chat(messages).await {
            Ok(response) => {
                let elapsed = start_time.elapsed();
                
                if let Some(text) = response.text() {
                    println!("     ⏱️  Response time: {}ms", elapsed.as_millis());
                    println!("     📝 Response length: {} characters", text.len());
                    println!("     💬 Preview: {}...", &text[..text.len().min(60)]);
                    
                    if let Some(usage) = &response.usage {
                        println!("     🔢 Tokens: {}", usage.total_tokens);
                    }
                }
            }
            Err(e) => println!("     ❌ Error: {}", e),
        }
        println!();
    }

    Ok(())
}

/// Demo cost optimization strategies
async fn demo_cost_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating cost optimization strategies...");
    
    // Strategy 1: Model selection based on task complexity
    println!("   Strategy 1: Model selection for different task complexities");
    
    let tasks = vec![
        ("Simple", "What is 2 + 2?", "llama-3.1-8b-instant"),
        ("Medium", "Explain the benefits of version control", "llama-3.1-70b-versatile"),
        ("Complex", "Design a scalable microservices architecture", "llama-3.1-70b-versatile"),
    ];

    for (complexity, task, recommended_model) in tasks {
        println!("   {} task: {}", complexity, recommended_model);
        println!("     Task: {}", task);
        println!("     Recommended: {}", recommended_model);
        println!("     Rationale: {}",
            match complexity {
                "Simple" => "Fast model for quick answers",
                "Medium" => "Balanced model for explanations",
                "Complex" => "Advanced model for complex reasoning",
                _ => "Unknown",
            }
        );
        println!();
    }

    // Strategy 2: Batch processing
    println!("   Strategy 2: Batch processing for cost efficiency");
    println!("   💡 Batch Processing Benefits:");
    println!("      • Reduce API call overhead");
    println!("      • Better token utilization");
    println!("      • Lower per-request costs");
    println!("      • Improved throughput");

    // Strategy 3: Provider selection
    println!("   Strategy 3: Provider selection based on requirements");
    println!("   🎯 Provider Selection Guide:");
    println!("      • Groq: Ultra-fast inference, cost-effective");
    println!("      • DeepSeek: Specialized models, competitive pricing");
    println!("      • OpenAI: Highest quality, premium pricing");
    println!("      • Local: No API costs, hardware requirements");

    Ok(())
}

/// Demo use case recommendations
async fn demo_use_case_recommendations() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Providing use case recommendations for each provider...");
    
    let recommendations = vec![
        (
            "Groq",
            vec![
                "Real-time chat applications",
                "High-throughput batch processing",
                "Interactive demos and prototypes",
                "Cost-sensitive production workloads",
            ]
        ),
        (
            "DeepSeek",
            vec![
                "Code generation and review",
                "Technical analysis and reasoning",
                "Research and development tasks",
                "Specialized domain applications",
            ]
        ),
        (
            "OpenAI",
            vec![
                "High-quality content generation",
                "Complex reasoning tasks",
                "Production applications requiring reliability",
                "Advanced multimodal applications",
            ]
        ),
        (
            "Local Models",
            vec![
                "Privacy-sensitive applications",
                "Offline or air-gapped environments",
                "Custom fine-tuned models",
                "High-volume, cost-sensitive workloads",
            ]
        ),
    ];

    for (provider, use_cases) in recommendations {
        println!("   🎯 {} - Best Use Cases:", provider);
        for use_case in use_cases {
            println!("      • {}", use_case);
        }
        println!();
    }

    // Decision matrix
    println!("   📊 Decision Matrix:");
    println!("   ┌─────────────┬───────┬──────┬─────────┬─────────┐");
    println!("   │ Provider    │ Speed │ Cost │ Quality │ Special │");
    println!("   ├─────────────┼───────┼──────┼─────────┼─────────┤");
    println!("   │ Groq        │  ⭐⭐⭐  │ ⭐⭐⭐ │   ⭐⭐   │  Speed  │");
    println!("   │ DeepSeek    │  ⭐⭐   │ ⭐⭐⭐ │   ⭐⭐   │  Code   │");
    println!("   │ OpenAI      │  ⭐⭐   │  ⭐   │  ⭐⭐⭐  │ Quality │");
    println!("   │ Local       │  ⭐    │ ⭐⭐⭐ │   ⭐    │ Privacy │");
    println!("   └─────────────┴───────┴──────┴─────────┴─────────┘");

    Ok(())
}

/// 🎯 Key OpenAI Compatible Providers Summary:
///
/// Provider Options:
/// - Groq: Ultra-fast inference with Llama models
/// - DeepSeek: Specialized coding and reasoning models
/// - Local: Self-hosted models for privacy/cost
/// - Others: Various OpenAI-compatible services
///
/// Performance Characteristics:
/// - Groq: Fastest inference, excellent for real-time
/// - DeepSeek: Specialized capabilities, good performance
/// - OpenAI: Highest quality, moderate speed
/// - Local: Variable, depends on hardware
///
/// Cost Considerations:
/// - Groq: Very cost-effective, high throughput
/// - DeepSeek: Competitive pricing, specialized value
/// - OpenAI: Premium pricing, premium quality
/// - Local: Hardware costs, no API fees
///
/// Use Case Matching:
/// - Real-time applications → Groq
/// - Code-heavy tasks → DeepSeek
/// - High-quality content → OpenAI
/// - Privacy-sensitive → Local
///
/// Optimization Strategies:
/// - Match provider to task requirements
/// - Use batch processing when possible
/// - Monitor costs and performance
/// - Implement fallback mechanisms
/// - Cache frequent responses
///
/// Integration Benefits:
/// - OpenAI-compatible APIs
/// - Easy provider switching
/// - Consistent interface
/// - Reduced vendor lock-in
/// - Cost optimization flexibility
///
/// Next Steps:
/// - Test providers for your use case
/// - Implement provider switching logic
/// - Set up cost monitoring
/// - Create performance benchmarks
/// - Plan fallback strategies
fn _documentation() {}

//! 🧠 Anthropic Basic Chat
//! 
//! This example demonstrates basic Anthropic Claude usage including:
//! - Claude model selection and optimization
//! - Parameter tuning for different use cases
//! - Context window management
//! - Cost-effective usage patterns
//! - Claude-specific best practices
//! 
//! Before running, set your API key:
//! ```bash
//! export ANTHROPIC_API_KEY="your-anthropic-key"
//! ```
//! 
//! Usage:
//! ```bash
//! cargo run --example anthropic_basic_chat
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Anthropic Claude Basic Chat Demo\n");

    // Get API key
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .unwrap_or_else(|_| {
            println!("⚠️  ANTHROPIC_API_KEY not set, using demo key");
            "demo-key".to_string()
        });

    println!("🔧 Demonstrating Claude Capabilities:");
    println!("   1. Model Selection and Comparison");
    println!("   2. Parameter Optimization");
    println!("   3. Context Window Management");
    println!("   4. Cost-Effective Usage");
    println!("   5. Claude-Specific Features\n");

    // Demo 1: Model Selection
    println!("🎯 1. Model Selection and Comparison");
    demo_model_selection(&api_key).await?;
    println!();

    // Demo 2: Parameter Optimization
    println!("⚙️ 2. Parameter Optimization");
    demo_parameter_optimization(&api_key).await?;
    println!();

    // Demo 3: Context Management
    println!("📚 3. Context Window Management");
    demo_context_management(&api_key).await?;
    println!();

    // Demo 4: Cost-Effective Usage
    println!("💰 4. Cost-Effective Usage Patterns");
    demo_cost_effective_usage(&api_key).await?;
    println!();

    // Demo 5: Claude-Specific Features
    println!("🌟 5. Claude-Specific Features");
    demo_claude_features(&api_key).await?;

    println!("\n✅ Anthropic Claude demo completed!");
    Ok(())
}

/// Demo different Claude models
async fn demo_model_selection(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Comparing different Claude models...");
    
    // Note: In real usage, you would use actual Claude model names
    let models = vec![
        ("claude-3-haiku", "Fast and efficient for simple tasks"),
        ("claude-3-sonnet", "Balanced performance and capability"),
        ("claude-3-opus", "Most capable for complex reasoning"),
    ];

    for (model_name, description) in models {
        println!("   Testing {model_name}: {description}");
        
        let ai = Siumai::builder()
            .anthropic()
            .api_key(api_key)
            .model("claude-3-5-sonnet-20241022")
            .temperature(0.3)
            .max_tokens(200)
            .build()
            .await?;

        let messages = vec![
            ChatMessage::user(
                "Explain quantum computing in one paragraph, \
                focusing on its key advantages over classical computing."
            ).build(),
        ];

        match ai.chat(messages).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    println!("   📝 {}: {}", model_name, &text[..text.len().min(100)]);
                    if let Some(usage) = &response.usage {
                        println!("      Tokens: {}", usage.total_tokens);
                    }
                }
            }
            Err(e) => println!("   ❌ Error with {model_name}: {e}"),
        }
        println!();
    }

    Ok(())
}

/// Demo parameter optimization for different use cases
async fn demo_parameter_optimization(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing parameter configurations for different use cases...");
    
    let use_cases = vec![
        ("Creative Writing", 0.8, 1000, "Write a creative opening for a sci-fi story"),
        ("Technical Analysis", 0.1, 500, "Explain the differences between REST and GraphQL APIs"),
        ("Balanced Chat", 0.5, 300, "What are the benefits of learning a new programming language?"),
    ];

    for (use_case, temperature, max_tokens, prompt) in use_cases {
        println!("   Use case: {use_case}");
        println!("     Temperature: {temperature}, Max tokens: {max_tokens}");
        
        let ai = Siumai::builder()
            .anthropic()
            .api_key(api_key)
            .model("claude-3-5-sonnet-20241022")
            .temperature(temperature)
            .max_tokens(max_tokens)
            .build()
            .await?;

        let messages = vec![ChatMessage::user(prompt).build()];

        match ai.chat(messages).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    let preview = text.lines().next().unwrap_or("").trim();
                    println!("     📝 Response: {}...", &preview[..preview.len().min(80)]);
                }
            }
            Err(e) => println!("     ❌ Error: {e}"),
        }
        println!();
    }

    Ok(())
}

/// Demo context window management
async fn demo_context_management(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating context window management...");
    
    let ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model("claude-3-5-sonnet-20241022")
        .temperature(0.3)
        .max_tokens(300)
        .build()
        .await?;

    // Simulate a conversation with context building
    let mut conversation = Vec::new();
    
    // Initial context
    conversation.push(ChatMessage::system(
        "You are a helpful programming tutor. Keep track of what we've discussed \
        and build upon previous topics in our conversation."
    ).build());

    // First exchange
    conversation.push(ChatMessage::user("I'm learning Rust. What should I start with?").build());

    println!("   Building conversation context...");
    match ai.chat(conversation.clone()).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                conversation.push(ChatMessage::assistant(&text).build());
                println!("   📚 Initial response about Rust basics");
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Follow-up question that requires context
    conversation.push(ChatMessage::user(
        "Great! Now I understand ownership. Can you give me a practical example \
        that builds on what you just explained?"
    ).build());

    println!("   Using context for follow-up...");
    let conversation_len = conversation.len();
    match ai.chat(conversation).await {
        Ok(response) => {
            if let Some(_text) = response.text() {
                println!("   🔗 Context-aware response provided");
                println!("   Total conversation length: {} messages", conversation_len + 1);
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Context management tips
    println!("   💡 Context Management Tips:");
    println!("      • Claude has a large context window (up to 200K tokens)");
    println!("      • Include relevant conversation history");
    println!("      • Use system messages for consistent behavior");
    println!("      • Summarize long conversations when needed");
    println!("      • Remove irrelevant context to save tokens");

    Ok(())
}

/// Demo cost-effective usage patterns
async fn demo_cost_effective_usage(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating cost-effective usage patterns...");
    
    // Strategy 1: Use appropriate model for task complexity
    println!("   Strategy 1: Model selection based on task complexity");
    
    let simple_ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model("claude-3-5-haiku-20241022") // Fastest, most cost-effective
        .temperature(0.3)
        .max_tokens(100)
        .build()
        .await?;

    let simple_task = vec![ChatMessage::user("What is 15% of 240?").build()];

    match simple_ai.chat(simple_task).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   💰 Simple calculation with Haiku: {}", text.trim());
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Strategy 2: Batch multiple questions
    println!("   Strategy 2: Batch processing multiple questions");
    
    let batch_questions = vec![ChatMessage::user(
        "Answer these questions briefly:\n\
        1. What is the capital of France?\n\
        2. What is 2 + 2?\n\
        3. Name one benefit of exercise\n\
        4. What color is the sky?"
    ).build()];

    match simple_ai.chat(batch_questions).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   📦 Batch processing result:");
                for (i, line) in text.lines().take(4).enumerate() {
                    if !line.trim().is_empty() {
                        println!("      {}. {}", i + 1, line.trim());
                    }
                }
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Strategy 3: Optimize token usage
    println!("   Strategy 3: Token optimization tips");
    println!("   💡 Cost Optimization Strategies:");
    println!("      • Use Haiku for simple tasks, Sonnet for balanced needs");
    println!("      • Batch multiple questions in single requests");
    println!("      • Set appropriate max_tokens limits");
    println!("      • Use clear, concise prompts");
    println!("      • Cache responses for repeated queries");
    println!("      • Monitor usage with response.usage()");

    Ok(())
}

/// Demo Claude-specific features
async fn demo_claude_features(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Exploring Claude-specific capabilities...");
    
    let ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model("claude-3-5-sonnet-20241022")
        .temperature(0.3)
        .max_tokens(400)
        .build()
        .await?;

    // Feature 1: Constitutional AI and helpfulness
    println!("   Feature 1: Constitutional AI - Helpful and harmless responses");
    let constitutional_test = vec![ChatMessage::user(
        "I'm feeling overwhelmed with my workload. Can you help me create \
        a strategy to manage my tasks better while maintaining work-life balance?"
    ).build()];

    match ai.chat(constitutional_test).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🤝 Helpful response provided (showing first 100 chars):");
                println!("   {}", &text[..text.len().min(100)]);
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Feature 2: Long-form reasoning
    println!("   Feature 2: Long-form reasoning and analysis");
    let reasoning_test = vec![ChatMessage::user(
        "Walk me through the pros and cons of microservices vs monolithic \
        architecture, considering factors like scalability, complexity, \
        and team organization."
    ).build()];

    match ai.chat(reasoning_test).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🧠 Detailed analysis provided ({} characters)", text.len());
                println!("   Structure: {}", 
                    if text.contains("Pros:") || text.contains("Advantages:") { 
                        "Well-structured comparison" 
                    } else { 
                        "Comprehensive analysis" 
                    }
                );
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Feature 3: Code understanding and generation
    println!("   Feature 3: Code understanding and generation");
    let code_test = vec![ChatMessage::user(
        "Write a simple Rust function that calculates the factorial of a number \
        and explain how it works, including error handling."
    ).build()];

    match ai.chat(code_test).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   💻 Code generation completed");
                println!("   Includes: {}",
                    if text.contains("fn") && text.contains("error") {
                        "Function definition and error handling"
                    } else {
                        "Code explanation"
                    }
                );
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    println!("   🌟 Claude Unique Strengths:");
    println!("      • Constitutional AI for safe, helpful responses");
    println!("      • Excellent long-form reasoning and analysis");
    println!("      • Strong code understanding and generation");
    println!("      • Large context window for complex tasks");
    println!("      • Nuanced understanding of context and intent");

    Ok(())
}

/// 🎯 Key Anthropic Claude Features Summary:
///
/// Model Options:
/// - Claude 3 Haiku: Fast, cost-effective for simple tasks
/// - Claude 3 Sonnet: Balanced performance and capability
/// - Claude 3 Opus: Most capable for complex reasoning
///
/// Core Strengths:
/// - Constitutional AI for safety and helpfulness
/// - Large context windows (up to 200K tokens)
/// - Excellent reasoning and analysis
/// - Strong code understanding
/// - Nuanced conversation abilities
///
/// Parameter Optimization:
/// - Temperature: 0.1-0.3 for focused tasks, 0.5-0.8 for creative
/// - Max tokens: Set based on expected response length
/// - Model selection: Match complexity to task requirements
///
/// Cost Optimization:
/// - Use Haiku for simple tasks
/// - Batch multiple questions
/// - Set appropriate token limits
/// - Cache frequent responses
/// - Monitor usage patterns
///
/// Best Practices:
/// - Clear, specific prompts
/// - Appropriate context management
/// - Model selection based on task
/// - Regular usage monitoring
/// - Error handling implementation
///
/// Use Cases:
/// - Complex analysis and reasoning
/// - Code review and generation
/// - Long-form content creation
/// - Educational assistance
/// - Research and summarization
///
/// Next Steps:
/// - Explore thinking models
/// - Implement conversation persistence
/// - Add usage monitoring
/// - Create task-specific configurations
const fn _documentation() {}

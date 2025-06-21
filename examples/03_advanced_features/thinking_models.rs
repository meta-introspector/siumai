//! 🧠 Thinking Models - AI reasoning and thinking process
//!
//! This example demonstrates how to work with AI models that show their reasoning:
//! - Accessing thinking processes from models like Claude
//! - Working with reasoning models like o1
//! - Optimizing prompts for reasoning tasks
//! - Understanding reasoning vs. final output
//!
//! Before running, set your API keys:
//! ```bash
//! export ANTHROPIC_API_KEY="your-key"  # For Claude thinking
//! export OPENAI_API_KEY="your-key"     # For o1 reasoning models
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example thinking_models
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Thinking Models - AI reasoning and thinking process\n");

    // Demonstrate different aspects of thinking models
    demonstrate_claude_thinking().await;
    demonstrate_reasoning_optimization().await;
    demonstrate_thinking_analysis().await;
    demonstrate_reasoning_vs_output().await;
    demonstrate_thinking_best_practices().await;

    println!("\n✅ Thinking models examples completed!");
    Ok(())
}

/// Demonstrate Claude's thinking process
async fn demonstrate_claude_thinking() {
    println!("🤔 Claude Thinking Process:\n");

    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        match LlmBuilder::new()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-sonnet-20241022")
            .temperature(0.7)
            .build()
            .await
        {
            Ok(client) => {
                let messages = vec![
                    system!(
                        "Think step by step about the problem. Show your reasoning process clearly."
                    ),
                    user!(
                        "A farmer has 17 sheep. All but 9 die. How many sheep are left? Think through this carefully."
                    ),
                ];

                match client.chat(messages).await {
                    Ok(response) => {
                        println!(
                            "   Problem: A farmer has 17 sheep. All but 9 die. How many sheep are left?"
                        );

                        // Check if thinking process is available
                        if let Some(thinking) = &response.thinking {
                            println!("\n   🧠 Claude's Thinking Process:");
                            println!("   {thinking}");
                            println!("\n   📝 Final Answer:");
                        } else {
                            println!("\n   📝 Response (thinking not available):");
                        }

                        if let Some(text) = response.content_text() {
                            println!("   {text}");
                        }

                        println!("   ✅ Claude thinking demonstration successful");
                    }
                    Err(e) => {
                        println!("   ❌ Claude thinking failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Failed to create Claude client: {e}");
            }
        }
    } else {
        println!("   ⚠️  ANTHROPIC_API_KEY not set, skipping Claude thinking example");
    }

    println!();
}

/// Demonstrate reasoning optimization techniques
async fn demonstrate_reasoning_optimization() {
    println!("⚡ Reasoning Optimization:\n");

    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        match LlmBuilder::new()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-sonnet-20241022")
            .temperature(0.1) // Lower temperature for more consistent reasoning
            .build()
            .await
        {
            Ok(client) => {
                // Test different reasoning prompts
                let reasoning_prompts = vec![
                    ("Basic", "Solve this math problem: What is 15% of 240?"),
                    (
                        "Step-by-step",
                        "Solve this math problem step by step: What is 15% of 240? Show each calculation.",
                    ),
                    (
                        "Chain of thought",
                        "Let's think about this step by step. What is 15% of 240? First, I need to understand what 15% means...",
                    ),
                ];

                for (approach, prompt) in reasoning_prompts {
                    println!("   Approach: {approach}");
                    println!("   Prompt: {prompt}");

                    let messages = vec![user!(prompt)];

                    match client.chat(messages).await {
                        Ok(response) => {
                            if let Some(text) = response.content_text() {
                                let preview = if text.len() > 200 {
                                    format!("{}...", &text[..200])
                                } else {
                                    text.to_string()
                                };
                                println!("   Response: {preview}");
                            }

                            if let Some(thinking) = &response.thinking {
                                println!("   Thinking length: {} characters", thinking.len());
                            }

                            println!("   ✅ Success");
                        }
                        Err(e) => {
                            println!("   ❌ Failed: {e}");
                        }
                    }
                    println!();
                }
            }
            Err(e) => {
                println!("   ❌ Failed to create client: {e}");
            }
        }
    } else {
        println!("   ⚠️  ANTHROPIC_API_KEY not set, skipping reasoning optimization");
    }
}

/// Demonstrate thinking process analysis
async fn demonstrate_thinking_analysis() {
    println!("🔍 Thinking Process Analysis:\n");

    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        match LlmBuilder::new()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-sonnet-20241022")
            .build()
            .await
        {
            Ok(client) => {
                let complex_problem = "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons of water? Think through different approaches.";

                let messages = vec![
                    system!(
                        "Think through this problem carefully, considering multiple approaches and potential solutions."
                    ),
                    user!(complex_problem),
                ];

                match client.chat(messages).await {
                    Ok(response) => {
                        println!("   Problem: {complex_problem}");

                        if let Some(thinking) = &response.thinking {
                            println!("\n   🧠 Thinking Analysis:");

                            // Analyze thinking process
                            let thinking_stats = analyze_thinking_process(thinking);
                            println!("      Length: {} characters", thinking_stats.length);
                            println!("      Steps identified: {}", thinking_stats.steps);
                            println!("      Questions asked: {}", thinking_stats.questions);
                            println!("      Approaches considered: {}", thinking_stats.approaches);

                            // Show first part of thinking
                            let preview = if thinking.len() > 300 {
                                format!("{}...", &thinking[..300])
                            } else {
                                thinking.clone()
                            };
                            println!("\n   🔍 Thinking Preview:");
                            println!("   {preview}");
                        }

                        if let Some(text) = response.content_text() {
                            println!("\n   📝 Final Solution:");
                            println!("   {text}");
                        }

                        println!("\n   ✅ Thinking analysis completed");
                    }
                    Err(e) => {
                        println!("   ❌ Analysis failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Failed to create client: {e}");
            }
        }
    } else {
        println!("   ⚠️  ANTHROPIC_API_KEY not set, skipping thinking analysis");
    }

    println!();
}

/// Demonstrate reasoning vs. output comparison
async fn demonstrate_reasoning_vs_output() {
    println!("⚖️  Reasoning vs. Output Comparison:\n");

    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        match LlmBuilder::new()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-sonnet-20241022")
            .build()
            .await
        {
            Ok(client) => {
                let problem = "Should a company prioritize short-term profits or long-term sustainability? Consider multiple perspectives.";

                let messages = vec![user!(problem)];

                match client.chat(messages).await {
                    Ok(response) => {
                        println!("   Question: {problem}");

                        if let Some(thinking) = &response.thinking {
                            println!("\n   🧠 Internal Reasoning:");
                            println!("      - Considers multiple perspectives");
                            println!("      - Weighs pros and cons");
                            println!("      - Explores nuances");
                            println!("      - Length: {} characters", thinking.len());

                            // Extract key reasoning elements
                            let reasoning_elements = extract_reasoning_elements(thinking);
                            println!(
                                "      - Perspectives considered: {}",
                                reasoning_elements.perspectives
                            );
                            println!("      - Arguments made: {}", reasoning_elements.arguments);
                        }

                        if let Some(text) = response.content_text() {
                            println!("\n   📝 Public Output:");
                            println!("      - Balanced presentation");
                            println!("      - Clear conclusion");
                            println!("      - Length: {} characters", text.len());

                            let preview = if text.len() > 200 {
                                format!("{}...", &text[..200])
                            } else {
                                text.to_string()
                            };
                            println!("      Preview: {preview}");
                        }

                        println!(
                            "\n   💡 Key Insight: Thinking shows the reasoning process, output shows the refined conclusion"
                        );
                        println!("   ✅ Comparison completed");
                    }
                    Err(e) => {
                        println!("   ❌ Comparison failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Failed to create client: {e}");
            }
        }
    } else {
        println!("   ⚠️  ANTHROPIC_API_KEY not set, skipping reasoning vs output comparison");
    }

    println!();
}

/// Demonstrate thinking model best practices
async fn demonstrate_thinking_best_practices() {
    println!("💡 Thinking Model Best Practices:\n");

    println!("   🎯 Prompt Design for Thinking Models:");
    println!("      ✅ Use 'Think step by step' or similar phrases");
    println!("      ✅ Ask for explicit reasoning");
    println!("      ✅ Request consideration of multiple approaches");
    println!("      ✅ Use lower temperature for consistent reasoning");
    println!("      ❌ Don't rush the model with time pressure");
    println!("      ❌ Don't ask for reasoning in the output if thinking is available");

    println!("\n   🔧 Technical Considerations:");
    println!("      • Thinking content increases token usage");
    println!("      • Reasoning models may have higher latency");
    println!("      • Not all providers support thinking access");
    println!("      • Thinking quality varies by model and prompt");

    println!("\n   📊 Use Cases for Thinking Models:");
    println!("      • Complex problem solving");
    println!("      • Multi-step reasoning tasks");
    println!("      • Decision making with trade-offs");
    println!("      • Educational content creation");
    println!("      • Code debugging and analysis");

    println!("\n   ⚠️  Limitations:");
    println!("      • Higher computational cost");
    println!("      • Longer response times");
    println!("      • Thinking may contain errors");
    println!("      • Not suitable for simple tasks");

    println!("\n   ✅ Best practices overview completed");
}

/// Analyze thinking process structure
#[derive(Debug)]
struct ThinkingStats {
    length: usize,
    steps: usize,
    questions: usize,
    approaches: usize,
}

fn analyze_thinking_process(thinking: &str) -> ThinkingStats {
    let length = thinking.len();

    // Simple analysis - count patterns
    let steps = thinking.matches("step").count()
        + thinking.matches("first").count()
        + thinking.matches("then").count()
        + thinking.matches("next").count();

    let questions = thinking.matches("?").count();

    let approaches = thinking.matches("approach").count()
        + thinking.matches("method").count()
        + thinking.matches("way").count();

    ThinkingStats {
        length,
        steps,
        questions,
        approaches,
    }
}

/// Extract reasoning elements from thinking
#[derive(Debug)]
struct ReasoningElements {
    perspectives: usize,
    arguments: usize,
}

fn extract_reasoning_elements(thinking: &str) -> ReasoningElements {
    let perspectives = thinking.matches("perspective").count()
        + thinking.matches("viewpoint").count()
        + thinking.matches("side").count();

    let arguments = thinking.matches("argument").count()
        + thinking.matches("reason").count()
        + thinking.matches("because").count();

    ReasoningElements {
        perspectives,
        arguments,
    }
}

/*
🎯 Key Thinking Models Concepts:

Thinking vs. Output:
- Thinking: Internal reasoning process, raw thoughts
- Output: Refined, polished response for the user
- Thinking shows "how", output shows "what"

Model Types:
- Claude: Provides thinking process in separate field
- o1: Reasoning models with built-in chain of thought
- Other models: May show reasoning in output text

Optimization Strategies:
- Use explicit reasoning prompts
- Lower temperature for consistency
- Allow sufficient time/tokens for reasoning
- Structure prompts to encourage step-by-step thinking

Best Practices:
1. Design prompts specifically for reasoning
2. Monitor thinking quality and relevance
3. Consider cost implications of longer responses
4. Use thinking for complex tasks, not simple ones
5. Validate reasoning logic when possible

Applications:
- Complex problem solving
- Educational content
- Decision support systems
- Code analysis and debugging
- Research and analysis tasks

Next Steps:
- multimodal_processing.rs: Combine reasoning with other modalities
- batch_processing.rs: Scale reasoning tasks
- ../04_providers/: Provider-specific reasoning features
*/

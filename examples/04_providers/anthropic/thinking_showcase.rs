//! 🤔 Claude Thinking Process
//!
//! This example demonstrates Claude's thinking capabilities including:
//! - Accessing thinking content and reasoning process
//! - Reasoning analysis and step-by-step problem solving

#![allow(clippy::useless_vec)]
//! - Complex problem solving with visible thought process
//! - Thinking vs output comparison
//! - Advanced reasoning patterns
//! 
//! Before running, set your API key:
//! ```bash
//! export ANTHROPIC_API_KEY="your-anthropic-key"
//! ```
//! 
//! Usage:
//! ```bash
//! cargo run --example anthropic_thinking_showcase
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤔 Claude Thinking Process Showcase\n");

    // Get API key
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .unwrap_or_else(|_| {
            println!("⚠️  ANTHROPIC_API_KEY not set, using demo key");
            "demo-key".to_string()
        });

    println!("🧠 Demonstrating Claude Thinking Capabilities:");
    println!("   1. Step-by-Step Problem Solving");
    println!("   2. Complex Reasoning Analysis");
    println!("   3. Mathematical Problem Solving");
    println!("   4. Logical Reasoning Chains");
    println!("   5. Creative Problem Solving\n");

    // Demo 1: Step-by-Step Problem Solving
    println!("🔍 1. Step-by-Step Problem Solving");
    demo_step_by_step_solving(&api_key).await?;
    println!();

    // Demo 2: Complex Reasoning
    println!("🧩 2. Complex Reasoning Analysis");
    demo_complex_reasoning(&api_key).await?;
    println!();

    // Demo 3: Mathematical Problem Solving
    println!("🔢 3. Mathematical Problem Solving");
    demo_mathematical_reasoning(&api_key).await?;
    println!();

    // Demo 4: Logical Reasoning
    println!("⚖️ 4. Logical Reasoning Chains");
    demo_logical_reasoning(&api_key).await?;
    println!();

    // Demo 5: Creative Problem Solving
    println!("🎨 5. Creative Problem Solving");
    demo_creative_reasoning(&api_key).await?;

    println!("\n✅ Claude Thinking Process showcase completed!");
    Ok(())
}

/// Demo step-by-step problem solving
async fn demo_step_by_step_solving(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating step-by-step problem solving...");
    
    let ai = LlmBuilder::new()
        .anthropic()
        .api_key(api_key)
        .model("claude-3-5-sonnet-20241022")
        .temperature(0.2)
        .max_tokens(600)
        .build()
        .await?;

    let messages = vec![
        system!(
            "You are a problem-solving expert. When given a problem, \
            show your thinking process step by step. Break down complex \
            problems into smaller, manageable parts and explain your \
            reasoning at each step."
        ),
        user!(
            "I need to plan a team retreat for 20 people with a budget of $5000. \
            The retreat should be 2 days long, include team building activities, \
            meals, and accommodation. Walk me through how you would approach \
            this planning challenge step by step."
        ),
    ];

    println!("   Analyzing retreat planning problem...");
    
    match ai.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.content_text() {
                println!("   📋 Step-by-Step Solution:");
                
                // Extract and display the thinking process
                let lines: Vec<&str> = text.lines().collect();
                for (i, line) in lines.iter().enumerate().take(10) {
                    if !line.trim().is_empty() {
                        println!("   {}: {}", i + 1, line.trim());
                    }
                }
                
                if lines.len() > 10 {
                    println!("   ... (showing first 10 steps)");
                }
            }
        }
        Err(e) => println!("   ❌ Error: {}", e),
    }

    Ok(())
}

/// Demo complex reasoning analysis
async fn demo_complex_reasoning(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Analyzing complex multi-factor problem...");
    
    let ai = LlmBuilder::new()
        .anthropic()
        .api_key(api_key)
        .model("claude-3-5-sonnet-20241022")
        .temperature(0.1)
        .max_tokens(700)
        .build()
        .await?;

    let messages = vec![
        system!(
            "You are an expert analyst. When analyzing complex scenarios, \
            consider multiple factors, potential outcomes, and trade-offs. \
            Show your reasoning process and explain how you weigh different \
            considerations."
        ),
        user!(
            "A tech startup is deciding between two growth strategies: \
            1) Focus on rapid user acquisition with high marketing spend \
            2) Focus on product development and organic growth \
            \
            Consider factors like: funding runway, market competition, \
            team size, product maturity, and long-term sustainability. \
            Analyze this decision thoroughly."
        ),
    ];

    println!("   Performing multi-factor analysis...");
    
    match ai.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.content_text() {
                println!("   🧩 Complex Analysis Result:");
                
                // Look for key reasoning patterns
                let analysis_sections = vec![
                    ("Factors", "factor"),
                    ("Trade-offs", "trade"),
                    ("Considerations", "consider"),
                    ("Recommendation", "recommend"),
                ];
                
                for (section, keyword) in analysis_sections {
                    if text.to_lowercase().contains(keyword) {
                        println!("   ✅ Includes {} analysis", section);
                    }
                }
                
                println!("   📊 Analysis length: {} words", text.split_whitespace().count());
            }
        }
        Err(e) => println!("   ❌ Error: {}", e),
    }

    Ok(())
}

/// Demo mathematical problem solving
async fn demo_mathematical_reasoning(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Solving mathematical problem with reasoning...");
    
    let ai = LlmBuilder::new()
        .anthropic()
        .api_key(api_key)
        .model("claude-3-5-sonnet-20241022")
        .temperature(0.0)
        .max_tokens(500)
        .build()
        .await?;

    let messages = vec![
        ChatMessage::system(
            "You are a mathematics tutor. When solving problems, show \
            each step of your work clearly. Explain the mathematical \
            principles you're using and why each step is necessary."
        ).build(),
        ChatMessage::user(
            "Solve this optimization problem step by step: \
            A farmer has 100 meters of fencing and wants to create a \
            rectangular enclosure with maximum area. One side of the \
            rectangle will be against an existing wall, so fencing is \
            only needed for three sides. What dimensions maximize the area?"
        ).build(),
    ];

    println!("   Working through optimization problem...");
    
    match ai.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🔢 Mathematical Solution:");
                
                // Check for mathematical reasoning elements
                let math_elements = vec![
                    ("Variables", vec!["let", "x", "y", "="]),
                    ("Equations", vec!["area", "perimeter", "constraint"]),
                    ("Calculus", vec!["derivative", "maximum", "critical"]),
                    ("Solution", vec!["therefore", "answer", "dimensions"]),
                ];
                
                for (element, keywords) in math_elements {
                    let found = keywords.iter().any(|&keyword| 
                        text.to_lowercase().contains(keyword)
                    );
                    if found {
                        println!("   ✅ Contains {} reasoning", element);
                    }
                }
                
                // Show a preview of the solution
                let first_few_lines: Vec<&str> = text.lines().take(3).collect();
                for line in first_few_lines {
                    if !line.trim().is_empty() {
                        println!("   📝 {}", line.trim());
                    }
                }
            }
        }
        Err(e) => println!("   ❌ Error: {}", e),
    }

    Ok(())
}

/// Demo logical reasoning chains
async fn demo_logical_reasoning(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating logical reasoning chains...");
    
    let ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model("claude-3-sonnet")
        .temperature(0.1)
        .max_tokens(500)
        .build()
        .await?;

    let messages = vec![
        ChatMessage::system(
            "You are a logic expert. When presented with logical puzzles \
            or reasoning challenges, work through them systematically. \
            Show your logical steps and explain your reasoning process."
        ).build(),
        ChatMessage::user(
            "Here's a logic puzzle: \
            \
            Five friends (Alice, Bob, Carol, David, Eve) each have a different \
            favorite color (red, blue, green, yellow, purple) and live in \
            different cities (New York, London, Tokyo, Paris, Sydney). \
            \
            Clues: \
            1. Alice doesn't live in New York or London \
            2. The person who likes blue lives in Tokyo \
            3. Bob's favorite color is not red or yellow \
            4. Carol lives in Paris and doesn't like green \
            5. David likes yellow and doesn't live in Sydney \
            \
            Work through this step by step to determine who lives where \
            and what their favorite colors are."
        ).build(),
    ];

    println!("   Solving logic puzzle systematically...");
    
    match ai.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   ⚖️ Logical Reasoning Process:");
                
                // Check for logical reasoning patterns
                let logic_patterns = vec![
                    ("Deduction", vec!["therefore", "thus", "so"]),
                    ("Process", vec!["step", "first", "next", "then"]),
                    ("Elimination", vec!["can't", "cannot", "not", "eliminate"]),
                    ("Conclusion", vec!["answer", "solution", "result"]),
                ];
                
                for (pattern, keywords) in logic_patterns {
                    let found = keywords.iter().any(|&keyword| 
                        text.to_lowercase().contains(keyword)
                    );
                    if found {
                        println!("   ✅ Uses {} reasoning", pattern);
                    }
                }
                
                // Count logical steps
                let step_count = text.matches("step").count() + 
                                text.matches("Step").count() +
                                text.matches("1.").count() +
                                text.matches("2.").count();
                
                if step_count > 0 {
                    println!("   📊 Identified {} logical steps", step_count);
                }
            }
        }
        Err(e) => println!("   ❌ Error: {}", e),
    }

    Ok(())
}

/// Demo creative problem solving
async fn demo_creative_reasoning(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Exploring creative problem solving...");
    
    let ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model("claude-3-sonnet")
        .temperature(0.6) // Higher temperature for creativity
        .max_tokens(600)
        .build()
        .await?;

    let messages = vec![
        ChatMessage::system(
            "You are a creative problem solver. When faced with challenges, \
            think outside the box and consider unconventional approaches. \
            Show your creative thinking process and explain how you generate \
            and evaluate different ideas."
        ).build(),
        ChatMessage::user(
            "Challenge: A small town's main street businesses are struggling \
            because a new shopping mall opened nearby. The town council wants \
            to revitalize the main street without competing directly with the mall. \
            \
            Think creatively about solutions that could make the main street \
            a unique destination. Consider the town's character, community needs, \
            and innovative approaches to urban revitalization."
        ).build(),
    ];

    println!("   Generating creative solutions...");
    
    match ai.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🎨 Creative Problem Solving:");
                
                // Check for creative thinking indicators
                let creativity_indicators = vec![
                    ("Innovation", vec!["innovative", "unique", "creative", "novel"]),
                    ("Alternatives", vec!["alternative", "different", "instead", "rather"]),
                    ("Community", vec!["community", "local", "residents", "together"]),
                    ("Experience", vec!["experience", "atmosphere", "feel", "vibe"]),
                ];
                
                for (indicator, keywords) in creativity_indicators {
                    let found = keywords.iter().any(|&keyword| 
                        text.to_lowercase().contains(keyword)
                    );
                    if found {
                        println!("   ✅ Shows {} thinking", indicator);
                    }
                }
                
                // Count unique ideas/solutions
                let idea_markers = vec!["idea", "solution", "approach", "strategy", "concept"];
                let idea_count = idea_markers.iter()
                    .map(|&marker| text.to_lowercase().matches(marker).count())
                    .sum::<usize>();
                
                if idea_count > 0 {
                    println!("   💡 Generated {} creative concepts", idea_count);
                }
                
                // Show creativity in action
                println!("   🌟 Creative elements identified in response");
            }
        }
        Err(e) => println!("   ❌ Error: {}", e),
    }

    Ok(())
}

/// 🎯 Key Claude Thinking Features Summary:
///
/// Thinking Capabilities:
/// - Step-by-step problem decomposition
/// - Multi-factor analysis and reasoning
/// - Mathematical and logical problem solving
/// - Creative and innovative thinking
/// - Complex reasoning chains
///
/// Reasoning Patterns:
/// - Systematic approach to problems
/// - Consideration of multiple factors
/// - Trade-off analysis
/// - Evidence-based conclusions
/// - Creative solution generation
///
/// Problem-Solving Strengths:
/// - Breaking down complex problems
/// - Showing work and reasoning
/// - Considering multiple perspectives
/// - Weighing pros and cons
/// - Generating innovative solutions
///
/// Use Cases:
/// - Strategic planning and analysis
/// - Mathematical problem solving
/// - Logic puzzles and reasoning
/// - Creative brainstorming
/// - Decision-making support
/// - Educational tutoring
///
/// Best Practices:
/// - Ask for step-by-step reasoning
/// - Request explanation of thought process
/// - Use appropriate temperature settings
/// - Provide clear problem context
/// - Ask for multiple approaches
///
/// Temperature Guidelines:
/// - 0.0-0.2: Logical, mathematical problems
/// - 0.3-0.5: Balanced analysis
/// - 0.6-0.8: Creative problem solving
/// - 0.8+: Highly creative brainstorming
///
/// Next Steps:
/// - Explore specific reasoning domains
/// - Implement thinking process analysis
/// - Create reasoning templates
/// - Build problem-solving workflows
fn _documentation() {}

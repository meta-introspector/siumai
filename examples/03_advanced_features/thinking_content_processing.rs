//! Thinking Content Processing Example
//!
//! This example demonstrates how siumai handles `<think>` tags in responses,
//! which is commonly used by models like DeepSeek to separate reasoning from output.

use siumai::prelude::*;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🧠 Thinking Content Processing Example");
    println!("=====================================\n");

    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY")
        .or_else(|_| env::var("DEEPSEEK_API_KEY"))
        .expect("Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable");

    // Example 1: Simulate OpenAI-compatible provider with thinking content
    println!("📝 Example 1: OpenAI-compatible provider with <think> tags");
    println!("-----------------------------------------------------------");

    let _client = LlmBuilder::new()
        .openai()
        .api_key(&api_key)
        .model("gpt-4")
        .build()
        .await?;

    // Simulate a response that contains thinking tags
    // (In real usage, this would come from a model like DeepSeek)
    let simulated_response_with_thinking = r#"<think>
The user is asking about a mathematical problem. Let me break this down:
1. I need to calculate 15% of 240
2. 15% = 0.15
3. 0.15 × 240 = 36
4. So the answer is 36
</think>

To calculate 15% of 240:
15% × 240 = 0.15 × 240 = 36

The answer is 36."#;

    // Demonstrate the thinking content processing utilities
    println!("Original response with thinking tags:");
    println!("{}\n", simulated_response_with_thinking);

    // Test the utility functions
    use siumai::providers::openai::utils::*;

    // Check if content contains thinking tags
    let has_thinking = contains_thinking_tags(simulated_response_with_thinking);
    println!("Contains thinking tags: {}", has_thinking);

    // Extract thinking content
    if let Some(thinking) = extract_thinking_content(simulated_response_with_thinking) {
        println!("\n🤔 Extracted thinking content:");
        println!("{}", thinking);
    }

    // Filter out thinking content for display
    let filtered_content = filter_thinking_content(simulated_response_with_thinking);
    println!("\n📄 Filtered content (without thinking tags):");
    println!("{}", filtered_content);

    println!("\n{}", "=".repeat(60));

    // Example 2: DeepSeek-style thinking with Chinese content
    println!("\n📝 Example 2: DeepSeek-style thinking (Chinese)");
    println!("-----------------------------------------------");

    let deepseek_style_response = r#"<think>
用户询问了一个关于编程的问题。我需要：
1. 分析问题的核心
2. 提供清晰的解释
3. 给出实用的示例

这是一个关于Rust编程语言的问题，我应该提供准确和有用的信息。
</think>

这是一个很好的Rust编程问题！让我来详细解释一下：

Rust是一种系统编程语言，具有以下特点：
- 内存安全
- 零成本抽象
- 并发安全

这些特性使得Rust非常适合系统级编程。"#;

    println!("DeepSeek-style response:");
    println!("{}\n", deepseek_style_response);

    // Process the DeepSeek-style content
    if let Some(thinking) = extract_thinking_content(deepseek_style_response) {
        println!("🤔 Extracted thinking (Chinese):");
        println!("{}", thinking);
    }

    let filtered = filter_thinking_content(deepseek_style_response);
    println!("\n📄 Filtered response:");
    println!("{}", filtered);

    println!("\n{}", "=".repeat(60));

    // Example 3: Edge cases
    println!("\n📝 Example 3: Edge cases and error handling");
    println!("--------------------------------------------");

    let test_cases = vec![
        ("No thinking tags", "Just regular content without any tags"),
        (
            "Empty thinking",
            "<think></think>Content after empty thinking",
        ),
        (
            "Whitespace thinking",
            "<think>   </think>Content after whitespace",
        ),
        (
            "Multiple thinking blocks",
            "<think>First</think>Middle<think>Second</think>End",
        ),
        (
            "Incomplete thinking",
            "<think>Incomplete thinking without closing tag",
        ),
    ];

    for (description, content) in test_cases {
        println!("\nTest case: {}", description);
        println!("Content: {}", content);
        println!("Has thinking: {}", contains_thinking_tags(content));

        if let Some(thinking) = extract_thinking_content(content) {
            println!("Extracted: {}", thinking);
        } else {
            println!("Extracted: None");
        }

        let filtered = filter_thinking_content(content);
        println!("Filtered: {}", filtered);
        println!("{}", "-".repeat(40));
    }

    println!("\n✅ Thinking content processing examples completed!");
    println!("\nKey takeaways:");
    println!("- siumai automatically detects and processes <think> tags");
    println!("- Thinking content is extracted and made available separately");
    println!("- Main response content is filtered to remove thinking tags");
    println!("- Works with both English and Chinese content");
    println!("- Handles edge cases gracefully");

    Ok(())
}

//! Test Fixes Example
//!
//! This example tests that all the fixes are working correctly.

use siumai::{LlmBuilder, user, system, assistant, Provider};
use siumai::providers::openai_compatible::providers::{deepseek, openrouter, recommendations};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Fixes");
    println!("================");

    // Test 1: Model constants
    println!("\n1. Testing model constants:");
    println!("   ✓ DeepSeek Chat: {}", deepseek::CHAT);
    println!("   ✓ DeepSeek Coder: {}", deepseek::CODER);
    println!("   ✓ DeepSeek Reasoner: {}", deepseek::REASONER);
    println!("   ✓ DeepSeek ALL: {:?}", deepseek::ALL);

    println!("\n   ✓ OpenRouter GPT-4o: {}", openrouter::openai::GPT_4O);
    println!("   ✓ OpenRouter Claude: {}", openrouter::anthropic::CLAUDE_3_5_SONNET);
    println!("   ✓ OpenRouter Gemini: {}", openrouter::google::GEMINI_1_5_PRO);

    // Test 2: Recommendations
    println!("\n2. Testing recommendations:");
    println!("   ✓ For chat: {}", recommendations::for_chat());
    println!("   ✓ For coding: {}", recommendations::for_coding());
    println!("   ✓ For reasoning: {}", recommendations::for_reasoning());
    println!("   ✓ For fast response: {}", recommendations::for_fast_response());
    println!("   ✓ For cost-effective: {}", recommendations::for_cost_effective());
    println!("   ✓ For vision: {}", recommendations::for_vision());

    // Test 3: LlmBuilder static methods
    println!("\n3. Testing LlmBuilder static methods:");
    let _fast_builder = LlmBuilder::fast();
    println!("   ✓ LlmBuilder::fast() works");

    let _long_running_builder = LlmBuilder::long_running();
    println!("   ✓ LlmBuilder::long_running() works");

    let _defaults_builder = LlmBuilder::with_defaults();
    println!("   ✓ LlmBuilder::with_defaults() works");

    // Test 4: Message macros
    println!("\n4. Testing message macros:");
    let user_msg = user!("Hello");
    println!("   ✓ user! macro works");

    let system_msg = system!("You are helpful");
    println!("   ✓ system! macro works");

    let assistant_msg = assistant!("I can help");
    println!("   ✓ assistant! macro works");

    // Test 5: Builder pattern
    println!("\n5. Testing builder pattern:");
    let _builder = LlmBuilder::new()
        .deepseek()
        .model(deepseek::CHAT)
        .temperature(0.7);
    println!("   ✓ DeepSeek builder works");

    let _builder = LlmBuilder::new()
        .openrouter()
        .model(openrouter::openai::GPT_4O);
    println!("   ✓ OpenRouter builder works");

    println!("\n✅ All fixes are working correctly!");
    Ok(())
}

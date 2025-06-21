//! 测试所有提供商是否正确使用 siumai builder 的模型名和其他参数
//!
//! 这个示例演示了所有提供商现在正确使用从 builder 设置的模型名和其他参数。

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建一个 OpenAI 客户端，指定模型名
    let client = LlmBuilder::new()
        .openai()
        .api_key("test-key") // 这里使用测试密钥，实际使用时需要真实的 API 密钥
        .model("gpt-4-turbo") // 设置模型名
        .build()
        .await?;

    // 验证客户端是否存储了正确的模型名
    println!(
        "✅ OpenAI 客户端已创建，模型名: {}",
        client.common_params().model
    );

    // 创建一个测试消息
    let message = ChatMessage::user("Hello, world!");

    // 创建一个 ChatRequest 来测试模型名的使用
    let request = ChatRequest {
        messages: vec![message.build()],
        tools: None,
        common_params: client.common_params().clone(),
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

    // 测试请求体是否包含正确的模型名
    let body = client.chat_capability().build_chat_request_body(&request)?;

    if let Some(model) = body.get("model") {
        println!("✅ 请求体中的模型名: {}", model);
        assert_eq!(model, "gpt-4-turbo");
        println!("✅ 模型名验证成功！OpenAI 正确使用了 siumai builder 的模型名。");
    } else {
        println!("❌ 请求体中没有找到模型名");
        return Err("请求体中没有模型名".into());
    }

    // 测试不同的模型名
    println!("\n测试不同的模型名...");

    let client2 = LlmBuilder::new()
        .openai()
        .api_key("test-key")
        .model("gpt-3.5-turbo") // 不同的模型名
        .build()
        .await?;

    let request2 = ChatRequest {
        messages: vec![ChatMessage::user("Test").build()],
        tools: None,
        common_params: client2.common_params().clone(),
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

    let body2 = client2
        .chat_capability()
        .build_chat_request_body(&request2)?;

    if let Some(model) = body2.get("model") {
        println!("✅ 第二个客户端的模型名: {}", model);
        assert_eq!(model, "gpt-3.5-turbo");
        println!("✅ 第二个模型名验证成功！");
    }

    // 测试 Anthropic 提供商
    println!("\n测试 Anthropic 提供商...");

    let anthropic_client = LlmBuilder::new()
        .anthropic()
        .api_key("test-key")
        .model("claude-3-5-sonnet-20241022") // 设置模型名
        .temperature(0.8) // 设置温度
        .max_tokens(2000) // 设置最大令牌数
        .build()
        .await?;

    println!(
        "✅ Anthropic 客户端已创建，模型名: {}",
        anthropic_client.common_params().model
    );
    println!(
        "✅ Anthropic 客户端温度: {:?}",
        anthropic_client.common_params().temperature
    );
    println!(
        "✅ Anthropic 客户端最大令牌数: {:?}",
        anthropic_client.common_params().max_tokens
    );

    let anthropic_request = ChatRequest {
        messages: vec![ChatMessage::user("Test").build()],
        tools: None,
        common_params: anthropic_client.common_params().clone(),
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

    let anthropic_body = anthropic_client
        .chat_capability()
        .build_chat_request_body(&anthropic_request, Some(anthropic_client.specific_params()))?;

    if let Some(model) = anthropic_body.get("model") {
        println!("✅ Anthropic 请求体中的模型名: {}", model);
        assert_eq!(model, "claude-3-5-sonnet-20241022");
    }

    if let Some(temperature) = anthropic_body.get("temperature") {
        println!("✅ Anthropic 请求体中的温度: {}", temperature);
        // Use approximate comparison for floating point values
        let temp_val = temperature.as_f64().unwrap();
        assert!((temp_val - 0.8).abs() < 1e-6);
    }

    if let Some(max_tokens) = anthropic_body.get("max_tokens") {
        println!("✅ Anthropic 请求体中的最大令牌数: {}", max_tokens);
        assert_eq!(max_tokens, 2000);
    }

    // 测试 Ollama 提供商
    println!("\n测试 Ollama 提供商...");

    let ollama_client = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2") // 设置模型名
        .temperature(0.9) // 设置温度
        .max_tokens(1500) // 设置最大令牌数
        .build()
        .await?;

    println!(
        "✅ Ollama 客户端已创建，模型名: {}",
        ollama_client.common_params().model
    );
    println!(
        "✅ Ollama 客户端温度: {:?}",
        ollama_client.common_params().temperature
    );
    println!(
        "✅ Ollama 客户端最大令牌数: {:?}",
        ollama_client.common_params().max_tokens
    );

    let ollama_request = ChatRequest {
        messages: vec![ChatMessage::user("Test").build()],
        tools: None,
        common_params: ollama_client.common_params().clone(),
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

    let ollama_body = ollama_client
        .chat_capability()
        .build_chat_request_body(&ollama_request)?;

    println!("✅ Ollama 请求体中的模型名: {}", ollama_body.model);
    assert_eq!(ollama_body.model, "llama3.2");

    if let Some(options) = &ollama_body.options {
        if let Some(temperature) = options.get("temperature") {
            println!("✅ Ollama 请求体中的温度: {}", temperature);
            let temp_val = temperature.as_f64().unwrap();
            assert!((temp_val - 0.9).abs() < 1e-6);
        }
        if let Some(num_predict) = options.get("num_predict") {
            println!("✅ Ollama 请求体中的最大令牌数: {}", num_predict);
            assert_eq!(num_predict, 1500);
        }
    }

    // 测试 Gemini 提供商
    println!("\n测试 Gemini 提供商...");

    let gemini_client = LlmBuilder::new()
        .gemini()
        .api_key("test-key")
        .model("gemini-1.5-pro") // 设置模型名
        .temperature(0.6) // 设置温度
        .max_tokens(3000) // 设置最大令牌数
        .build()
        .await?;

    println!(
        "✅ Gemini 客户端已创建，模型名: {}",
        gemini_client.config().model
    );
    println!(
        "✅ Gemini 客户端温度: {:?}",
        gemini_client
            .config()
            .generation_config
            .as_ref()
            .and_then(|gc| gc.temperature)
    );
    println!(
        "✅ Gemini 客户端最大令牌数: {:?}",
        gemini_client
            .config()
            .generation_config
            .as_ref()
            .and_then(|gc| gc.max_output_tokens)
    );

    let gemini_request = gemini_client
        .chat_capability()
        .build_request_body(&[ChatMessage::user("Test").build()], None)?;

    println!("✅ Gemini 请求体中的模型名: {}", gemini_request.model);
    assert_eq!(gemini_request.model, "gemini-1.5-pro");

    if let Some(generation_config) = &gemini_request.generation_config {
        if let Some(temperature) = generation_config.temperature {
            println!("✅ Gemini 请求体中的温度: {}", temperature);
            assert!((temperature - 0.6).abs() < 1e-6);
        }
        if let Some(max_output_tokens) = generation_config.max_output_tokens {
            println!("✅ Gemini 请求体中的最大令牌数: {}", max_output_tokens);
            assert_eq!(max_output_tokens, 3000);
        }
    }

    println!("\n🎉 所有测试通过！所有提供商现在都正确使用 siumai builder 的模型名和其他参数。");

    Ok(())
}

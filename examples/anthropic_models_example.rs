//! Anthropic Models API 示例
//! 
//! 展示如何使用 Anthropic 的模型列表功能，符合官方 API 规范：
//! https://docs.anthropic.com/en/api/models-list

use siumai::providers::anthropic::AnthropicClient;
use siumai::traits::ModelListingCapability;
use siumai::types::*;
use siumai::error::LlmError;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 从环境变量获取 API 密钥
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("请设置 ANTHROPIC_API_KEY 环境变量");

    println!("🤖 Anthropic Models API 功能演示\n");

    // 创建 Anthropic 客户端
    let client = AnthropicClient::new(
        api_key,
        "https://api.anthropic.com".to_string(),
        reqwest::Client::new(),
        CommonParams::default(),
        Default::default(),
        Default::default(),
    );

    // 1. 列出所有可用模型
    demo_list_all_models(&client).await?;
    
    // 2. 获取特定模型信息
    demo_get_specific_model(&client).await?;
    
    // 3. 分析模型能力
    demo_analyze_model_capabilities(&client).await?;
    
    // 4. 比较模型规格
    demo_compare_model_specs(&client).await?;

    println!("✅ 所有演示完成！");
    Ok(())
}

/// 演示列出所有可用模型
async fn demo_list_all_models(client: &AnthropicClient) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 1. 列出所有可用模型");
    
    match client.list_models().await {
        Ok(models) => {
            println!("   找到 {} 个模型:", models.len());
            
            for (i, model) in models.iter().enumerate().take(10) { // 只显示前10个
                println!("   {}. {} ({})", 
                    i + 1, 
                    model.name.as_ref().unwrap_or(&model.id), 
                    model.id
                );
                
                if let Some(desc) = &model.description {
                    println!("      描述: {}", desc);
                }
                
                if let Some(context) = model.context_window {
                    println!("      上下文窗口: {} tokens", context);
                }
                
                if !model.capabilities.is_empty() {
                    println!("      能力: {}", model.capabilities.join(", "));
                }
                
                println!();
            }
            
            if models.len() > 10 {
                println!("   ... 还有 {} 个模型", models.len() - 10);
            }
        }
        Err(e) => {
            println!("   ❌ 获取模型列表失败: {}", e);
            
            // 分析错误类型
            match &e {
                LlmError::AuthenticationError(_) => {
                    println!("   💡 提示: 请检查您的 API 密钥是否正确");
                }
                LlmError::RateLimitError(_) => {
                    println!("   💡 提示: 请稍后重试，您已达到速率限制");
                }
                LlmError::ApiError { code, .. } => {
                    println!("   💡 API 错误码: {}", code);
                }
                _ => {
                    println!("   💡 其他错误类型");
                }
            }
        }
    }
    
    println!();
    Ok(())
}

/// 演示获取特定模型信息
async fn demo_get_specific_model(client: &AnthropicClient) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 2. 获取特定模型信息");
    
    let model_ids = vec![
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ];
    
    for model_id in model_ids {
        println!("   查询模型: {}", model_id);
        
        match client.get_model(model_id.to_string()).await {
            Ok(model) => {
                println!("   ✅ 模型信息:");
                println!("      ID: {}", model.id);
                println!("      名称: {}", model.name.unwrap_or("未知".to_string()));
                println!("      拥有者: {}", model.owned_by);
                
                if let Some(created) = model.created {
                    let datetime = chrono::DateTime::from_timestamp(created as i64, 0)
                        .unwrap_or_default();
                    println!("      创建时间: {}", datetime.format("%Y-%m-%d %H:%M:%S"));
                }
                
                if let Some(context) = model.context_window {
                    println!("      上下文窗口: {} tokens", context);
                }
                
                if let Some(max_output) = model.max_output_tokens {
                    println!("      最大输出: {} tokens", max_output);
                }
                
                if let Some(input_cost) = model.input_cost_per_token {
                    println!("      输入成本: ${:.8} per token", input_cost);
                }
                
                if let Some(output_cost) = model.output_cost_per_token {
                    println!("      输出成本: ${:.8} per token", output_cost);
                }
                
                println!("      能力: {}", model.capabilities.join(", "));
            }
            Err(e) => {
                println!("   ❌ 获取模型信息失败: {}", e);
            }
        }
        
        println!();
    }
    
    Ok(())
}

/// 演示分析模型能力
async fn demo_analyze_model_capabilities(client: &AnthropicClient) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 3. 分析模型能力");
    
    match client.list_models().await {
        Ok(models) => {
            let mut thinking_models = Vec::new();
            let mut vision_models = Vec::new();
            let mut tool_models = Vec::new();
            
            for model in &models {
                if model.capabilities.contains(&"thinking".to_string()) {
                    thinking_models.push(&model.id);
                }
                if model.capabilities.contains(&"vision".to_string()) {
                    vision_models.push(&model.id);
                }
                if model.capabilities.contains(&"tools".to_string()) {
                    tool_models.push(&model.id);
                }
            }
            
            println!("   🤔 支持 Thinking 的模型 ({} 个):", thinking_models.len());
            for model_id in thinking_models.iter().take(5) {
                println!("      - {}", model_id);
            }
            if thinking_models.len() > 5 {
                println!("      ... 还有 {} 个", thinking_models.len() - 5);
            }
            
            println!();
            println!("   👁️  支持视觉的模型 ({} 个):", vision_models.len());
            for model_id in vision_models.iter().take(5) {
                println!("      - {}", model_id);
            }
            if vision_models.len() > 5 {
                println!("      ... 还有 {} 个", vision_models.len() - 5);
            }
            
            println!();
            println!("   🔧 支持工具调用的模型 ({} 个):", tool_models.len());
            for model_id in tool_models.iter().take(5) {
                println!("      - {}", model_id);
            }
            if tool_models.len() > 5 {
                println!("      ... 还有 {} 个", tool_models.len() - 5);
            }
        }
        Err(e) => {
            println!("   ❌ 分析模型能力失败: {}", e);
        }
    }
    
    println!();
    Ok(())
}

/// 演示比较模型规格
async fn demo_compare_model_specs(client: &AnthropicClient) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 4. 比较模型规格");
    
    let comparison_models = vec![
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022", 
        "claude-3-5-haiku-20241022",
    ];
    
    println!("   模型规格对比:");
    println!("   {:<30} {:<15} {:<15} {:<15} {:<15}", 
        "模型", "上下文", "最大输出", "输入成本", "输出成本");
    println!("   {}", "-".repeat(90));
    
    for model_id in comparison_models {
        match client.get_model(model_id.to_string()).await {
            Ok(model) => {
                let context = model.context_window
                    .map(|c| format!("{}K", c / 1000))
                    .unwrap_or_else(|| "未知".to_string());
                
                let max_output = model.max_output_tokens
                    .map(|m| format!("{}K", m / 1000))
                    .unwrap_or_else(|| "未知".to_string());
                
                let input_cost = model.input_cost_per_token
                    .map(|c| format!("${:.6}", c))
                    .unwrap_or_else(|| "未知".to_string());
                
                let output_cost = model.output_cost_per_token
                    .map(|c| format!("${:.6}", c))
                    .unwrap_or_else(|| "未知".to_string());
                
                println!("   {:<30} {:<15} {:<15} {:<15} {:<15}", 
                    model_id, context, max_output, input_cost, output_cost);
            }
            Err(e) => {
                println!("   {:<30} 获取失败: {}", model_id, e);
            }
        }
    }
    
    println!();
    println!("   💡 成本说明:");
    println!("      - 成本以每个 token 的美元价格显示");
    println!("      - 实际成本可能因地区和使用量而异");
    println!("      - 建议查看官方文档获取最新定价");
    
    println!();
    Ok(())
}

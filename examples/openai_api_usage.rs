//! OpenAI API 使用示例
//!
//! 这个示例演示如何使用 siumai 库调用 OpenAI 的各种 API 端点：
//! - 文本嵌入 (Embeddings)
//! - 文本转语音 (Text-to-Speech)
//! - 语音转文本 (Speech-to-Text)
//! - 图像生成 (Image Generation)

use siumai::{
    providers::openai::{OpenAiAudio, OpenAiConfig, OpenAiEmbeddings, OpenAiImages},
    traits::{AudioCapability, EmbeddingCapability, ImageGenerationCapability},
    types::{ImageGenerationRequest, TtsRequest},
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    env_logger::init();

    println!("🚀 Siumai OpenAI API 使用示例");
    println!("==============================");

    // 获取 API 密钥
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("⚠️  请设置 OPENAI_API_KEY 环境变量");
        std::process::exit(1);
    });

    // 创建 OpenAI 配置
    let config = OpenAiConfig::new(api_key);
    let http_client = reqwest::Client::new();

    // 1. 文本嵌入示例
    println!("\n📊 1. 文本嵌入示例");
    println!("------------------");

    let embeddings_client = OpenAiEmbeddings::new(config.clone(), http_client.clone());

    let texts = vec![
        "Hello, world!".to_string(),
        "你好，世界！".to_string(),
        "Rust is a great programming language.".to_string(),
    ];

    match embeddings_client.embed(texts.clone()).await {
        Ok(response) => {
            println!("✅ 成功生成 {} 个文本的嵌入向量", response.embeddings.len());
            println!("📏 嵌入维度: {}", response.embeddings[0].len());
            println!("🤖 使用模型: {}", response.model);
            if let Some(usage) = response.usage {
                println!("📈 Token 使用: {} 个", usage.total_tokens);
            }
        }
        Err(e) => println!("❌ 嵌入生成失败: {}", e),
    }

    // 2. 文本转语音示例
    println!("\n🎵 2. 文本转语音示例");
    println!("--------------------");

    let audio_client = OpenAiAudio::new(config.clone(), http_client.clone());

    let tts_request = TtsRequest {
        text: "Hello, this is a test of the text-to-speech functionality in Siumai.".to_string(),
        voice: Some("alloy".to_string()),
        format: Some("mp3".to_string()),
        speed: Some(1.0),
        model: Some("tts-1".to_string()),
        extra_params: std::collections::HashMap::new(),
    };

    match audio_client.text_to_speech(tts_request).await {
        Ok(response) => {
            println!("✅ 成功生成语音");
            println!("📄 音频格式: {}", response.format);
            println!("📏 音频大小: {} 字节", response.audio_data.len());

            // 保存音频文件
            if let Err(e) = std::fs::write("output.mp3", &response.audio_data) {
                println!("⚠️  保存音频文件失败: {}", e);
            } else {
                println!("💾 音频已保存为 output.mp3");
            }
        }
        Err(e) => println!("❌ 语音生成失败: {}", e),
    }

    // 3. 图像生成示例
    println!("\n🎨 3. 图像生成示例");
    println!("------------------");

    let images_client = OpenAiImages::new(config.clone(), http_client.clone());

    let image_request = ImageGenerationRequest {
        prompt: "A beautiful sunset over mountains, digital art style".to_string(),
        negative_prompt: None,
        size: Some("1024x1024".to_string()),
        count: 1,
        model: Some("dall-e-3".to_string()),
        quality: Some("standard".to_string()),
        style: Some("vivid".to_string()),
        seed: None,
        steps: None,
        guidance_scale: None,
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: std::collections::HashMap::new(),
    };

    match images_client.generate_images(image_request).await {
        Ok(response) => {
            println!("✅ 成功生成 {} 张图像", response.images.len());
            for (i, image) in response.images.iter().enumerate() {
                if let Some(url) = &image.url {
                    println!("🖼️  图像 {}: {}", i + 1, url);
                }
                if let Some(revised_prompt) = &image.revised_prompt {
                    println!("📝 修订后的提示: {}", revised_prompt);
                }
            }
        }
        Err(e) => println!("❌ 图像生成失败: {}", e),
    }

    // 4. 显示支持的功能
    println!("\n📋 4. 支持的功能");
    println!("----------------");

    println!(
        "🔍 嵌入模型: {:?}",
        embeddings_client.supported_embedding_models()
    );
    println!("🎵 音频功能: {:?}", audio_client.supported_features());
    println!("🎨 图像尺寸: {:?}", images_client.get_supported_sizes());
    println!("📄 图像格式: {:?}", images_client.get_supported_formats());

    println!("\n✨ 示例完成！");
    println!("💡 提示：确保设置了有效的 OPENAI_API_KEY 环境变量");

    Ok(())
}

# OpenAI API 测试计划

## 概述

本文档定义了 OpenAI API 合规性修复的完整测试策略，确保我们的实现与官方 OpenAPI 规范 (`docs/openapi.documented.yml`) 完全兼容。

## 🧪 测试策略

### 测试层级

1. **单元测试** - 测试单个函数和方法
2. **集成测试** - 测试与真实 OpenAI API 的交互
3. **合规性测试** - 验证与 OpenAPI 规范的一致性
4. **回归测试** - 确保新功能不破坏现有功能

### 测试环境

- **模拟环境**: 使用 mock 服务器进行快速测试
- **沙盒环境**: 使用 OpenAI 测试 API 密钥
- **生产环境**: 使用真实 API 进行最终验证

## 📋 Chat Completions API 测试

### 1. 消息角色测试

**测试目标**: 验证新的 `developer` 角色支持

```rust
#[cfg(test)]
mod chat_role_tests {
    use super::*;
    
    #[test]
    fn test_developer_role_serialization() {
        let message = ChatMessage {
            role: ChatRole::Developer,
            content: "You are a helpful assistant.".to_string(),
        };
        
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"role\":\"developer\""));
    }
    
    #[test]
    fn test_developer_role_deserialization() {
        let json = r#"{"role":"developer","content":"test"}"#;
        let message: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.role, ChatRole::Developer);
    }
    
    #[tokio::test]
    async fn test_developer_role_api_call() {
        let client = create_test_client().await;
        let messages = vec![
            ChatMessage {
                role: ChatRole::Developer,
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hello!".to_string(),
            },
        ];
        
        let response = client.chat_with_tools(messages, None).await;
        assert!(response.is_ok());
    }
}
```

### 2. 新参数测试

**测试目标**: 验证所有新增的 Chat API 参数

```rust
#[cfg(test)]
mod chat_parameters_tests {
    use super::*;
    
    #[test]
    fn test_reasoning_effort_parameter() {
        let request = OpenAiChatRequest {
            model: "o1-preview".to_string(),
            messages: vec![test_message()],
            reasoning_effort: Some(ReasoningEffort::High),
            ..Default::default()
        };
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"reasoning_effort\":\"high\""));
    }
    
    #[test]
    fn test_frequency_penalty_validation() {
        let mut request = OpenAiChatRequest::default();
        
        // 有效范围
        request.frequency_penalty = Some(1.5);
        assert!(request.validate().is_ok());
        
        // 无效范围
        request.frequency_penalty = Some(3.0);
        assert!(request.validate().is_err());
        
        request.frequency_penalty = Some(-3.0);
        assert!(request.validate().is_err());
    }
    
    #[test]
    fn test_presence_penalty_validation() {
        let mut request = OpenAiChatRequest::default();
        
        // 有效范围
        request.presence_penalty = Some(-1.0);
        assert!(request.validate().is_ok());
        
        // 无效范围
        request.presence_penalty = Some(2.5);
        assert!(request.validate().is_err());
    }
    
    #[tokio::test]
    async fn test_max_completion_tokens() {
        let client = create_test_client().await;
        let request = ChatRequest {
            messages: vec![test_message()],
            max_completion_tokens: Some(100),
            ..Default::default()
        };
        
        let response = client.chat_with_request(request).await;
        assert!(response.is_ok());
        
        // 验证响应中的 token 使用情况
        let chat_response = response.unwrap();
        if let Some(usage) = chat_response.usage {
            assert!(usage.completion_tokens <= 100);
        }
    }
}
```

### 3. 推理模型兼容性测试

```rust
#[cfg(test)]
mod reasoning_model_tests {
    use super::*;
    
    #[test]
    fn test_reasoning_model_parameter_restrictions() {
        let mut request = OpenAiChatRequest {
            model: "o1-preview".to_string(),
            reasoning_effort: Some(ReasoningEffort::High),
            temperature: Some(0.7), // 不应该被允许
            ..Default::default()
        };
        
        // 推理模型不应该支持 temperature
        assert!(request.validate().is_err());
        
        request.temperature = None;
        request.top_p = Some(0.9); // 也不应该被允许
        assert!(request.validate().is_err());
    }
    
    #[tokio::test]
    async fn test_reasoning_model_response_format() {
        let client = create_test_client().await;
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "Solve this math problem: 2+2=?".to_string(),
        }];
        
        let response = client.chat_with_reasoning("o1-preview", messages, ReasoningEffort::High).await;
        assert!(response.is_ok());
        
        let chat_response = response.unwrap();
        // 验证推理内容是否存在
        assert!(chat_response.thinking.is_some());
    }
}
```

## 🎵 Audio API 测试

### 1. TTS 新功能测试

```rust
#[cfg(test)]
mod tts_tests {
    use super::*;
    
    #[test]
    fn test_new_tts_model_support() {
        let request = OpenAiTtsRequest {
            model: "gpt-4o-mini-tts".to_string(),
            input: "Hello world".to_string(),
            voice: "alloy".to_string(),
            instructions: Some("Speak slowly and clearly".to_string()),
            ..Default::default()
        };
        
        assert!(request.validate().is_ok());
    }
    
    #[test]
    fn test_instructions_model_compatibility() {
        // instructions 不应该用于旧模型
        let request = OpenAiTtsRequest {
            model: "tts-1".to_string(),
            input: "Hello".to_string(),
            voice: "alloy".to_string(),
            instructions: Some("Speak slowly".to_string()),
            ..Default::default()
        };
        
        assert!(request.validate().is_err());
    }
    
    #[test]
    fn test_new_voices() {
        let new_voices = vec!["ash", "ballad", "coral", "sage", "verse"];
        
        for voice in new_voices {
            let voice_enum: Result<TtsVoice, _> = serde_json::from_str(&format!("\"{}\"", voice));
            assert!(voice_enum.is_ok(), "Voice {} should be supported", voice);
        }
    }
    
    #[tokio::test]
    async fn test_tts_with_instructions() {
        let client = create_test_audio_client().await;
        let request = TtsRequest {
            text: "Hello, this is a test.".to_string(),
            model: Some("gpt-4o-mini-tts".to_string()),
            voice: Some("nova".to_string()),
            instructions: Some("Speak in a cheerful tone".to_string()),
            ..Default::default()
        };
        
        let response = client.text_to_speech(request).await;
        assert!(response.is_ok());
        
        let tts_response = response.unwrap();
        assert!(!tts_response.audio_data.is_empty());
    }
}
```

### 2. STT 流式测试

```rust
#[cfg(test)]
mod stt_streaming_tests {
    use super::*;
    use futures::StreamExt;
    
    #[tokio::test]
    async fn test_streaming_transcription() {
        let client = create_test_audio_client().await;
        let audio_data = load_test_audio_file("test_audio.mp3");
        
        let request = TranscriptionRequest {
            file: audio_data,
            model: "gpt-4o-transcribe".to_string(),
            stream: Some(true),
            ..Default::default()
        };
        
        let mut stream = client.transcribe_stream(request).await.unwrap();
        let mut transcript_parts = Vec::new();
        
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                TranscriptionEvent::TextDelta { text } => {
                    transcript_parts.push(text);
                }
                TranscriptionEvent::Complete { final_text } => {
                    assert!(!final_text.is_empty());
                    break;
                }
            }
        }
        
        assert!(!transcript_parts.is_empty());
    }
}
```

## 🖼️ Images API 测试

### 1. 新模型测试

```rust
#[cfg(test)]
mod image_model_tests {
    use super::*;
    
    #[test]
    fn test_gpt_image_1_prompt_length() {
        let long_prompt = "A".repeat(32000);
        let request = ImageGenerationRequest {
            prompt: long_prompt,
            model: Some("gpt-image-1".to_string()),
            ..Default::default()
        };
        
        assert!(request.validate().is_ok());
        
        // 超过限制应该失败
        let too_long_prompt = "A".repeat(32001);
        let invalid_request = ImageGenerationRequest {
            prompt: too_long_prompt,
            model: Some("gpt-image-1".to_string()),
            ..Default::default()
        };
        
        assert!(invalid_request.validate().is_err());
    }
    
    #[tokio::test]
    async fn test_gpt_image_1_generation() {
        let client = create_test_images_client().await;
        let request = ImageGenerationRequest {
            prompt: "A futuristic cityscape with flying cars and neon lights".to_string(),
            model: Some("gpt-image-1".to_string()),
            size: Some("1024x1024".to_string()),
            count: 1,
            ..Default::default()
        };
        
        let response = client.generate_images(request).await;
        assert!(response.is_ok());
        
        let image_response = response.unwrap();
        assert_eq!(image_response.images.len(), 1);
    }
}
```

### 2. 图像编辑测试

```rust
#[cfg(test)]
mod image_editing_tests {
    use super::*;
    
    #[test]
    fn test_edit_request_validation() {
        let image_data = load_test_image("test_image.png");
        let mask_data = load_test_image("test_mask.png");
        
        let request = ImageEditRequest {
            image: image_data,
            mask: Some(mask_data),
            prompt: "Add a red car to the scene".to_string(),
            model: Some(ImageModel::DallE2),
            ..Default::default()
        };
        
        assert!(request.validate().is_ok());
        
        // 测试不支持编辑的模型
        let invalid_request = ImageEditRequest {
            image: load_test_image("test_image.png"),
            prompt: "Edit this".to_string(),
            model: Some(ImageModel::DallE3), // 不支持编辑
            ..Default::default()
        };
        
        assert!(invalid_request.validate().is_err());
    }
    
    #[tokio::test]
    async fn test_image_editing_api() {
        let client = create_test_images_client().await;
        let image_data = load_test_image("test_image.png");
        
        let request = ImageEditRequest {
            image: image_data,
            prompt: "Add a blue sky background".to_string(),
            model: Some(ImageModel::DallE2),
            n: Some(1),
            ..Default::default()
        };
        
        let response = client.edit_image(request).await;
        assert!(response.is_ok());
        
        let edit_response = response.unwrap();
        assert!(!edit_response.images.is_empty());
    }
}
```

## 🔧 合规性测试

### OpenAPI 规范验证

```rust
#[cfg(test)]
mod openapi_compliance_tests {
    use super::*;
    
    #[test]
    fn test_chat_request_schema_compliance() {
        // 验证请求结构与 OpenAPI schema 匹配
        let request = OpenAiChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![test_message()],
            frequency_penalty: Some(0.5),
            presence_penalty: Some(-0.5),
            max_completion_tokens: Some(1000),
            reasoning_effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        };
        
        let json = serde_json::to_value(&request).unwrap();
        
        // 验证必需字段存在
        assert!(json["model"].is_string());
        assert!(json["messages"].is_array());
        
        // 验证可选字段格式
        assert!(json["frequency_penalty"].is_number());
        assert!(json["presence_penalty"].is_number());
        assert!(json["max_completion_tokens"].is_number());
        assert!(json["reasoning_effort"].is_string());
    }
    
    #[test]
    fn test_response_format_compliance() {
        // 验证响应格式与 OpenAPI schema 匹配
        let response_json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }"#;
        
        let response: ChatResponse = serde_json::from_str(response_json).unwrap();
        assert!(response.text().is_some());
        assert!(response.usage.is_some());
    }
}
```

## 📊 性能测试

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn test_chat_response_time() {
        let client = create_test_client().await;
        let messages = vec![test_message()];
        
        let start = Instant::now();
        let response = client.chat_with_tools(messages, None).await;
        let duration = start.elapsed();
        
        assert!(response.is_ok());
        assert!(duration.as_secs() < 30); // 响应时间应该在 30 秒内
    }
    
    #[tokio::test]
    async fn test_streaming_latency() {
        let client = create_test_client().await;
        let messages = vec![test_message()];
        
        let start = Instant::now();
        let mut stream = client.chat_stream(messages, None).await.unwrap();
        
        // 测试第一个 token 的延迟
        if let Some(first_event) = stream.next().await {
            let first_token_latency = start.elapsed();
            assert!(first_token_latency.as_secs() < 5); // 第一个 token 应该在 5 秒内
        }
    }
}
```

## 🚀 集成测试套件

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_end_to_end_chat_workflow() {
        let client = create_test_client().await;
        
        // 1. 基本对话
        let messages = vec![
            ChatMessage {
                role: ChatRole::Developer,
                content: "You are a helpful math tutor.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "What is 2+2?".to_string(),
            },
        ];
        
        let response = client.chat_with_tools(messages.clone(), None).await.unwrap();
        assert!(response.text().unwrap().contains("4"));
        
        // 2. 带工具的对话
        let tools = vec![create_calculator_tool()];
        let tool_response = client.chat_with_tools(messages, Some(tools)).await.unwrap();
        assert!(tool_response.tool_calls.is_some() || tool_response.text().is_some());
        
        // 3. 流式对话
        let mut stream = client.chat_stream(vec![test_message()], None).await.unwrap();
        let mut content = String::new();
        
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                ChatStreamEvent::TextDelta { text } => content.push_str(&text),
                ChatStreamEvent::Complete { .. } => break,
                _ => {}
            }
        }
        
        assert!(!content.is_empty());
    }
}
```

## 📋 测试执行计划

### 阶段 1: 单元测试 (1 周)
- [ ] Chat API 参数验证测试
- [ ] Audio API 新功能测试
- [ ] Images API 模型支持测试
- [ ] 序列化/反序列化测试

### 阶段 2: 集成测试 (1 周)
- [ ] 真实 API 调用测试
- [ ] 错误处理测试
- [ ] 边界条件测试
- [ ] 性能基准测试

### 阶段 3: 合规性测试 (1 周)
- [ ] OpenAPI 规范验证
- [ ] 响应格式验证
- [ ] 参数范围验证
- [ ] 模型兼容性测试

### 阶段 4: 回归测试 (持续)
- [ ] 现有功能验证
- [ ] 向后兼容性测试
- [ ] 自动化测试套件
- [ ] CI/CD 集成

---

*本测试计划确保所有新功能都经过充分验证，并与 OpenAI API 规范完全兼容。*

# OpenAI API 实现指南

## 概述

本文档提供了基于 OpenAI OpenAPI 规范 (`docs/openapi.documented.yml`) 的详细实现指南，包含具体的代码示例、OpenAPI 引用和实现步骤。

## 🎯 阶段 1: Chat Completions API 修复

### 1.1 消息角色扩展

**OpenAPI 规范引用:**
- 文件: `docs/openapi.documented.yml`
- 行号: 3181-3189 (Chat Completions 示例)
- 内容: 显示 `"role": "developer"` 的使用

**当前问题:**
```rust
// src/types.rs - 缺少 developer 角色
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}
```

**修复实现:**
```rust
// src/types.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Developer,  // 新增 - 用于开发者级别的系统指令
    Tool,
}
```

### 1.2 新请求参数实现

**OpenAPI 规范引用:**
- 文件: `docs/openapi.documented.yml`
- 行号: 30441-30500 (CreateChatCompletionRequest schema)

**需要添加的参数:**

1. **modalities 参数**
   ```yaml
   # OpenAPI 定义 (行 30463-30464)
   modalities:
     $ref: '#/components/schemas/ResponseModalities'
   ```

2. **reasoning_effort 参数**
   ```yaml
   # OpenAPI 定义 (行 30465-30466)
   reasoning_effort:
     $ref: '#/components/schemas/ReasoningEffort'
   ```

3. **max_completion_tokens 参数**
   ```yaml
   # OpenAPI 定义 (行 30467-30472)
   max_completion_tokens:
     description: >
       An upper bound for the number of tokens that can be generated for a completion,
       including visible output tokens and reasoning tokens.
     type: integer
     nullable: true
   ```

**实现代码:**
```rust
// src/providers/openai/types.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseModalities {
    pub text: Option<bool>,
    pub audio: Option<bool>,
}

// src/providers/openai/chat.rs - 扩展请求结构
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    
    // 新增参数
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    
    // 现有参数
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}
```

### 1.3 参数验证逻辑

```rust
impl OpenAiChatRequest {
    pub fn validate(&self) -> Result<(), LlmError> {
        // 验证 frequency_penalty 范围
        if let Some(penalty) = self.frequency_penalty {
            if penalty < -2.0 || penalty > 2.0 {
                return Err(LlmError::InvalidInput(
                    "frequency_penalty must be between -2.0 and 2.0".to_string()
                ));
            }
        }
        
        // 验证 presence_penalty 范围
        if let Some(penalty) = self.presence_penalty {
            if penalty < -2.0 || penalty > 2.0 {
                return Err(LlmError::InvalidInput(
                    "presence_penalty must be between -2.0 and 2.0".to_string()
                ));
            }
        }
        
        // 验证推理模型的参数兼容性
        if self.reasoning_effort.is_some() {
            // 推理模型通常不支持某些参数
            if self.temperature.is_some() || self.top_p.is_some() {
                return Err(LlmError::InvalidInput(
                    "reasoning models do not support temperature or top_p parameters".to_string()
                ));
            }
        }
        
        Ok(())
    }
}
```

## 🎵 阶段 2: Audio API 增强

### 2.1 TTS 新模型和参数

**OpenAPI 规范引用:**
- 文件: `docs/openapi.documented.yml`
- 行号: 33346-33380 (CreateSpeechRequest schema)

**新模型支持:**
```yaml
# OpenAPI 定义 (行 33356-33359)
model:
  enum:
    - tts-1
    - tts-1-hd
    - gpt-4o-mini-tts  # 新模型
```

**instructions 参数:**
```yaml
# OpenAPI 定义 (行 33365-33370)
instructions:
  type: string
  description: >-
    Control the voice of your generated audio with additional instructions.
    Does not work with `tts-1` or `tts-1-hd`.
  maxLength: 4096
```

**实现代码:**
```rust
// src/providers/openai/audio.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TtsModel {
    Tts1,
    Tts1Hd,
    #[serde(rename = "gpt-4o-mini-tts")]
    Gpt4oMiniTts,  // 新增
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiTtsRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,  // 新增
}

impl OpenAiTtsRequest {
    pub fn validate(&self) -> Result<(), LlmError> {
        // 验证 instructions 参数的模型兼容性
        if self.instructions.is_some() {
            if self.model == "tts-1" || self.model == "tts-1-hd" {
                return Err(LlmError::InvalidInput(
                    "instructions parameter is not supported for tts-1 and tts-1-hd models".to_string()
                ));
            }
        }
        
        // 验证输入长度
        if self.input.len() > 4096 {
            return Err(LlmError::InvalidInput(
                "input text cannot exceed 4096 characters".to_string()
            ));
        }
        
        // 验证 instructions 长度
        if let Some(instructions) = &self.instructions {
            if instructions.len() > 4096 {
                return Err(LlmError::InvalidInput(
                    "instructions cannot exceed 4096 characters".to_string()
                ));
            }
        }
        
        Ok(())
    }
}
```

### 2.2 新语音选项

**OpenAPI 规范引用:**
- 文件: `docs/openapi.documented.yml`
- 行号: 33371-33376 (voice 参数定义)

```yaml
voice:
  description: >-
    The voice to use when generating the audio. Supported voices are `alloy`, `ash`, `ballad`,
    `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, and `verse`.
```

**实现代码:**
```rust
// src/providers/openai/audio.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsVoice {
    Alloy,
    Ash,      // 新增
    Ballad,   // 新增
    Coral,    // 新增
    Echo,
    Fable,
    Nova,
    Onyx,
    Sage,     // 新增
    Shimmer,
    Verse,    // 新增
}

impl TtsVoice {
    pub fn all_voices() -> Vec<TtsVoice> {
        vec![
            TtsVoice::Alloy,
            TtsVoice::Ash,
            TtsVoice::Ballad,
            TtsVoice::Coral,
            TtsVoice::Echo,
            TtsVoice::Fable,
            TtsVoice::Nova,
            TtsVoice::Onyx,
            TtsVoice::Sage,
            TtsVoice::Shimmer,
            TtsVoice::Verse,
        ]
    }
    
    pub fn is_supported_by_model(&self, model: &TtsModel) -> bool {
        // 所有语音都支持所有模型
        true
    }
}
```

### 2.3 流式转录支持

**OpenAPI 规范引用:**
- 文件: `docs/openapi.documented.yml`
- 行号: 1210-1245 (Streaming transcription 示例)

**实现代码:**
```rust
// src/providers/openai/audio.rs
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    pub file: Vec<u8>,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<f32>,
    pub timestamp_granularities: Option<Vec<String>>,
    pub stream: Option<bool>,  // 新增流式支持
}

impl OpenAiAudio {
    pub async fn transcribe_stream(
        &self,
        request: TranscriptionRequest,
    ) -> Result<impl Stream<Item = Result<TranscriptionEvent, LlmError>>, LlmError> {
        if !request.stream.unwrap_or(false) {
            return Err(LlmError::InvalidInput(
                "stream parameter must be true for streaming transcription".to_string()
            ));
        }
        
        // 构建 multipart form data
        let form = self.build_transcription_form(request)?;
        let url = format!("{}/audio/transcriptions", self.config.base_url);
        
        // 发送流式请求
        let response = self.http_client
            .post(&url)
            .headers(self.build_headers()?)
            .multipart(form)
            .send()
            .await?;
            
        // 解析 SSE 流
        Ok(self.parse_transcription_stream(response))
    }
}
```

## 🖼️ 阶段 3: Images API 完善

### 3.1 新模型支持

**OpenAPI 规范引用:**
- 文件: `docs/openapi.documented.yml`
- 行号: 32428-32442 (CreateImageRequest model 定义)

```yaml
model:
  enum:
    - dall-e-2
    - dall-e-3
    - gpt-image-1  # 新模型
  default: dall-e-2
```

**提示长度限制:**
- `gpt-image-1`: 32000 字符 (行 32424-32426)
- `dall-e-3`: 4000 字符
- `dall-e-2`: 1000 字符

**实现代码:**
```rust
// src/providers/openai/images.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageModel {
    #[serde(rename = "dall-e-2")]
    DallE2,
    #[serde(rename = "dall-e-3")]
    DallE3,
    #[serde(rename = "gpt-image-1")]
    GptImage1,  // 新增
}

impl ImageModel {
    pub fn max_prompt_length(&self) -> usize {
        match self {
            ImageModel::DallE2 => 1000,
            ImageModel::DallE3 => 4000,
            ImageModel::GptImage1 => 32000,
        }
    }
    
    pub fn supports_editing(&self) -> bool {
        match self {
            ImageModel::DallE2 | ImageModel::GptImage1 => true,
            ImageModel::DallE3 => false,
        }
    }
    
    pub fn supports_variations(&self) -> bool {
        match self {
            ImageModel::DallE2 => true,
            ImageModel::DallE3 | ImageModel::GptImage1 => false,
        }
    }
}
```

### 3.2 图像编辑功能

**OpenAPI 规范引用:**
- 文件: `docs/openapi.documented.yml`
- 行号: 12695-12710 (/images/edits 端点)

```yaml
/images/edits:
  post:
    operationId: createImageEdit
    summary: >-
      Creates an edited or extended image given one or more source images and a prompt.
      This endpoint only supports `gpt-image-1` and `dall-e-2`.
```

**实现代码:**
```rust
// src/providers/openai/images.rs
#[derive(Debug, Clone)]
pub struct ImageEditRequest {
    pub image: Vec<u8>,           // 原始图像数据
    pub mask: Option<Vec<u8>>,    // 可选的遮罩图像
    pub prompt: String,           // 编辑描述
    pub model: Option<ImageModel>, // 模型选择
    pub n: Option<u32>,          // 生成数量 (1-10)
    pub size: Option<String>,     // 图像尺寸
    pub response_format: Option<String>, // url 或 b64_json
    pub user: Option<String>,     // 用户标识
}

impl ImageEditRequest {
    pub fn validate(&self) -> Result<(), LlmError> {
        // 验证模型支持
        if let Some(model) = &self.model {
            if !model.supports_editing() {
                return Err(LlmError::InvalidInput(
                    format!("Model {:?} does not support image editing", model)
                ));
            }
        }
        
        // 验证生成数量
        if let Some(n) = self.n {
            if n < 1 || n > 10 {
                return Err(LlmError::InvalidInput(
                    "n must be between 1 and 10".to_string()
                ));
            }
        }
        
        Ok(())
    }
}

impl OpenAiImages {
    pub async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        request.validate()?;
        
        // 构建 multipart form data
        let mut form = reqwest::multipart::Form::new();
        
        // 添加图像文件
        let image_part = reqwest::multipart::Part::bytes(request.image)
            .file_name("image.png")
            .mime_str("image/png")?;
        form = form.part("image", image_part);
        
        // 添加遮罩文件 (如果提供)
        if let Some(mask) = request.mask {
            let mask_part = reqwest::multipart::Part::bytes(mask)
                .file_name("mask.png")
                .mime_str("image/png")?;
            form = form.part("mask", mask_part);
        }
        
        // 添加其他参数
        form = form.text("prompt", request.prompt);
        
        if let Some(model) = request.model {
            form = form.text("model", format!("{:?}", model).to_lowercase());
        }
        
        if let Some(n) = request.n {
            form = form.text("n", n.to_string());
        }
        
        // 发送请求
        let url = format!("{}/images/edits", self.config.base_url);
        let response = self.http_client
            .post(&url)
            .headers(self.build_headers()?)
            .multipart(form)
            .send()
            .await?;
            
        self.parse_image_response(response).await
    }
}
```

---

*本实现指南提供了详细的代码示例和 OpenAPI 规范引用。每个实现都包含了适当的验证逻辑和错误处理。*

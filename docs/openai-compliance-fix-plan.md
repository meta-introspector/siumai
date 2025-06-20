# OpenAI API 合规性修复计划

## 概述

基于对 OpenAI OpenAPI 规范文档 (`docs/openapi.documented.yml`) 的详细分析，我们的 Rust LLM 库在 OpenAI provider 实现上存在多个功能缺失和不合规问题。本文档制定了详细的修复计划，确保我们的实现与官方 OpenAI API 规范完全兼容。

## 🔍 当前状态分析

### ✅ 已正确实现的功能
- Chat Completions 基本功能
- Audio TTS/STT 基本功能  
- Images 基本生成功能
- Embeddings 基本功能
- 流式响应支持
- 工具调用支持

### ❌ 主要问题和缺失功能
1. **Chat API 不完整** - 缺少新角色、参数和功能
2. **Audio API 功能受限** - 缺少新模型和参数
3. **Images API 功能不全** - 缺少编辑和变体功能
4. **完全缺失的 API** - Assistants、Files、Moderations 等
5. **参数支持不完整** - 很多 OpenAI 特定参数未实现

## 📋 修复计划

### 阶段 1: 核心 API 完善 (高优先级)

#### 1.1 Chat Completions API 修复

**参考 OpenAPI 规范:**
```yaml
# docs/openapi.documented.yml:30441-30500
CreateChatCompletionRequest:
  properties:
    messages:
      type: array
      items:
        $ref: '#/components/schemas/ChatCompletionRequestMessage'
    model:
      $ref: '#/components/schemas/ModelIdsShared'
    modalities:
      $ref: '#/components/schemas/ResponseModalities'
    reasoning_effort:
      $ref: '#/components/schemas/ReasoningEffort'
    max_completion_tokens:
      type: integer
      nullable: true
    frequency_penalty:
      type: number
      minimum: -2
      maximum: 2
    presence_penalty:
      type: number
      minimum: -2
      maximum: 2
```

**需要修复的问题:**

1. **添加 `developer` 角色支持**
   - 位置: `src/types.rs` - `ChatRole` 枚举
   - 参考: OpenAPI 示例中的 `"role": "developer"` 消息

2. **添加缺失的请求参数**
   - `modalities`: 响应模态控制 (文本、音频等)
   - `reasoning_effort`: 推理努力程度 (low/medium/high)
   - `max_completion_tokens`: 替代 max_tokens
   - `frequency_penalty`: 频率惩罚 (-2.0 到 2.0)
   - `presence_penalty`: 存在惩罚 (-2.0 到 2.0)
   - `logit_bias`: Token 概率偏置
   - `seed`: 可重现输出的种子
   - `user`: 用户标识符
   - `service_tier`: 服务层级

3. **音频消息支持**
   - 支持音频输入消息类型
   - 支持音频输出响应

**实现步骤:**
```rust
// 1. 扩展 ChatRole 枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Developer,  // 新增
    Tool,
}

// 2. 添加新的请求参数结构
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    // ... 其他参数
}
```

#### 1.2 Audio API 增强

**参考 OpenAPI 规范:**
```yaml
# docs/openapi.documented.yml:33346-33380
CreateSpeechRequest:
  properties:
    model:
      enum:
        - tts-1
        - tts-1-hd
        - gpt-4o-mini-tts  # 新模型
    input:
      type: string
      maxLength: 4096
    instructions:  # 新参数
      type: string
      maxLength: 4096
    voice:
      # 新增语音: ash, ballad, coral, fable, sage, shimmer, verse
```

**需要修复的问题:**

1. **添加新的 TTS 模型支持**
   - `gpt-4o-mini-tts`
   - 更新默认模型列表

2. **添加 `instructions` 参数**
   - 用于控制生成音频的语音特征
   - 仅适用于新模型，不适用于 `tts-1` 和 `tts-1-hd`

3. **扩展语音选项**
   - 当前: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
   - 新增: `ash`, `ballad`, `coral`, `sage`, `verse`

4. **流式转录支持**
   - 添加 `stream` 参数到转录请求
   - 实现流式转录响应处理

#### 1.3 Images API 完善

**参考 OpenAPI 规范:**
```yaml
# docs/openapi.documented.yml:32419-32450
CreateImageRequest:
  properties:
    model:
      enum:
        - dall-e-2
        - dall-e-3
        - gpt-image-1  # 新模型
    prompt:
      type: string
      # gpt-image-1: 32000 chars, dall-e-3: 4000 chars, dall-e-2: 1000 chars
```

**需要修复的问题:**

1. **添加新模型支持**
   - `gpt-image-1` 模型
   - 支持更长的提示 (32000 字符)

2. **实现图像编辑功能**
   - `/images/edits` 端点
   - 支持图像和遮罩上传

3. **实现图像变体功能**
   - `/images/variations` 端点
   - 基于现有图像生成变体

### 阶段 2: 缺失 API 实现 (中优先级)

#### 2.1 Files API 实现

**参考 OpenAPI 规范:**
```yaml
# docs/openapi.documented.yml:9251-9270
/files:
  post:
    operationId: createFile
    summary: Upload a file that can be used across various endpoints
```

**实现范围:**
- 文件上传 (`POST /files`)
- 文件列表 (`GET /files`)
- 文件检索 (`GET /files/{file_id}`)
- 文件删除 (`DELETE /files/{file_id}`)
- 文件内容获取 (`GET /files/{file_id}/content`)

#### 2.2 Moderations API 实现

**参考 OpenAPI 规范:**
```yaml
# docs/openapi.documented.yml:13577-13590
/moderations:
  post:
    operationId: createModeration
    summary: Classifies if text and/or image inputs are potentially harmful
```

**实现范围:**
- 文本内容审核
- 图像内容审核 (如果支持)
- 详细的分类结果
- 置信度分数

#### 2.3 Models API 实现

**实现范围:**
- 模型列表 (`GET /models`)
- 模型详情 (`GET /models/{model}`)
- 模型能力信息
- 模型状态和可用性

### 阶段 3: 高级功能实现 (低优先级)

#### 3.1 Assistants API 实现

**参考 OpenAPI 规范:**
```yaml
# docs/openapi.documented.yml:53-107
/assistants:
  get:
    operationId: listAssistants
  post:
    operationId: createAssistant
```

**实现范围:**
- 助手创建、修改、删除
- 助手列表和检索
- 工具集成 (代码解释器、文件搜索、函数调用)
- 线程和消息管理
- 运行管理

#### 3.2 Responses API 实现

**参考 OpenAPI 规范:**
```yaml
# docs/openapi.documented.yml:17752-17770
/responses:
  post:
    operationId: createResponse
    summary: Creates a model response with built-in tools
```

**实现范围:**
- 统一的响应接口
- 内置工具支持 (网络搜索、文件搜索等)
- 多轮工作流程
- 后台处理支持

## 🛠️ 实现指南

### 代码结构建议

```
src/providers/openai/
├── mod.rs              # 模块导出和文档
├── client.rs           # HTTP 客户端和基础功能
├── config.rs           # 配置管理
├── chat.rs             # Chat Completions API (需要大幅更新)
├── audio.rs            # Audio API (需要增强)
├── images.rs           # Images API (需要完善)
├── embeddings.rs       # Embeddings API (基本完整)
├── files.rs            # Files API (新增)
├── moderations.rs      # Moderations API (新增)
├── models.rs           # Models API (需要完善)
├── assistants.rs       # Assistants API (新增)
├── responses.rs        # Responses API (新增)
├── types.rs            # OpenAI 特定类型定义
└── utils.rs            # 工具函数
```

### 测试策略

1. **单元测试**: 每个 API 端点的基本功能
2. **集成测试**: 与真实 OpenAI API 的兼容性测试
3. **参数验证测试**: 确保所有参数正确传递
4. **错误处理测试**: 验证错误响应的正确处理

### 文档更新

1. **API 文档**: 更新所有新增功能的文档
2. **示例代码**: 提供每个新功能的使用示例
3. **迁移指南**: 为现有用户提供升级指南
4. **兼容性说明**: 明确哪些功能需要特定的 OpenAI 模型

## 📅 时间计划

- **阶段 1** (2-3 周): 核心 API 完善
- **阶段 2** (3-4 周): 缺失 API 实现
- **阶段 3** (4-6 周): 高级功能实现

## 🎯 成功标准

1. **100% OpenAPI 规范兼容性**: 所有支持的端点完全符合官方规范
2. **向后兼容性**: 现有代码无需修改即可继续工作
3. **完整的测试覆盖**: 所有新功能都有对应的测试
4. **详细的文档**: 每个功能都有清晰的使用说明和示例

## 📖 详细实现参考

### Chat Completions API 详细修复

#### 消息角色扩展

**OpenAPI 参考 (行 3181-3189):**
```yaml
{
  "role": "developer",
  "content": "You are a helpful assistant."
},
{
  "role": "user",
  "content": "Hello!"
}
```

**当前实现问题:**
```rust
// src/types.rs - 当前只支持 4 种角色
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}
```

**修复后实现:**
```rust
// 需要添加 Developer 角色
pub enum ChatRole {
    System,
    User,
    Assistant,
    Developer,  // 新增 - 用于系统级指令
    Tool,
}
```

#### 推理模型参数支持

**OpenAPI 参考 (行 30465-30466):**
```yaml
reasoning_effort:
  $ref: '#/components/schemas/ReasoningEffort'
```

**ReasoningEffort 定义 (需要在 OpenAPI 中查找):**
- `low`: 快速推理，较少思考时间
- `medium`: 平衡的推理努力
- `high`: 深度推理，更多思考时间

**实现建议:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

// 在 ChatRequest 中添加
pub struct ChatRequest {
    // ... 现有字段
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
}
```

#### 音频消息支持

**OpenAPI 参考 (行 3362-3388):**
```yaml
# 图像输入示例，音频输入类似
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "What is in this image?"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://example.com/image.jpg"
      }
    }
  ]
}
```

**音频消息实现:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    Audio { audio: AudioContent },  // 新增
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioContent {
    pub data: Option<String>,  // base64 编码的音频数据
    pub format: Option<String>, // 音频格式
}
```

### Audio API 详细修复

#### TTS Instructions 参数

**OpenAPI 参考 (行 33365-33370):**
```yaml
instructions:
  type: string
  description: >-
    Control the voice of your generated audio with additional instructions.
    Does not work with `tts-1` or `tts-1-hd`.
  maxLength: 4096
```

**当前实现问题:**
```rust
// src/providers/openai/audio.rs - 缺少 instructions 参数
pub struct OpenAiTtsRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    pub response_format: Option<String>,
    pub speed: Option<f32>,
    // 缺少 instructions 字段
}
```

**修复实现:**
```rust
pub struct OpenAiTtsRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    pub response_format: Option<String>,
    pub speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,  // 新增
}

// 验证逻辑
impl OpenAiTtsRequest {
    pub fn validate(&self) -> Result<(), LlmError> {
        // instructions 不适用于 tts-1 和 tts-1-hd
        if self.instructions.is_some() &&
           (self.model == "tts-1" || self.model == "tts-1-hd") {
            return Err(LlmError::InvalidInput(
                "instructions parameter not supported for tts-1 and tts-1-hd models".to_string()
            ));
        }
        Ok(())
    }
}
```

#### 新语音选项

**OpenAPI 参考 (行 33371-33376):**
```yaml
voice:
  description: >-
    The voice to use when generating the audio. Supported voices are `alloy`, `ash`, `ballad`,
    `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, and `verse`.
```

**当前实现更新:**
```rust
// 扩展语音枚举
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
```

### Images API 详细修复

#### 新模型支持

**OpenAPI 参考 (行 32428-32442):**
```yaml
model:
  enum:
    - dall-e-2
    - dall-e-3
    - gpt-image-1  # 新模型
  default: dall-e-2
  description: >-
    The model to use for image generation. One of `dall-e-2`, `dall-e-3`, or `gpt-image-1`.
```

**提示长度限制:**
- `gpt-image-1`: 32000 字符
- `dall-e-3`: 4000 字符
- `dall-e-2`: 1000 字符

**实现更新:**
```rust
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
}
```

#### 图像编辑功能

**OpenAPI 参考 (行 12695-12710):**
```yaml
/images/edits:
  post:
    operationId: createImageEdit
    summary: >-
      Creates an edited or extended image given one or more source images and a prompt.
      This endpoint only supports `gpt-image-1` and `dall-e-2`.
```

**实现建议:**
```rust
pub struct ImageEditRequest {
    pub image: Vec<u8>,           // 原始图像数据
    pub mask: Option<Vec<u8>>,    // 可选的遮罩图像
    pub prompt: String,           // 编辑描述
    pub model: Option<String>,    // 模型选择
    pub n: Option<u32>,          // 生成数量
    pub size: Option<String>,     // 图像尺寸
    pub response_format: Option<String>,
    pub user: Option<String>,
}

#[async_trait]
impl ImageGenerationCapability for OpenAiImages {
    async fn edit_image(&self, request: ImageEditRequest) -> Result<ImageGenerationResponse, LlmError> {
        // 验证模型支持
        if let Some(model) = &request.model {
            if model != "dall-e-2" && model != "gpt-image-1" {
                return Err(LlmError::InvalidInput(
                    "Image editing only supports dall-e-2 and gpt-image-1 models".to_string()
                ));
            }
        }

        // 构建 multipart/form-data 请求
        // ... 实现细节
    }
}
```

## 🔧 实现检查清单

### Chat API 修复清单
- [ ] 添加 `developer` 角色支持
- [ ] 实现 `modalities` 参数
- [ ] 实现 `reasoning_effort` 参数
- [ ] 实现 `max_completion_tokens` 参数
- [ ] 实现 `frequency_penalty` 参数
- [ ] 实现 `presence_penalty` 参数
- [ ] 实现 `logit_bias` 参数
- [ ] 实现 `seed` 参数
- [ ] 实现 `user` 参数
- [ ] 实现 `service_tier` 参数
- [ ] 添加音频消息支持
- [ ] 添加存储功能支持

### Audio API 修复清单
- [ ] 添加 `gpt-4o-mini-tts` 模型
- [ ] 实现 `instructions` 参数
- [ ] 添加新语音选项 (ash, ballad, coral, sage, verse)
- [ ] 实现流式转录支持
- [ ] 更新语音枚举定义

### Images API 修复清单
- [ ] 添加 `gpt-image-1` 模型支持
- [ ] 实现图像编辑功能 (`/images/edits`)
- [ ] 实现图像变体功能 (`/images/variations`)
- [ ] 支持更长的提示文本
- [ ] 添加质量参数支持

### 新 API 实现清单
- [ ] Files API 完整实现
- [ ] Moderations API 完整实现
- [ ] Models API 完善
- [ ] Assistants API 实现
- [ ] Responses API 实现

---

*本文档将随着实现进度持续更新。如有疑问或建议，请参考 OpenAPI 规范文档或提出 issue。*

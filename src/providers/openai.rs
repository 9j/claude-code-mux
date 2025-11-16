use super::{AnthropicProvider, ProviderResponse, ContentBlock, Usage, error::ProviderError};
use crate::models::{AnthropicRequest, CountTokensRequest, CountTokensResponse, MessageContent};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::pin::Pin;
use futures::stream::Stream;
use bytes::Bytes;

/// OpenAI Chat Completions request format
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

/// OpenAI Responses API request format (for Codex models)
#[derive(Debug, Serialize)]
struct OpenAIResponsesRequest {
    model: String,
    input: OpenAIResponsesInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,  // Responses API uses max_output_tokens, not max_tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Input for Responses API can be string or array of messages
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIResponsesInput {
    Text(String),
    Messages(Vec<OpenAIResponsesMessage>),
}

/// Message format for Responses API
#[derive(Debug, Serialize)]
struct OpenAIResponsesMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// Content can be string or array of content parts
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum OpenAIContent {
    String(String),
    Parts(Vec<OpenAIContentPart>),
}

/// Content part (text or image_url)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum OpenAIContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OpenAIImageUrl },
}

/// Image URL object
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIImageUrl {
    url: String,
}

/// Tool call in assistant message
#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    r#type: String, // "function"
    function: OpenAIFunctionCall,
}

/// Function call details
#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String, // JSON string
}

/// Tool definition
#[derive(Debug, Serialize, Deserialize)]
struct OpenAITool {
    r#type: String, // "function"
    function: OpenAIFunctionDef,
}

/// Function definition
#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionDef {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// OpenAI Chat Completions response format
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    #[serde(default, rename = "object")]
    _object: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

/// OpenAI Responses API response format (for Codex models)
#[derive(Debug, Deserialize)]
struct OpenAIResponsesResponse {
    id: String,
    model: String,
    output: Vec<ResponsesOutput>,
    usage: ResponsesUsage,
}

#[derive(Debug, Deserialize)]
struct ResponsesOutput {
    #[serde(rename = "type")]
    output_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Vec<ResponsesContentBlock>>,
}

#[derive(Debug, Deserialize)]
struct ResponsesContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    input_tokens: u32,
    output_tokens: u32,
}

/// OpenAI provider implementation
pub struct OpenAIProvider {
    api_key: String,
    base_url: String,
    client: Client,
    models: Vec<String>,
    custom_headers: Vec<(String, String)>,
}

impl OpenAIProvider {
    pub fn new(api_key: String, base_url: Option<String>, models: Vec<String>) -> Self {
        Self {
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            client: Client::new(),
            models,
            custom_headers: Vec::new(),
        }
    }

    /// Check if the model is a Codex model that requires /v1/responses endpoint
    fn is_codex_model(model: &str) -> bool {
        model.to_lowercase().contains("codex")
    }

    /// Transform Anthropic request to OpenAI Responses API format
    fn transform_to_responses_request(&self, request: &AnthropicRequest) -> Result<OpenAIResponsesRequest, ProviderError> {
        // Convert messages to Responses API input format
        let mut messages = Vec::new();

        // Add system message first if present
        if let Some(ref system) = request.system {
            let system_text = match system {
                crate::models::SystemPrompt::Text(text) => text.clone(),
                crate::models::SystemPrompt::Blocks(blocks) => {
                    blocks.iter()
                        .map(|b| b.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            };
            messages.push(OpenAIResponsesMessage {
                role: "system".to_string(),
                content: Some(system_text),
            });
        }

        // Transform messages
        for msg in &request.messages {
            let content = match &msg.content {
                MessageContent::Text(text) => text.clone(),
                MessageContent::Blocks(blocks) => {
                    let text = blocks.iter()
                        .filter_map(|block| {
                            match block {
                                crate::models::ContentBlock::Text { text } => Some(text.clone()),
                                _ => None,
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    // Responses API requires content, use empty string if none
                    if text.is_empty() {
                        String::new()
                    } else {
                        text
                    }
                }
            };

            messages.push(OpenAIResponsesMessage {
                role: msg.role.clone(),
                content: Some(content),  // Always provide content
            });
        }

        Ok(OpenAIResponsesRequest {
            model: request.model.clone(),
            input: OpenAIResponsesInput::Messages(messages),
            max_output_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
            stream: request.stream,
        })
    }

    pub fn with_headers(api_key: String, base_url: Option<String>, models: Vec<String>, custom_headers: Vec<(String, String)>) -> Self {
        Self {
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            client: Client::new(),
            models,
            custom_headers,
        }
    }

    /// OpenRouter - OpenAI-compatible with optional referer headers
    pub fn openrouter(api_key: String, models: Vec<String>) -> Self {
        Self::with_headers(
            api_key,
            Some("https://openrouter.ai/api/v1".to_string()),
            models,
            vec![
                ("HTTP-Referer".to_string(), "https://github.com/bahkchanhee/claude-code-mux".to_string()),
                ("X-Title".to_string(), "Claude Code Mux".to_string()),
            ],
        )
    }

    /// Deepinfra - Fully OpenAI-compatible
    pub fn deepinfra(api_key: String, models: Vec<String>) -> Self {
        Self::new(
            api_key,
            Some("https://api.deepinfra.com/v1/openai".to_string()),
            models,
        )
    }

    /// NovitaAI - OpenAI-compatible with source header
    pub fn novita(api_key: String, models: Vec<String>) -> Self {
        Self::with_headers(
            api_key,
            Some("https://api.novita.ai/v3/openai".to_string()),
            models,
            vec![("X-Novita-Source".to_string(), "claude-code-mux".to_string())],
        )
    }

    /// Baseten - OpenAI-compatible
    pub fn baseten(api_key: String, models: Vec<String>) -> Self {
        Self::new(
            api_key,
            Some("https://inference.baseten.co/v1".to_string()),
            models,
        )
    }

    /// Together AI - OpenAI-compatible
    pub fn together(api_key: String, models: Vec<String>) -> Self {
        Self::new(
            api_key,
            Some("https://api.together.xyz/v1".to_string()),
            models,
        )
    }

    /// Fireworks AI - OpenAI-compatible
    pub fn fireworks(api_key: String, models: Vec<String>) -> Self {
        Self::new(
            api_key,
            Some("https://api.fireworks.ai/inference/v1".to_string()),
            models,
        )
    }

    /// Groq - Fast OpenAI-compatible inference
    pub fn groq(api_key: String, models: Vec<String>) -> Self {
        Self::new(
            api_key,
            Some("https://api.groq.com/openai/v1".to_string()),
            models,
        )
    }

    /// Nebius - OpenAI-compatible
    pub fn nebius(api_key: String, models: Vec<String>) -> Self {
        Self::new(
            api_key,
            Some("https://api.studio.nebius.ai/v1".to_string()),
            models,
        )
    }

    /// Cerebras - Fast OpenAI-compatible inference
    pub fn cerebras(api_key: String, models: Vec<String>) -> Self {
        Self::new(
            api_key,
            Some("https://api.cerebras.ai/v1".to_string()),
            models,
        )
    }

    pub fn moonshot(api_key: String, models: Vec<String>) -> Self {
        Self::new(
            api_key,
            Some("https://api.moonshot.cn/v1".to_string()),
            models,
        )
    }

    /// Transform Anthropic request to OpenAI format
    fn transform_request(&self, request: &AnthropicRequest) -> Result<OpenAIRequest, ProviderError> {
        let mut openai_messages = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            let system_text = match system {
                crate::models::SystemPrompt::Text(text) => text.clone(),
                crate::models::SystemPrompt::Blocks(blocks) => {
                    blocks.iter()
                        .map(|b| b.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            };
            openai_messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIContent::String(system_text)),
                reasoning: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Transform messages
        for msg in &request.messages {
            match &msg.content {
                MessageContent::Text(text) => {
                    // Simple text message
                    openai_messages.push(OpenAIMessage {
                        role: msg.role.clone(),
                        content: Some(OpenAIContent::String(text.clone())),
                        reasoning: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                MessageContent::Blocks(blocks) => {
                    // Check if we have any tool results - they need separate messages
                    let tool_results: Vec<_> = blocks.iter()
                        .filter_map(|block| {
                            if let crate::models::ContentBlock::ToolResult { tool_use_id, content } = block {
                                Some((tool_use_id.clone(), content.clone()))
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Extract tool_calls from ToolUse blocks
                    let tool_calls: Vec<_> = blocks.iter()
                        .filter_map(|block| {
                            if let crate::models::ContentBlock::ToolUse { id, name, input } = block {
                                Some(OpenAIToolCall {
                                    id: id.clone(),
                                    r#type: "function".to_string(),
                                    function: OpenAIFunctionCall {
                                        name: name.clone(),
                                        arguments: serde_json::to_string(input).unwrap_or_default(),
                                    },
                                })
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Build content parts (text and images, excluding tool use/result)
                    let mut content_parts = Vec::new();
                    for block in blocks {
                        match block {
                            crate::models::ContentBlock::Text { text } => {
                                content_parts.push(OpenAIContentPart::Text {
                                    text: text.clone(),
                                });
                            }
                            crate::models::ContentBlock::Image { source } => {
                                // Convert Anthropic image format to OpenAI format
                                let url = if source.r#type == "base64" {
                                    // data:image/{media_type};base64,{data}
                                    let media_type = source.media_type.as_ref()
                                        .map(|s| s.as_str())
                                        .unwrap_or("image/png");
                                    let data = source.data.as_ref()
                                        .map(|s| s.as_str())
                                        .unwrap_or("");
                                    format!("data:{};base64,{}", media_type, data)
                                } else if let Some(url) = &source.url {
                                    url.clone()
                                } else {
                                    continue; // Skip invalid image sources
                                };

                                content_parts.push(OpenAIContentPart::ImageUrl {
                                    image_url: OpenAIImageUrl { url },
                                });
                            }
                            crate::models::ContentBlock::ToolUse { .. } => {
                                // Already handled in tool_calls
                            }
                            crate::models::ContentBlock::ToolResult { .. } => {
                                // Will be handled as separate messages below
                            }
                            crate::models::ContentBlock::Thinking { .. } => {
                                // OpenAI doesn't have thinking blocks, skip
                            }
                        }
                    }

                    // Add main message with content and/or tool_calls
                    if !content_parts.is_empty() || !tool_calls.is_empty() {
                        let content = if content_parts.is_empty() {
                            None
                        } else if content_parts.len() == 1 {
                            // Single text part - use string format for compatibility
                            if let OpenAIContentPart::Text { text } = &content_parts[0] {
                                Some(OpenAIContent::String(text.clone()))
                            } else {
                                Some(OpenAIContent::Parts(content_parts.clone()))
                            }
                        } else {
                            Some(OpenAIContent::Parts(content_parts))
                        };

                        openai_messages.push(OpenAIMessage {
                            role: msg.role.clone(),
                            content,
                            reasoning: None,
                            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                            tool_call_id: None,
                        });
                    }

                    // Add separate tool result messages
                    for (tool_use_id, result_content) in tool_results {
                        openai_messages.push(OpenAIMessage {
                            role: "tool".to_string(),
                            content: Some(OpenAIContent::String(result_content)),
                            reasoning: None,
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id),
                        });
                    }
                }
            }
        }

        // Transform tools if present
        let tools = request.tools.as_ref().map(|anthropic_tools| {
            anthropic_tools.iter()
                .filter_map(|tool| {
                    // Anthropic tools have name, description, input_schema
                    Some(OpenAITool {
                        r#type: "function".to_string(),
                        function: OpenAIFunctionDef {
                            name: tool.name.as_ref()?.clone(),
                            description: tool.description.clone(),
                            parameters: tool.input_schema.clone(),
                        },
                    })
                })
                .collect()
        });

        Ok(OpenAIRequest {
            model: request.model.clone(),
            messages: openai_messages,
            max_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
            stream: request.stream,
            tools,
            tool_choice: None, // TODO: Add tool_choice support if needed
        })
    }

    /// Transform OpenAI response to Anthropic format
    fn transform_response(&self, response: OpenAIResponse) -> ProviderResponse {
        let choice = response.choices.into_iter().next()
            .expect("OpenAI response must have at least one choice");

        // Extract text from content or reasoning (for GLM models via Cerebras)
        let text = if let Some(content) = choice.message.content {
            match content {
                OpenAIContent::String(s) => s,
                OpenAIContent::Parts(parts) => {
                    // Extract text from all text parts
                    parts.iter()
                        .filter_map(|part| {
                            if let OpenAIContentPart::Text { text } = part {
                                Some(text.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
        } else if let Some(reasoning) = choice.message.reasoning {
            reasoning
        } else {
            String::new()
        };

        ProviderResponse {
            id: response.id,
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text,
            }],
            model: response.model,
            stop_reason: choice.finish_reason,
            stop_sequence: None,
            usage: Usage {
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
            },
        }
    }

    /// Transform Responses API response to Anthropic format
    fn transform_responses_response(&self, response: OpenAIResponsesResponse) -> ProviderResponse {
        // Extract text from output messages
        let text = response.output.iter()
            .filter(|output| output.output_type == "message")
            .filter_map(|output| output.content.as_ref())
            .flat_map(|content_blocks| {
                content_blocks.iter()
                    .filter(|block| block.block_type == "output_text")
                    .filter_map(|block| block.text.clone())
            })
            .collect::<Vec<_>>()
            .join("\n");

        ProviderResponse {
            id: response.id,
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text,
            }],
            model: response.model,
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
            },
        }
    }
}

#[async_trait]
impl AnthropicProvider for OpenAIProvider {
    async fn send_message(&self, request: AnthropicRequest) -> Result<ProviderResponse, ProviderError> {
        // Check if this is a Codex model
        let is_codex = Self::is_codex_model(&request.model);

        if is_codex {
            // Use /v1/responses endpoint for Codex models
            let responses_request = self.transform_to_responses_request(&request)?;
            let url = format!("{}/responses", self.base_url);

            tracing::debug!("Using /v1/responses endpoint for Codex model: {}", request.model);

            let mut req_builder = self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json");

            // Add custom headers
            for (key, value) in &self.custom_headers {
                req_builder = req_builder.header(key, value);
            }

            let response = req_builder
                .json(&responses_request)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                tracing::error!("Responses API error ({}): {}", status, error_text);
                return Err(ProviderError::ApiError {
                    status,
                    message: error_text,
                });
            }

            let response_text = response.text().await?;
            tracing::debug!("Responses API response body: {}", response_text);

            // Parse as Responses API response
            let responses_response: OpenAIResponsesResponse = serde_json::from_str(&response_text)
                .map_err(|e| {
                    tracing::error!("Failed to parse Responses API response: {}", e);
                    tracing::error!("Response body was: {}", response_text);
                    e
                })?;

            Ok(self.transform_responses_response(responses_response))
        } else {
            // Use standard /v1/chat/completions endpoint for non-Codex models
            let openai_request = self.transform_request(&request)?;
            let url = format!("{}/chat/completions", self.base_url);

            let mut req_builder = self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json");

            // Add custom headers (for OpenRouter, NovitaAI, etc.)
            for (key, value) in &self.custom_headers {
                req_builder = req_builder.header(key, value);
            }

            let response = req_builder
                .json(&openai_request)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                return Err(ProviderError::ApiError {
                    status,
                    message: error_text,
                });
            }

            // Get response body as text for debugging
            let response_text = response.text().await?;
            tracing::debug!("OpenAI provider response body: {}", response_text);

            // Try to parse the response
            let openai_response: OpenAIResponse = serde_json::from_str(&response_text)
                .map_err(|e| {
                    tracing::error!("Failed to parse OpenAI response: {}", e);
                    tracing::error!("Response body was: {}", response_text);
                    e
                })?;

            Ok(self.transform_response(openai_response))
        }
    }

    async fn count_tokens(&self, request: CountTokensRequest) -> Result<CountTokensResponse, ProviderError> {
        // For OpenAI, we'll use tiktoken-rs for local token counting
        // This is a placeholder - actual implementation would use tiktoken

        // Rough estimate: ~4 chars per token
        let mut total_chars = 0;

        if let Some(ref system) = request.system {
            let system_text = match system {
                crate::models::SystemPrompt::Text(text) => text.clone(),
                crate::models::SystemPrompt::Blocks(blocks) => {
                    blocks.iter().map(|b| b.text.clone()).collect::<Vec<_>>().join("\n")
                }
            };
            total_chars += system_text.len();
        }

        for msg in &request.messages {
            let content = match &msg.content {
                MessageContent::Text(text) => text.clone(),
                MessageContent::Blocks(blocks) => {
                    blocks.iter()
                        .filter_map(|block| {
                            match block {
                                crate::models::ContentBlock::Text { text } => Some(text.clone()),
                                _ => None,
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            };
            total_chars += content.len();
        }

        let estimated_tokens = (total_chars / 4) as u32;

        Ok(CountTokensResponse {
            input_tokens: estimated_tokens,
        })
    }

    async fn send_message_stream(
        &self,
        request: AnthropicRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Bytes, ProviderError>> + Send>>, ProviderError> {
        use futures::stream::TryStreamExt;

        // Check if this is a Codex model
        let is_codex = Self::is_codex_model(&request.model);

        let (url, request_body) = if is_codex {
            // Use /v1/responses endpoint for Codex models
            tracing::debug!("Using /v1/responses endpoint for Codex model (streaming): {}", request.model);
            let responses_request = self.transform_to_responses_request(&request)?;
            let body = serde_json::to_value(&responses_request)
                .map_err(|e| ProviderError::SerializationError(e))?;
            (format!("{}/responses", self.base_url), body)
        } else {
            // Use standard /v1/chat/completions endpoint
            let openai_request = self.transform_request(&request)?;
            let body = serde_json::to_value(&openai_request)
                .map_err(|e| ProviderError::SerializationError(e))?;
            (format!("{}/chat/completions", self.base_url), body)
        };

        // Send streaming request
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        // Check for errors
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::ApiError {
                status,
                message: error_text,
            });
        }

        // TODO: Transform OpenAI SSE format to Anthropic SSE format
        // For now, just pass through the stream
        let stream = response.bytes_stream().map_err(|e| ProviderError::HttpError(e));

        Ok(Box::pin(stream))
    }

    fn supports_model(&self, model: &str) -> bool {
        self.models.iter().any(|m| m == model)
    }
}

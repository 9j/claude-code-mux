use super::{AnthropicProvider, ProviderError, ProviderResponse, Usage};
use crate::auth::{OAuthClient, OAuthConfig, TokenStore};
use crate::models::{AnthropicRequest, ContentBlock, MessageContent, SystemPrompt};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Google Gemini provider supporting three authentication methods:
/// 1. OAuth 2.0 (Google AI Pro/Ultra) - Uses Code Assist API
/// 2. API Key (Google AI Studio) - Uses public Gemini API
/// 3. Vertex AI (Google Cloud) - Uses Vertex AI API
pub struct GeminiProvider {
    pub name: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub models: Vec<String>,
    pub client: Client,
    pub custom_headers: HashMap<String, String>,
    // Vertex AI fields
    pub project_id: Option<String>,
    pub location: Option<String>,
    // OAuth fields
    pub oauth_provider_id: Option<String>,
    pub token_store: Option<TokenStore>,
}

/// Remove JSON Schema metadata fields that Gemini API doesn't support
fn clean_json_schema(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            // Remove JSON Schema metadata fields
            map.remove("$schema");
            map.remove("$id");
            map.remove("$ref");
            map.remove("$comment");
            map.remove("exclusiveMinimum");
            map.remove("exclusiveMaximum");
            map.remove("definitions");
            map.remove("$defs");

            // Recursively clean nested objects
            for (_, v) in map.iter_mut() {
                clean_json_schema(v);
            }
        }
        serde_json::Value::Array(arr) => {
            // Recursively clean array elements
            for item in arr.iter_mut() {
                clean_json_schema(item);
            }
        }
        _ => {}
    }
}

impl GeminiProvider {
    pub fn new(
        name: String,
        api_key: Option<String>,
        base_url: Option<String>,
        models: Vec<String>,
        custom_headers: HashMap<String, String>,
        oauth_provider_id: Option<String>,
        token_store: Option<TokenStore>,
        project_id: Option<String>,
        location: Option<String>,
    ) -> Self {
        let base_url = base_url.unwrap_or_else(|| {
            if oauth_provider_id.is_some() {
                // Code Assist API (OAuth)
                "https://cloudcode-pa.googleapis.com/v1internal".to_string()
            } else if project_id.is_some() && location.is_some() {
                // Vertex AI
                format!(
                    "https://{}-aiplatform.googleapis.com/v1",
                    location.as_ref().unwrap()
                )
            } else {
                // Google AI (API Key)
                "https://generativelanguage.googleapis.com/v1beta".to_string()
            }
        });

        Self {
            name,
            api_key,
            base_url,
            models,
            client: Client::new(),
            custom_headers,
            project_id,
            location,
            oauth_provider_id,
            token_store,
        }
    }

    /// Check if this provider uses OAuth (Code Assist API)
    fn is_oauth(&self) -> bool {
        self.oauth_provider_id.is_some() && self.token_store.is_some()
    }

    /// Check if this provider uses Vertex AI
    fn is_vertex_ai(&self) -> bool {
        self.project_id.is_some() && self.location.is_some()
    }

    /// Get OAuth bearer token (with automatic refresh)
    async fn get_auth_header(&self) -> Result<Option<String>, ProviderError> {
        if let (Some(oauth_provider_id), Some(token_store)) =
            (&self.oauth_provider_id, &self.token_store)
        {
            if let Some(token) = token_store.get(oauth_provider_id) {
                // Check if token needs refresh
                if token.needs_refresh() {
                    tracing::info!("🔄 Token for '{}' needs refresh, refreshing...", oauth_provider_id);

                    // Refresh token
                    let config = OAuthConfig::gemini();
                    let oauth_client = OAuthClient::new(config, token_store.clone());

                    match oauth_client.refresh_token(oauth_provider_id).await {
                        Ok(new_token) => {
                            tracing::info!("✅ Token refreshed successfully");
                            return Ok(Some(format!("Bearer {}", new_token.access_token)));
                        }
                        Err(e) => {
                            tracing::error!("❌ Failed to refresh token: {}", e);
                            return Err(ProviderError::AuthError(format!(
                                "Failed to refresh OAuth token: {}", e
                            )));
                        }
                    }
                } else {
                    // Token is still valid
                    return Ok(Some(format!("Bearer {}", token.access_token)));
                }
            } else {
                return Err(ProviderError::AuthError(format!(
                    "OAuth provider '{}' configured but no token found in store",
                    oauth_provider_id
                )));
            }
        }
        Ok(None)
    }

    /// Transform Anthropic request to Gemini format
    fn transform_request(
        &self,
        request: &AnthropicRequest,
    ) -> Result<GeminiRequest, ProviderError> {
        // Transform system prompt
        let system_instruction = request.system.as_ref().map(|system| {
            let text = match system {
                SystemPrompt::Text(text) => text.clone(),
                SystemPrompt::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            GeminiSystemInstruction {
                parts: vec![GeminiPart::Text { text }],
            }
        });

        // Transform messages
        let mut contents = Vec::new();
        for msg in &request.messages {
            let role = match msg.role.as_str() {
                "user" => "user",
                "assistant" => "model",
                _ => continue,
            };

            let parts = match &msg.content {
                MessageContent::Text(text) => {
                    vec![GeminiPart::Text {
                        text: text.clone(),
                    }]
                }
                MessageContent::Blocks(blocks) => {
                    let mut parts = Vec::new();
                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => {
                                parts.push(GeminiPart::Text {
                                    text: text.clone(),
                                });
                            }
                            ContentBlock::Image { source } => {
                                // Convert to Gemini inline_data format
                                if let (Some(media_type), Some(data)) =
                                    (&source.media_type, &source.data)
                                {
                                    parts.push(GeminiPart::InlineData {
                                        inline_data: GeminiInlineData {
                                            mime_type: media_type.clone(),
                                            data: data.clone(),
                                        },
                                    });
                                }
                            }
                            ContentBlock::Thinking { thinking, .. } => {
                                // Gemini doesn't have thinking blocks, convert to text
                                parts.push(GeminiPart::Text {
                                    text: thinking.clone(),
                                });
                            }
                            _ => {
                                // Skip tool use/result for now
                            }
                        }
                    }
                    parts
                }
            };

            contents.push(GeminiContent {
                role: role.to_string(),
                parts,
            });
        }

        // Transform generation config
        let generation_config = GeminiGenerationConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: Some(40), // Gemini default
            max_output_tokens: Some(request.max_tokens as i32),
            stop_sequences: request.stop_sequences.clone(),
        };

        // Transform tools if present
        let tools = request.tools.as_ref().map(|anthropic_tools| {
            vec![GeminiTool {
                function_declarations: anthropic_tools
                    .iter()
                    .filter_map(|tool| {
                        let mut parameters = tool.input_schema.clone().unwrap_or_default();
                        // Clean JSON Schema metadata that Gemini doesn't support
                        clean_json_schema(&mut parameters);

                        Some(GeminiFunctionDeclaration {
                            name: tool.name.as_ref()?.clone(),
                            description: tool.description.clone().unwrap_or_default(),
                            parameters,
                        })
                    })
                    .collect(),
            }]
        });

        Ok(GeminiRequest {
            contents,
            system_instruction,
            generation_config: Some(generation_config),
            tools,
        })
    }

    /// Transform Gemini response to Anthropic format
    fn transform_response(
        &self,
        response: GeminiResponse,
        model: String,
    ) -> Result<ProviderResponse, ProviderError> {
        let candidate = response
            .candidates
            .first()
            .ok_or_else(|| ProviderError::ApiError {
                status: 500,
                message: "No candidates in response".to_string(),
            })?;

        let content = candidate
            .content
            .parts
            .iter()
            .map(|part| match part {
                GeminiPart::Text { text } => ContentBlock::Text {
                    text: text.clone(),
                },
                _ => ContentBlock::Text {
                    text: String::new(),
                },
            })
            .collect();

        let stop_reason = match candidate.finish_reason.as_deref() {
            Some("STOP") => Some("end_turn".to_string()),
            Some("MAX_TOKENS") => Some("max_tokens".to_string()),
            _ => None,
        };

        let usage = Usage {
            input_tokens: response
                .usage_metadata
                .as_ref()
                .and_then(|u| u.prompt_token_count)
                .unwrap_or(0) as u32,
            output_tokens: response
                .usage_metadata
                .as_ref()
                .and_then(|u| u.candidates_token_count)
                .unwrap_or(0) as u32,
        };

        Ok(ProviderResponse {
            id: format!("gemini-{}", chrono::Utc::now().timestamp_millis()),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content,
            model,
            stop_reason,
            stop_sequence: None,
            usage,
        })
    }
}

#[async_trait]
impl AnthropicProvider for GeminiProvider {
    async fn send_message(
        &self,
        request: AnthropicRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let model = request.model.clone();

        // Check if using OAuth (Code Assist API)
        if self.is_oauth() {
            // Use Code Assist API endpoint
            let gemini_request = self.transform_request(&request)?;

            // Get OAuth bearer token
            let auth_header = self.get_auth_header().await?;
            let bearer_token = auth_header.ok_or_else(|| {
                ProviderError::AuthError("OAuth configured but no token available".to_string())
            })?;

            // Get project_id from token store
            let project_id = if let (Some(oauth_provider_id), Some(token_store)) =
                (&self.oauth_provider_id, &self.token_store) {
                token_store
                    .get(oauth_provider_id)
                    .and_then(|token| token.project_id.clone())
            } else {
                None
            };

            if project_id.is_none() {
                tracing::warn!("⚠️ No project_id found in token for Gemini OAuth. Code Assist API may fail.");
            }

            // Generate unique user_prompt_id
            let user_prompt_id = format!("gemini-{}", chrono::Utc::now().timestamp_millis());

            // Wrap in Code Assist API format
            let code_assist_request = CodeAssistRequest {
                model: model.clone(),
                project: project_id,
                user_prompt_id: Some(user_prompt_id),
                request: CodeAssistInnerRequest {
                    contents: gemini_request.contents,
                    system_instruction: gemini_request.system_instruction,
                    generation_config: gemini_request.generation_config,
                    tools: gemini_request.tools,
                    session_id: None, // Optional
                },
            };

            // Code Assist API endpoint: https://cloudcode-pa.googleapis.com/v1internal:generateContent
            let url = format!("{}:generateContent", self.base_url);

            tracing::debug!("🔐 Using OAuth Code Assist API: {}", url);

            // Build request
            let mut req_builder = self.client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("Authorization", bearer_token);

            // Add custom headers
            for (key, value) in &self.custom_headers {
                req_builder = req_builder.header(key, value);
            }

            // Send request
            let response = req_builder.json(&code_assist_request).send().await?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());

                // Special handling for 404 errors (model not found)
                if status == 404 {
                    let model_name = &model;
                    let user_friendly_msg = if model_name.contains("gemini-3") || model_name.contains("preview") {
                        format!(
                            "Model '{}' is not available. This may be a preview model that requires special access. \
                            Try using gemini-2.5-pro or gemini-2.0-flash-exp instead. \
                            Original error: {}",
                            model_name, error_text
                        )
                    } else {
                        format!("Model '{}' not found. Original error: {}", model_name, error_text)
                    };
                    tracing::warn!("⚠️ Model not found (404): {}", user_friendly_msg);
                    return Err(ProviderError::ApiError {
                        status,
                        message: user_friendly_msg,
                    });
                }

                tracing::error!("Code Assist API error ({}): {}", status, error_text);
                return Err(ProviderError::ApiError {
                    status,
                    message: error_text,
                });
            }

            // Parse Code Assist response
            let code_assist_response: CodeAssistResponse = response.json().await?;
            self.transform_response(code_assist_response.response, model)
        } else {
            // Use public Gemini API or Vertex AI
            let gemini_request = self.transform_request(&request)?;

            // Build URL
            let url = if self.is_vertex_ai() {
                // Vertex AI endpoint
                format!(
                    "{}/projects/{}/locations/{}/publishers/google/models/{}:generateContent",
                    self.base_url,
                    self.project_id.as_ref().unwrap(),
                    self.location.as_ref().unwrap(),
                    model
                )
            } else if self.api_key.is_some() {
                // API Key endpoint (key in query parameter)
                format!(
                    "{}/models/{}:generateContent?key={}",
                    self.base_url,
                    model,
                    self.api_key.as_ref().unwrap()
                )
            } else {
                return Err(ProviderError::ConfigError(
                    "Gemini provider requires either api_key, OAuth, or Vertex AI configuration".to_string()
                ));
            };

            // Build request
            let mut req_builder = self.client.post(&url).header("Content-Type", "application/json");

            // Add custom headers
            for (key, value) in &self.custom_headers {
                req_builder = req_builder.header(key, value);
            }

            // Send request
            let response = req_builder.json(&gemini_request).send().await?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                tracing::error!("Gemini API error ({}): {}", status, error_text);
                return Err(ProviderError::ApiError {
                    status,
                    message: error_text,
                });
            }

            let gemini_response: GeminiResponse = response.json().await?;
            self.transform_response(gemini_response, model)
        }
    }

    async fn send_message_stream(
        &self,
        _request: AnthropicRequest,
    ) -> Result<std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<bytes::Bytes, ProviderError>> + Send>>, ProviderError> {
        // TODO: Implement streaming for Gemini
        Err(ProviderError::ConfigError(
            "Streaming not yet implemented for Gemini".to_string(),
        ))
    }

    async fn count_tokens(
        &self,
        _request: crate::models::CountTokensRequest,
    ) -> Result<crate::models::CountTokensResponse, ProviderError> {
        // TODO: Implement token counting for Gemini
        Err(ProviderError::ConfigError(
            "Token counting not yet implemented for Gemini".to_string(),
        ))
    }

    fn supports_model(&self, model: &str) -> bool {
        self.models.contains(&model.to_string())
    }
}

// Gemini API structures

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: GeminiInlineData },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: Option<i32>,
    candidates_token_count: Option<i32>,
    total_token_count: Option<i32>,
}

// Code Assist API structures (for OAuth)

#[derive(Debug, Serialize)]
struct CodeAssistRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_prompt_id: Option<String>,
    request: CodeAssistInnerRequest,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CodeAssistInnerRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodeAssistResponse {
    response: GeminiResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_id: Option<String>,
}

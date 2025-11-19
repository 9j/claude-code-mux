use super::{AnthropicProvider, ProviderError, ProviderResponse, Usage};
use crate::auth::TokenStore;
use crate::models::{AnthropicRequest, ContentBlock, MessageContent, SystemPrompt};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Google Gemini provider supporting three authentication methods:
/// 1. OAuth 2.0 (Google AI Pro/Ultra)
/// 2. API Key (Google AI Studio)
/// 3. Vertex AI (Google Cloud)
pub struct GeminiProvider {
    pub name: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub models: Vec<String>,
    pub client: Client,
    pub custom_headers: HashMap<String, String>,
    // OAuth fields
    pub oauth_provider_id: Option<String>,
    pub token_store: Option<TokenStore>,
    // Vertex AI fields
    pub project_id: Option<String>,
    pub location: Option<String>,
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
            if project_id.is_some() && location.is_some() {
                // Vertex AI
                format!(
                    "https://{}-aiplatform.googleapis.com/v1",
                    location.as_ref().unwrap()
                )
            } else {
                // Google AI (OAuth or API Key)
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
            oauth_provider_id,
            token_store,
            project_id,
            location,
        }
    }

    /// Check if this provider uses OAuth
    fn is_oauth(&self) -> bool {
        self.oauth_provider_id.is_some() && self.token_store.is_some()
    }

    /// Check if this provider uses Vertex AI
    fn is_vertex_ai(&self) -> bool {
        self.project_id.is_some() && self.location.is_some()
    }

    /// Get authorization header value (OAuth token or API key)
    async fn get_auth_header(&self) -> Result<Option<String>, ProviderError> {
        if let (Some(oauth_provider_id), Some(token_store)) =
            (&self.oauth_provider_id, &self.token_store)
        {
            // OAuth: Get and refresh token if needed
            if let Some(token) = token_store.get(oauth_provider_id) {
                // Check if token needs refresh
                if token.needs_refresh() {
                    tracing::info!(
                        "🔄 Token for '{}' needs refresh, refreshing...",
                        oauth_provider_id
                    );

                    // Refresh token
                    let config = crate::auth::OAuthConfig::gemini();
                    let oauth_client = crate::auth::OAuthClient::new(config, token_store.clone());

                    match oauth_client.refresh_token(oauth_provider_id).await {
                        Ok(new_token) => {
                            tracing::info!("✅ Token refreshed successfully");
                            return Ok(Some(format!("Bearer {}", new_token.access_token)));
                        }
                        Err(e) => {
                            tracing::error!("❌ Failed to refresh token: {}", e);
                            return Err(ProviderError::AuthError(format!(
                                "Failed to refresh OAuth token: {}",
                                e
                            )));
                        }
                    }
                } else {
                    // Token is still valid
                    return Ok(Some(format!("Bearer {}", token.access_token)));
                }
            } else {
                return Err(ProviderError::AuthError(format!(
                    "No OAuth token found for provider '{}'",
                    oauth_provider_id
                )));
            }
        } else if self.api_key.is_some() {
            // API Key: Will be added as query parameter, not header
            Ok(None)
        } else {
            // Vertex AI: Uses Application Default Credentials (handled externally)
            Ok(None)
        }
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
                        Some(GeminiFunctionDeclaration {
                            name: tool.name.as_ref()?.clone(),
                            description: tool.description.clone().unwrap_or_default(),
                            parameters: tool.input_schema.clone().unwrap_or_default(),
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
            // OAuth endpoint
            format!("{}/models/{}:generateContent", self.base_url, model)
        };

        // Build request
        let mut req_builder = self.client.post(&url).header("Content-Type", "application/json");

        // Add authorization header for OAuth or Vertex AI
        if let Some(auth_header) = self.get_auth_header().await? {
            req_builder = req_builder.header("Authorization", auth_header);
        }

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

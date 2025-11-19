pub mod error;
pub mod openai;
pub mod anthropic_compatible;
pub mod gemini;
pub mod registry;
pub mod streaming;

use async_trait::async_trait;
use crate::models::{AnthropicRequest, CountTokensRequest, CountTokensResponse, ContentBlock};
use error::ProviderError;
use serde::{Deserialize, Serialize};
use bytes::Bytes;
use futures::stream::Stream;
use std::pin::Pin;

/// Provider response that maintains Anthropic API compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Main provider trait - all providers must implement this
/// Maintains Anthropic Messages API compatibility
#[async_trait]
pub trait AnthropicProvider: Send + Sync {
    /// Send a message request to the provider
    /// Must transform to/from Anthropic format as needed
    async fn send_message(&self, request: AnthropicRequest) -> Result<ProviderResponse, ProviderError>;

    /// Send a streaming message request to the provider
    /// Returns a stream of raw bytes (SSE format)
    async fn send_message_stream(
        &self,
        request: AnthropicRequest
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Bytes, ProviderError>> + Send>>, ProviderError>;

    /// Count tokens for a request
    /// Provider-specific implementation (tiktoken for OpenAI, etc.)
    async fn count_tokens(&self, request: CountTokensRequest) -> Result<CountTokensResponse, ProviderError>;

    /// Check if provider supports a specific model
    fn supports_model(&self, model: &str) -> bool;
}

/// Authentication type for providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AuthType {
    /// API key authentication
    ApiKey,
    /// OAuth 2.0 authentication
    OAuth,
}

impl Default for AuthType {
    fn default() -> Self {
        AuthType::ApiKey
    }
}

/// Provider configuration from TOML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub provider_type: String,

    /// Authentication type (default: api_key)
    #[serde(default)]
    pub auth_type: AuthType,

    /// API key (required for auth_type = "apikey")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// OAuth provider ID (required for auth_type = "oauth")
    /// References a token stored in TokenStore
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_provider: Option<String>,

    /// Google Cloud Project ID (for Vertex AI provider)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,

    /// Location/Region (for Vertex AI provider)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,

    pub base_url: Option<String>,
    pub models: Vec<String>,
    pub enabled: Option<bool>,
}

impl ProviderConfig {
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }

    /// Get the API key or OAuth provider ID
    pub fn get_auth_credential(&self) -> Option<String> {
        match self.auth_type {
            AuthType::ApiKey => self.api_key.clone(),
            AuthType::OAuth => self.oauth_provider.clone(),
        }
    }
}

// Re-export provider implementations
pub use openai::OpenAIProvider;
pub use anthropic_compatible::AnthropicCompatibleProvider;
pub use registry::ProviderRegistry;

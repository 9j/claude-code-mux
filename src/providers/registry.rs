use super::{AnthropicProvider, ProviderConfig, OpenAIProvider, AnthropicCompatibleProvider, error::ProviderError};
use super::gemini::GeminiProvider;
use crate::auth::TokenStore;
use std::collections::HashMap;
use std::sync::Arc;

/// Provider registry that manages all configured providers
pub struct ProviderRegistry {
    /// Map of provider name -> provider instance
    providers: HashMap<String, Arc<Box<dyn AnthropicProvider>>>,
    /// Map of model name -> provider name for fast lookup
    model_to_provider: HashMap<String, String>,
}

impl ProviderRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            model_to_provider: HashMap::new(),
        }
    }

    /// Load providers from configuration
    pub fn from_configs(configs: &[ProviderConfig], token_store: Option<TokenStore>) -> Result<Self, ProviderError> {
        let mut registry = Self::new();

        for config in configs {
            // Skip disabled providers
            if !config.is_enabled() {
                continue;
            }

            // Get API key - required for API key auth, skipped for OAuth
            let api_key = match &config.auth_type {
                super::AuthType::ApiKey => {
                    config.api_key.clone().ok_or_else(|| {
                        ProviderError::ConfigError(
                            format!("Provider '{}' requires api_key for ApiKey auth", config.name)
                        )
                    })?
                }
                super::AuthType::OAuth => {
                    // OAuth providers will handle authentication differently
                    // For now, use a placeholder - will be replaced with token
                    config.oauth_provider.clone().unwrap_or_else(|| config.name.clone())
                }
            };

            // Create provider instance based on type
            let provider: Box<dyn AnthropicProvider> = match config.provider_type.as_str() {
                // OpenAI
                "openai" => Box::new(OpenAIProvider::new(
                    config.name.clone(),
                    api_key,
                    config.base_url.clone().unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                    config.models.clone(),
                    config.oauth_provider.clone(),
                    token_store.clone(),
                )),

                // Anthropic-compatible providers
                "anthropic" => Box::new(AnthropicCompatibleProvider::new(
                    config.name.clone(),
                    api_key,
                    config.base_url.clone().unwrap_or_else(|| "https://api.anthropic.com".to_string()),
                    config.models.clone(),
                    config.oauth_provider.clone(),
                    token_store.clone(),
                )),
                "z.ai" => Box::new(AnthropicCompatibleProvider::zai(
                    api_key,
                    config.models.clone(),
                    token_store.clone(),
                )),
                "minimax" => Box::new(AnthropicCompatibleProvider::minimax(
                    api_key,
                    config.models.clone(),
                    token_store.clone(),
                )),
                "zenmux" => Box::new(AnthropicCompatibleProvider::zenmux(
                    api_key,
                    config.models.clone(),
                    token_store.clone(),
                )),
                "kimi-coding" => Box::new(AnthropicCompatibleProvider::kimi_coding(
                    api_key,
                    config.models.clone(),
                    token_store.clone(),
                )),

                // OpenAI-compatible providers
                "openrouter" => Box::new(OpenAIProvider::openrouter(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "deepinfra" => Box::new(OpenAIProvider::deepinfra(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "novita" => Box::new(OpenAIProvider::novita(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "baseten" => Box::new(OpenAIProvider::baseten(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "together" => Box::new(OpenAIProvider::together(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "fireworks" => Box::new(OpenAIProvider::fireworks(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "groq" => Box::new(OpenAIProvider::groq(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "nebius" => Box::new(OpenAIProvider::nebius(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "cerebras" => Box::new(OpenAIProvider::cerebras(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),
                "moonshot" => Box::new(OpenAIProvider::moonshot(
                    config.name.clone(),
                    api_key,
                    config.models.clone(),
                )),

                // Google Gemini (supports OAuth, API Key, Vertex AI)
                "gemini" => {
                    let api_key_opt = if config.auth_type == super::AuthType::ApiKey {
                        Some(api_key.clone())
                    } else {
                        None
                    };

                    Box::new(GeminiProvider::new(
                        config.name.clone(),
                        api_key_opt,
                        config.base_url.clone(),
                        config.models.clone(),
                        HashMap::new(), // custom headers
                        config.oauth_provider.clone(),
                        token_store.clone(),
                        None, // No project_id/location for Gemini (AI Studio/OAuth only)
                        None,
                    ))
                }

                "vertex-ai" => {
                    // Vertex AI provider (separate from Gemini)
                    // Uses Google Cloud Vertex AI with ADC authentication
                    Box::new(GeminiProvider::new(
                        config.name.clone(),
                        None, // No API key for Vertex AI (uses ADC)
                        config.base_url.clone(),
                        config.models.clone(),
                        HashMap::new(), // custom headers
                        None, // No OAuth for Vertex AI
                        token_store.clone(),
                        config.project_id.clone(), // GCP project ID
                        config.location.clone(),   // GCP location
                    ))
                }

                other => {
                    return Err(ProviderError::ConfigError(
                        format!("Unknown provider type: {}", other)
                    ));
                }
            };

            // NOTE: models field in provider config is deprecated
            // Model mappings are now defined in [[models]] section
            // We only register the provider by name

            // Add provider to registry
            registry.providers.insert(config.name.clone(), Arc::new(provider));
        }

        Ok(registry)
    }

    /// Get a provider by name
    pub fn get_provider(&self, name: &str) -> Option<Arc<Box<dyn AnthropicProvider>>> {
        self.providers.get(name).cloned()
    }

    /// Get a provider for a specific model
    pub fn get_provider_for_model(&self, model: &str) -> Result<Arc<Box<dyn AnthropicProvider>>, ProviderError> {
        // First, check if we have a direct model → provider mapping
        if let Some(provider_name) = self.model_to_provider.get(model) {
            if let Some(provider) = self.providers.get(provider_name) {
                return Ok(provider.clone());
            }
        }

        // If no direct mapping, search through all providers
        for provider in self.providers.values() {
            if provider.supports_model(model) {
                return Ok(provider.clone());
            }
        }

        Err(ProviderError::ModelNotSupported(model.to_string()))
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<String> {
        self.model_to_provider.keys().cloned().collect()
    }

    /// List all providers
    pub fn list_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_registry() {
        let registry = ProviderRegistry::new();
        assert!(registry.list_models().is_empty());
        assert!(registry.list_providers().is_empty());
    }

    #[test]
    fn test_get_provider_for_model_not_found() {
        let registry = ProviderRegistry::new();
        let result = registry.get_provider_for_model("gpt-4");
        assert!(result.is_err());
    }
}

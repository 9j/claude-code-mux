use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};

use super::token_store::{OAuthToken, TokenStore};

/// PKCE verifier for OAuth flow
#[derive(Debug, Clone)]
pub struct PKCEVerifier {
    pub verifier: String,
    pub challenge: String,
}

impl PKCEVerifier {
    /// Generate a new PKCE code verifier and challenge
    pub fn generate() -> Self {
        // Generate random verifier (43-128 characters)
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
        let verifier = URL_SAFE_NO_PAD.encode(&random_bytes);

        // Generate challenge (SHA256 of verifier)
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let challenge_bytes = hasher.finalize();
        let challenge = URL_SAFE_NO_PAD.encode(&challenge_bytes);

        Self { verifier, challenge }
    }
}

/// Authorization URL with PKCE
#[derive(Debug, Clone)]
pub struct AuthorizationUrl {
    pub url: String,
    pub verifier: PKCEVerifier,
}

/// OAuth provider configuration
#[derive(Debug, Clone)]
pub struct OAuthConfig {
    pub client_id: String,
    pub client_secret: Option<String>,  // Some providers require client_secret (e.g., Google)
    pub auth_url: String,
    pub token_url: String,
    pub redirect_uri: String,
    pub scopes: Vec<String>,
}

impl OAuthConfig {
    /// Anthropic Claude Pro/Max OAuth configuration
    pub fn anthropic() -> Self {
        Self {
            client_id: "9d1c250a-e61b-44d9-88ed-5944d1962f5e".to_string(),
            client_secret: None,  // PKCE-based public client
            auth_url: "https://claude.ai/oauth/authorize".to_string(),
            token_url: "https://console.anthropic.com/v1/oauth/token".to_string(),
            redirect_uri: "https://console.anthropic.com/oauth/code/callback".to_string(),
            scopes: vec![
                "org:create_api_key".to_string(),
                "user:profile".to_string(),
                "user:inference".to_string(),
            ],
        }
    }

    /// Anthropic Console (for API key creation)
    pub fn anthropic_console() -> Self {
        let mut config = Self::anthropic();
        config.auth_url = "https://console.anthropic.com/oauth/authorize".to_string();
        config
    }

    /// OpenAI ChatGPT Plus/Pro OAuth configuration (for Codex)
    ///
    /// Note: OpenAI's official Codex CLI OAuth app has a fixed redirect_uri.
    /// The client_id "app_EMoamEEZ73f0CkXaXp7hrann" only allows:
    /// - http://localhost:1455/auth/callback
    ///
    /// This is hardcoded in OpenAI's OAuth app registration and cannot be changed.
    pub fn openai_codex() -> Self {
        Self {
            client_id: "app_EMoamEEZ73f0CkXaXp7hrann".to_string(),
            client_secret: None,  // PKCE-based public client
            auth_url: "https://auth.openai.com/oauth/authorize".to_string(),
            token_url: "https://auth.openai.com/oauth/token".to_string(),
            redirect_uri: "http://localhost:1455/auth/callback".to_string(),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
                "offline_access".to_string(),
            ],
        }
    }

    /// Google Gemini (AI Pro/Ultra) OAuth configuration
    ///
    /// Note: This uses Google's official Gemini CLI OAuth app credentials.
    /// The client_id and client_secret are public (as documented in Google's official CLI).
    /// See: https://github.com/google-gemini/gemini-cli
    /// "It's ok to save this in git because this is an installed application"
    /// https://developers.google.com/identity/protocols/oauth2#installed
    pub fn gemini() -> Self {
        Self {
            client_id: "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com".to_string(),
            client_secret: Some("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".to_string()),
            auth_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_url: "https://oauth2.googleapis.com/token".to_string(),
            redirect_uri: "http://localhost:13456/api/oauth/callback".to_string(),
            scopes: vec![
                "https://www.googleapis.com/auth/cloud-platform".to_string(),
                "https://www.googleapis.com/auth/userinfo.email".to_string(),
                "https://www.googleapis.com/auth/userinfo.profile".to_string(),
            ],
        }
    }
}

/// OAuth client for handling authentication flows
pub struct OAuthClient {
    config: OAuthConfig,
    token_store: TokenStore,
    http_client: reqwest::Client,
}

impl OAuthClient {
    /// Create a new OAuth client
    pub fn new(config: OAuthConfig, token_store: TokenStore) -> Self {
        Self {
            config,
            token_store,
            http_client: reqwest::Client::new(),
        }
    }

    /// Generate authorization URL with PKCE
    pub fn get_authorization_url(&self) -> AuthorizationUrl {
        let pkce = PKCEVerifier::generate();

        let mut url = url::Url::parse(&self.config.auth_url)
            .expect("Invalid auth URL");

        // Check provider type based on client_id
        let is_openai_codex = self.config.client_id == "app_EMoamEEZ73f0CkXaXp7hrann";
        let is_gemini = self.config.client_id.starts_with("681255809395-");

        if is_openai_codex {
            // OpenAI uses a separate random state (not the PKCE verifier)
            // Generate random state for CSRF protection
            use rand::Rng;
            let random_bytes: Vec<u8> = (0..16).map(|_| rand::thread_rng().gen()).collect();
            let state = random_bytes.iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>();

            // OpenAI Codex specific parameters
            url.query_pairs_mut()
                .append_pair("response_type", "code")
                .append_pair("client_id", &self.config.client_id)
                .append_pair("redirect_uri", &self.config.redirect_uri)
                .append_pair("scope", &self.config.scopes.join(" "))
                .append_pair("code_challenge", &pkce.challenge)
                .append_pair("code_challenge_method", "S256")
                .append_pair("state", &state)  // Random state, NOT verifier
                .append_pair("id_token_add_organizations", "true")
                .append_pair("codex_cli_simplified_flow", "true")
                .append_pair("originator", "codex_cli_rs");
        } else if is_gemini {
            // Google OAuth uses standard OAuth 2.0 with PKCE
            url.query_pairs_mut()
                .append_pair("response_type", "code")
                .append_pair("client_id", &self.config.client_id)
                .append_pair("redirect_uri", &self.config.redirect_uri)
                .append_pair("scope", &self.config.scopes.join(" "))
                .append_pair("code_challenge", &pkce.challenge)
                .append_pair("code_challenge_method", "S256")
                .append_pair("state", &pkce.verifier)  // Use verifier as state
                .append_pair("access_type", "offline")  // Request refresh token
                .append_pair("prompt", "consent");  // Force consent screen
        } else {
            // Anthropic specific parameters (uses verifier as state)
            url.query_pairs_mut()
                .append_pair("code", "true")  // Anthropic-specific non-standard parameter
                .append_pair("client_id", &self.config.client_id)
                .append_pair("response_type", "code")
                .append_pair("redirect_uri", &self.config.redirect_uri)
                .append_pair("scope", &self.config.scopes.join(" "))
                .append_pair("code_challenge", &pkce.challenge)
                .append_pair("code_challenge_method", "S256")
                .append_pair("state", &pkce.verifier);
        }

        AuthorizationUrl {
            url: url.to_string(),
            verifier: pkce,
        }
    }

    /// Exchange authorization code for tokens
    pub async fn exchange_code(
        &self,
        code: &str,
        verifier: &str,
        provider_id: &str,
    ) -> Result<OAuthToken> {
        // Parse code (backward compatible: "code#state" or just "code")
        // Note: For OpenAI, we now only receive "code" without state
        let auth_code = if code.contains('#') {
            code.split('#').next().unwrap_or(code)
        } else {
            code
        };

        #[derive(Deserialize)]
        struct TokenResponse {
            access_token: String,
            refresh_token: String,
            expires_in: i64,
        }

        let is_openai_codex = self.config.client_id == "app_EMoamEEZ73f0CkXaXp7hrann";
        let is_gemini = self.config.client_id.starts_with("681255809395-");

        let response = if is_gemini {
            // Google OAuth uses form-urlencoded with client_secret
            tracing::debug!("🔍 Gemini token exchange:");
            tracing::debug!("  code: {}", auth_code);
            tracing::debug!("  code_verifier: {}", verifier);
            tracing::debug!("  redirect_uri: {}", &self.config.redirect_uri);
            tracing::debug!("  client_id: {}", &self.config.client_id);

            let client_secret = self.config.client_secret.as_ref()
                .ok_or_else(|| anyhow!("Gemini OAuth requires client_secret"))?;

            let form_params = [
                ("grant_type", "authorization_code"),
                ("client_id", &self.config.client_id),
                ("client_secret", client_secret),
                ("code", auth_code),
                ("code_verifier", verifier),
                ("redirect_uri", &self.config.redirect_uri),
            ];

            self.http_client
                .post(&self.config.token_url)
                .header("Content-Type", "application/x-www-form-urlencoded")
                .form(&form_params)
                .send()
                .await
                .context("Failed to exchange code for token")?
        } else if is_openai_codex {
            // OpenAI uses form-urlencoded and only needs code + code_verifier
            tracing::debug!("🔍 OpenAI token exchange:");
            tracing::debug!("  code: {}", auth_code);
            tracing::debug!("  code_verifier: {}", verifier);
            tracing::debug!("  redirect_uri: {}", &self.config.redirect_uri);
            tracing::debug!("  client_id: {}", &self.config.client_id);

            let form_params = [
                ("grant_type", "authorization_code"),
                ("client_id", &self.config.client_id),
                ("code", auth_code),
                ("code_verifier", verifier),  // This is the PKCE verifier from frontend
                ("redirect_uri", &self.config.redirect_uri),
            ];

            self.http_client
                .post(&self.config.token_url)
                .header("Content-Type", "application/x-www-form-urlencoded")
                .form(&form_params)
                .send()
                .await
                .context("Failed to exchange code for token")?
        } else {
            // Anthropic uses JSON and requires state (which equals verifier)
            #[derive(Serialize)]
            struct TokenRequest {
                code: String,
                state: String,
                grant_type: String,
                client_id: String,
                redirect_uri: String,
                code_verifier: String,
            }

            let request = TokenRequest {
                code: auth_code.to_string(),
                state: verifier.to_string(),  // Anthropic uses verifier as state
                grant_type: "authorization_code".to_string(),
                client_id: self.config.client_id.clone(),
                redirect_uri: self.config.redirect_uri.clone(),
                code_verifier: verifier.to_string(),
            };

            self.http_client
                .post(&self.config.token_url)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .context("Failed to exchange code for token")?
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("Token exchange failed: {} - {}", status, body));
        }

        let token_response: TokenResponse = response.json().await
            .context("Failed to parse token response")?;

        let expires_at = Utc::now() + chrono::Duration::seconds(token_response.expires_in);

        let token = OAuthToken {
            provider_id: provider_id.to_string(),
            access_token: token_response.access_token,
            refresh_token: token_response.refresh_token,
            expires_at,
            enterprise_url: None,
            project_id: None,  // Will be set by loadCodeAssist for Gemini
        };

        // Save token
        self.token_store.save(token.clone())?;

        Ok(token)
    }

    /// Refresh an access token
    pub async fn refresh_token(&self, provider_id: &str) -> Result<OAuthToken> {
        let existing_token = self.token_store.get(provider_id)
            .context("No token found for provider")?;

        #[derive(Deserialize)]
        struct TokenResponse {
            access_token: String,
            refresh_token: String,
            expires_in: i64,
        }

        let is_openai_codex = self.config.client_id == "app_EMoamEEZ73f0CkXaXp7hrann";

        let response = if is_openai_codex {
            // OpenAI uses form-urlencoded
            let form_params = [
                ("grant_type", "refresh_token"),
                ("refresh_token", &existing_token.refresh_token),
                ("client_id", &self.config.client_id),
            ];

            self.http_client
                .post(&self.config.token_url)
                .header("Content-Type", "application/x-www-form-urlencoded")
                .form(&form_params)
                .send()
                .await
                .context("Failed to refresh token")?
        } else {
            // Anthropic uses JSON
            #[derive(Serialize)]
            struct RefreshRequest {
                grant_type: String,
                refresh_token: String,
                client_id: String,
            }

            let request = RefreshRequest {
                grant_type: "refresh_token".to_string(),
                refresh_token: existing_token.refresh_token.clone(),
                client_id: self.config.client_id.clone(),
            };

            self.http_client
                .post(&self.config.token_url)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .context("Failed to refresh token")?
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("Token refresh failed: {} - {}", status, body));
        }

        let token_response: TokenResponse = response.json().await
            .context("Failed to parse token response")?;

        let expires_at = Utc::now() + chrono::Duration::seconds(token_response.expires_in);

        let token = OAuthToken {
            provider_id: provider_id.to_string(),
            access_token: token_response.access_token,
            refresh_token: token_response.refresh_token,
            expires_at,
            enterprise_url: existing_token.enterprise_url,
            project_id: existing_token.project_id,  // Preserve project_id from existing token
        };

        // Save refreshed token
        self.token_store.save(token.clone())?;

        Ok(token)
    }

    /// Load Code Assist for Gemini and get project ID
    /// This must be called after OAuth exchange for Gemini providers
    pub async fn load_code_assist(&self, access_token: &str) -> Result<String> {
        #[derive(Serialize)]
        struct LoadCodeAssistRequest {
            #[serde(skip_serializing_if = "Option::is_none")]
            cloudaicompanionProject: Option<String>,
            metadata: ClientMetadata,
        }

        #[derive(Serialize)]
        struct ClientMetadata {
            ideType: String,
            platform: String,
            pluginType: String,
        }

        #[derive(Deserialize)]
        struct LoadCodeAssistResponse {
            #[serde(rename = "cloudaicompanionProject")]
            cloudaicompanion_project: Option<String>,
        }

        // Try to get project ID from environment variables (like gemini-cli does)
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT")
            .or_else(|_| std::env::var("GOOGLE_CLOUD_PROJECT_ID"))
            .ok();

        if let Some(ref pid) = project_id {
            tracing::info!("🔍 Using project ID from environment: {}", pid);
        } else {
            tracing::warn!("⚠️ No GOOGLE_CLOUD_PROJECT env var set. loadCodeAssist may not return project ID.");
        }

        let request = LoadCodeAssistRequest {
            cloudaicompanionProject: project_id.clone(),
            metadata: ClientMetadata {
                ideType: "IDE_UNSPECIFIED".to_string(),
                platform: "PLATFORM_UNSPECIFIED".to_string(),
                pluginType: "GEMINI".to_string(),
            },
        };

        tracing::debug!("🔍 Calling loadCodeAssist with project_id={:?}", project_id);

        let response = self.http_client
            .post("https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist")
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to call loadCodeAssist")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            tracing::error!("❌ loadCodeAssist API error {}: {}", status, body);
            return Err(anyhow!("loadCodeAssist failed: {} - {}", status, body));
        }

        // Get response text first for debugging
        let response_text = response.text().await
            .context("Failed to read loadCodeAssist response")?;

        tracing::debug!("📥 loadCodeAssist API response: {}", response_text);

        let load_response: LoadCodeAssistResponse = serde_json::from_str(&response_text)
            .context("Failed to parse loadCodeAssist response")?;

        tracing::debug!("🔍 Parsed loadCodeAssist response: cloudaicompanion_project={:?}", load_response.cloudaicompanion_project);

        // If loadCodeAssist returned a project ID, use it
        // Otherwise, use the one we sent (from environment variables)
        // This matches gemini-cli behavior
        let final_project_id = load_response.cloudaicompanion_project
            .or(project_id);

        final_project_id
            .ok_or_else(|| anyhow!("No project ID available. Set GOOGLE_CLOUD_PROJECT environment variable."))
    }

    /// Get a valid access token (refreshing if needed)
    pub async fn get_valid_token(&self, provider_id: &str) -> Result<String> {
        let token = self.token_store.get(provider_id)
            .context("No token found for provider")?;

        if token.needs_refresh() {
            let refreshed = self.refresh_token(provider_id).await?;
            Ok(refreshed.access_token)
        } else {
            Ok(token.access_token)
        }
    }

    /// Create an API key using OAuth token (for Anthropic Console flow)
    pub async fn create_api_key(&self, provider_id: &str) -> Result<String> {
        let access_token = self.get_valid_token(provider_id).await?;

        #[derive(Deserialize)]
        struct ApiKeyResponse {
            raw_key: String,
        }

        let response = self.http_client
            .post("https://api.anthropic.com/api/oauth/claude_cli/create_api_key")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", access_token))
            .send()
            .await
            .context("Failed to create API key")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("API key creation failed: {} - {}", status, body));
        }

        let api_key_response: ApiKeyResponse = response.json().await
            .context("Failed to parse API key response")?;

        Ok(api_key_response.raw_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pkce_generation() {
        let pkce = PKCEVerifier::generate();

        // Verifier should be base64 URL-safe encoded
        assert!(!pkce.verifier.is_empty());
        assert!(!pkce.challenge.is_empty());

        // Challenge should be different from verifier
        assert_ne!(pkce.verifier, pkce.challenge);
    }

    #[test]
    fn test_authorization_url() {
        let config = OAuthConfig::anthropic();
        let token_store = TokenStore::new(std::env::temp_dir().join("test_tokens.json")).unwrap();
        let client = OAuthClient::new(config, token_store);

        let auth_url = client.get_authorization_url();

        assert!(auth_url.url.contains("client_id="));
        assert!(auth_url.url.contains("code_challenge="));
        assert!(auth_url.url.contains("code_challenge_method=S256"));
        assert!(auth_url.url.contains("scope="));
    }
}

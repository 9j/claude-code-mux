# Gemini Integration Guide

## Overview

Google Gemini provider supports three authentication methods:
1. **OAuth 2.0** - For Google AI Pro/Ultra subscribers (FREE)
2. **API Key** - From Google AI Studio
3. **Vertex AI** - Using Google Cloud Project

## Authentication Methods

### 1. OAuth 2.0 (Google AI Pro/Ultra)

**OAuth Configuration:**
```
client_id: 681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com
client_secret: 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
auth_url: https://accounts.google.com/o/oauth2/v2/auth
token_url: https://oauth2.googleapis.com/token
redirect_uri: http://localhost:{dynamic_port}/oauth2callback
scopes:
  - https://www.googleapis.com/auth/cloud-platform
  - https://www.googleapis.com/auth/userinfo.email
  - https://www.googleapis.com/auth/userinfo.profile
```

**OAuth Flow:**
1. Generate authorization URL with PKCE (optional, but recommended for security)
2. User authorizes in browser → redirected to callback URL
3. Exchange authorization code for access token + refresh token
4. Store tokens for automatic refresh

**API Endpoint (OAuth/API Key):**
```
https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
```

**Headers:**
```
Authorization: Bearer {access_token}  // For OAuth
Content-Type: application/json
```

### 2. API Key (Google AI Studio)

**API Key Configuration:**
- Get API key from: https://aistudio.google.com/app/apikey

**API Endpoint:**
```
https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}
```

**Headers:**
```
Content-Type: application/json
```

### 3. Vertex AI (Google Cloud)

**Vertex AI Configuration:**
```
project_id: YOUR_PROJECT_ID
location: YOUR_LOCATION (e.g., us-central1)
```

**API Endpoint:**
```
https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent
```

**Authentication Options:**
- Application Default Credentials (ADC) via gcloud
- Service Account JSON key
- Google Cloud API key

**Headers:**
```
Authorization: Bearer {access_token}  // From ADC or Service Account
Content-Type: application/json
```

## API Format

### Request Format (Gemini generateContent API)

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Hello, how are you?"
        }
      ]
    }
  ],
  "generationConfig": {
    "temperature": 1.0,
    "topP": 0.95,
    "topK": 40,
    "maxOutputTokens": 8192,
    "stopSequences": []
  },
  "systemInstruction": {
    "parts": [
      {
        "text": "You are a helpful assistant."
      }
    ]
  }
}
```

### Response Format (Non-streaming)

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "I'm doing well, thank you for asking!"
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      "index": 0,
      "safetyRatings": [...]
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 10,
    "candidatesTokenCount": 15,
    "totalTokenCount": 25
  }
}
```

### Streaming Response Format

```
data: {"candidates": [{"content": {"parts": [{"text": "Hello"}],"role": "model"}}]}

data: {"candidates": [{"content": {"parts": [{"text": " there"}],"role": "model"}}]}

data: {"candidates": [{"content": {"parts": [{"text": "!"}],"role": "model"}],"finishReason": "STOP","usageMetadata": {...}}]}
```

## Anthropic → Gemini Conversion

### Role Mapping
```
Anthropic → Gemini
user      → user
assistant → model
```

### Content Block Mapping

**Text:**
```rust
// Anthropic
ContentBlock::Text { text: "Hello" }

// Gemini
{ "text": "Hello" }
```

**Image:**
```rust
// Anthropic
ContentBlock::Image {
  source: ImageSource {
    type: "base64",
    media_type: "image/png",
    data: "iVBORw0KG..."
  }
}

// Gemini
{
  "inline_data": {
    "mime_type": "image/png",
    "data": "iVBORw0KG..."
  }
}
```

**Thinking (Extended Thinking):**
```rust
// Anthropic
ContentBlock::Thinking {
  thinking: "Let me think...",
  signature: "..."
}

// Gemini (convert to text)
{ "text": "Let me think..." }
```

### System Prompt
```rust
// Anthropic
request.system = Some("You are a helpful assistant")

// Gemini
{
  "systemInstruction": {
    "parts": [
      { "text": "You are a helpful assistant" }
    ]
  }
}
```

### Generation Config
```rust
// Anthropic → Gemini
max_tokens       → maxOutputTokens
temperature      → temperature
top_p            → topP
stop_sequences   → stopSequences

// Gemini only
topK: 40 (default)
```

### Tools (Function Calling)

**Anthropic:**
```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather information",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        }
      }
    }
  ]
}
```

**Gemini:**
```json
{
  "tools": [
    {
      "functionDeclarations": [
        {
          "name": "get_weather",
          "description": "Get weather information",
          "parameters": {
            "type": "object",
            "properties": {
              "location": { "type": "string" }
            }
          }
        }
      ]
    }
  ]
}
```

## Model Names

### Gemini Models (via OAuth/API Key)
```
gemini-1.5-pro
gemini-1.5-flash
gemini-1.5-flash-8b
gemini-2.0-flash-exp
gemini-exp-1206
```

### Vertex AI Model Names (prefix with `publishers/google/models/`)
```
publishers/google/models/gemini-1.5-pro
publishers/google/models/gemini-1.5-flash
publishers/google/models/gemini-2.0-flash-exp
```

## Error Handling

### Common Errors

**401 Unauthorized:**
```json
{
  "error": {
    "code": 401,
    "message": "Request is missing required authentication credential.",
    "status": "UNAUTHENTICATED"
  }
}
```

**403 Forbidden:**
```json
{
  "error": {
    "code": 403,
    "message": "User does not have permission to access this resource.",
    "status": "PERMISSION_DENIED"
  }
}
```

**429 Rate Limit:**
```json
{
  "error": {
    "code": 429,
    "message": "Resource has been exhausted (e.g. check quota).",
    "status": "RESOURCE_EXHAUSTED"
  }
}
```

## Implementation Checklist

- [ ] GeminiProvider struct with OAuth/API Key/Vertex AI fields
- [ ] OAuth configuration in auth/oauth.rs
- [ ] OAuth token refresh logic
- [ ] Request transformation (Anthropic → Gemini)
- [ ] Response transformation (Gemini → Anthropic)
- [ ] Streaming support
- [ ] Tool/Function calling support
- [ ] Image support (inline_data)
- [ ] Error handling and mapping
- [ ] Admin UI integration
- [ ] OAuth flow UI (similar to Anthropic/OpenAI)

## References

- Gemini CLI OAuth: `/tmp/gemini-cli/packages/core/src/code_assist/oauth2.ts`
- Gemini API Docs: https://ai.google.dev/gemini-api/docs
- Vertex AI Docs: https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/api-quickstart

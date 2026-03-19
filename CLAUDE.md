# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **OpenAI API Go client library** that provides Go wrappers for OpenAI's REST APIs. It supports:
- ChatGPT (GPT-4o, o1, GPT-3.5, GPT-4)
- DALL·E 2/3 image generation
- Whisper audio transcription
- Embeddings, Fine-tuning, Assistants API, Vector Stores, and more

## Common Commands

```bash
# Run all tests
go test ./...

# Run a specific test
go test -v -run TestFunctionName

# Run linter (golangci-lint must be installed)
golangci-lint run

# Build the project
go build ./...
```

## Code Architecture

The library follows a straightforward pattern:

- **Client** (`client.go`): Central HTTP client struct that handles all API requests
- **API modules**: Each feature area has its own file (e.g., `chat.go`, `embeddings.go`, `files.go`, `assistant.go`)
- **internal/**: Internal utilities for request building, error handling, and JSON marshaling
- **jsonschema/**: JSON Schema generation for function calling and structured outputs

### Core Design Patterns

1. Each API endpoint is a method on `*Client`
2. Request/Response types are defined alongside their corresponding API method files
3. Streaming uses `StreamReader` from `stream_reader.go`
4. Azure OpenAI is supported via `DefaultAzureConfig()` in `config.go`

### Key Files

- `client.go` - Core Client struct and HTTP handling
- `chat.go` - ChatGPT API (most commonly used)
- `config.go` - Client configuration including Azure setup
- `stream_reader.go` - SSE stream parsing
- `error.go` - API error types

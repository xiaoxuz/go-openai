package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
)

// Response API constants
const responsesSuffix = "/responses"

// ResponseStatus represents the status of a response
type ResponseStatus string

const (
	ResponseStatusCompleted  ResponseStatus = "completed"
	ResponseStatusFailed     ResponseStatus = "failed"
	ResponseStatusInProgress ResponseStatus = "in_progress"
	ResponseStatusCancelled  ResponseStatus = "cancelled"
	ResponseStatusQueued     ResponseStatus = "queued"
	ResponseStatusIncomplete ResponseStatus = "incomplete"
)

// TextFormatType represents the type of text format
type TextFormatType string

const (
	TextFormatTypeText       TextFormatType = "text"
	TextFormatTypeJSONObject TextFormatType = "json_object"
	TextFormatTypeJSONSchema TextFormatType = "json_schema"
)

// TextFormatConfig is the configuration for text output
type TextFormatConfig struct {
	Format *TextFormat `json:"format,omitempty"`
}

// TextFormat defines the format for structured output
type TextFormat struct {
	Type        TextFormatType `json:"type,omitempty"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Schema      any            `json:"schema,omitempty"`
	Strict      bool           `json:"strict,omitempty"`
}

// ResponseInputContentType represents the type of input content
type ResponseInputContentType string

const (
	ResponseInputContentTypeText  ResponseInputContentType = "input_text"
	ResponseInputContentTypeImage ResponseInputContentType = "input_image"
	ResponseInputContentTypeFile  ResponseInputContentType = "input_file"
)

// ResponseInputImageDetail represents the detail level for images
type ResponseInputImageDetail string

const (
	ResponseInputImageDetailLow  ResponseInputImageDetail = "low"
	ResponseInputImageDetailHigh ResponseInputImageDetail = "high"
	ResponseInputImageDetailAuto ResponseInputImageDetail = "auto"
)

// ResponseInputContent represents content in the input (text, image, or file)
type ResponseInputContent struct {
	Type     ResponseInputContentType `json:"type"`
	Text     string                   `json:"text,omitempty"`
	ImageURL string                   `json:"image_url,omitempty"`
	Detail   ResponseInputImageDetail `json:"detail,omitempty"`
	FileID   string                   `json:"file_id,omitempty"`
	FileData string                   `json:"file_data,omitempty"`
	FileURL  string                   `json:"file_url,omitempty"`
	Filename string                   `json:"filename,omitempty"`
}

// InputItem represents an input item in the Responses API
type InputItem struct {
	Type    string `json:"type"`
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	Name    string `json:"name,omitempty"`
	// For multi-modal input (text + image)
	Contents []ResponseInputContent `json:"contents,omitempty"`
}

// ResponseRequest represents a request to the Responses API
type ResponseRequest struct {
	Model              string            `json:"model"`
	Input              any               `json:"input,omitempty"`
	Instructions       string            `json:"instructions,omitempty"`
	Store              bool              `json:"store,omitempty"`
	PreviousResponseID string            `json:"previous_response_id,omitempty"`
	Text               *TextFormatConfig `json:"text,omitempty"`
	Tools              []Tool            `json:"tools,omitempty"`
	MaxOutputTokens    int               `json:"max_output_tokens,omitempty"`
	Temperature        float32           `json:"temperature,omitempty"`
	TopP               float32           `json:"top_p,omitempty"`
	Stream             bool              `json:"stream,omitempty"`
	StreamOptions      *StreamOptions    `json:"stream_options,omitempty"`
	// Reasoning effort for o-series and gpt-5 models
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
	// Metadata for the request
	Metadata map[string]string `json:"metadata,omitempty"`
	// User identifier
	User string `json:"user,omitempty"`
	// Service tier
	ServiceTier string `json:"service_tier,omitempty"`
	// Truncation strategy
	Truncation string `json:"truncation,omitempty"`
}

type ResponseUsageInputTokensDetails struct {
	// The number of tokens that were retrieved from the cache.
	// [More on prompt caching](https://platform.openai.com/docs/guides/prompt-caching).
	CachedTokens int64 `json:"cached_tokens" api:"required"`
}

// A detailed breakdown of the output tokens.
type ResponseUsageOutputTokensDetails struct {
	// The number of reasoning tokens.
	ReasoningTokens int64 `json:"reasoning_tokens" api:"required"`
}

type ResponseUsage struct {
	// The number of input tokens.
	InputTokens int64 `json:"input_tokens" api:"required"`
	// A detailed breakdown of the input tokens.
	InputTokensDetails ResponseUsageInputTokensDetails `json:"input_tokens_details" api:"required"`
	// The number of output tokens.
	OutputTokens int64 `json:"output_tokens" api:"required"`
	// A detailed breakdown of the output tokens.
	OutputTokensDetails ResponseUsageOutputTokensDetails `json:"output_tokens_details" api:"required"`
	// The total number of tokens used.
	TotalTokens int64 `json:"total_tokens" api:"required"`
}

// APIResponse represents a response from the Responses API
type APIResponse struct {
	ID           string         `json:"id"`
	Object       string         `json:"object"`
	CreatedAt    int64          `json:"created_at"`
	Model        string         `json:"model"`
	Output       []OutputItem   `json:"output"`
	Status       ResponseStatus `json:"status"`
	Usage        ResponseUsage  `json:"usage"`
	Instructions string         `json:"instructions,omitempty"`
	CompletedAt  *int64         `json:"completed_at,omitempty"`
	Background   bool           `json:"background,omitempty"`
}

// SetHeader implements the Response interface
func (r *APIResponse) SetHeader(header http.Header) {
	// No-op for now, can be extended to handle response headers
	_ = header
}

// OutputItem represents an item in the response output
type OutputItem struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	Role   string `json:"role,omitempty"`
	Status string `json:"status,omitempty"`
	Name   string `json:"name,omitempty"`
	CallID string `json:"call_id,omitempty"`
	Index  int    `json:"index,omitempty"`
	// Content for message type output (can include text and refusal)
	Content []OutputContentItem `json:"content,omitempty"`
	// For function calls
	Arguments string `json:"arguments,omitempty"`
	// For reasoning - summary is an array
	Summary          []string `json:"summary,omitempty"`
	EncryptedContent string   `json:"encrypted_content,omitempty"`
	// For image generation
	Result string `json:"result,omitempty"`
	// For code interpreter
	Code        string `json:"code,omitempty"`
	ContainerID string `json:"container_id,omitempty"`
	Outputs     any    `json:"outputs,omitempty"`
}

// OutputContentItem represents content within an output item
type OutputContentItem struct {
	Type    string `json:"type"` // "output_text", "refusal"
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`
}

// OutputText returns the concatenated text output from all output items
func (r *APIResponse) OutputText() string {
	var sb strings.Builder
	for _, item := range r.Output {
		// Check Content array first
		for _, content := range item.Content {
			if content.Type == "output_text" {
				sb.WriteString(content.Text)
			}
		}
	}
	return sb.String()
}

// CreateResponse creates a response using the Responses API
func (c *Client) CreateResponse(ctx context.Context, request ResponseRequest) (response APIResponse, err error) {
	if request.Stream {
		err = ErrChatCompletionStreamNotSupported
		return
	}

	req, err := c.newRequest(
		ctx,
		http.MethodPost,
		c.fullURL(responsesSuffix),
		withBody(request),
	)
	if err != nil {
		return
	}

	err = c.sendRequest(req, &response)
	return
}

// CreateResponseStream creates a streaming response using the Responses API
func (c *Client) CreateResponseStream(
	ctx context.Context,
	request ResponseRequest,
) (stream *ResponseStream, err error) {
	if !request.Stream {
		request.Stream = true
	}

	req, err := c.newRequest(
		ctx,
		http.MethodPost,
		c.fullURL(responsesSuffix),
		withBody(request),
	)
	if err != nil {
		return nil, err
	}

	// Set streaming headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Connection", "keep-alive")

	if len(c.Header) > 0 {
		for k, v := range c.Header {
			req.Header.Set(k, v)
		}
	}

	resp, err := c.config.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}

	if isFailureStatusCode(resp) {
		return nil, c.handleErrorResp(resp)
	}

	stream = &ResponseStream{
		reader:   bufio.NewReader(resp.Body),
		response: resp,
	}
	return stream, nil
}

// ResponseStream represents a streaming response
type ResponseStream struct {
	reader   *bufio.Reader
	response *http.Response
}

// Recv receives the next streaming event
func (s *ResponseStream) Recv() (ResponseStreamEvent, error) {
	line, err := s.reader.ReadBytes('\n')
	if err != nil {
		if errors.Is(err, io.EOF) {
			return ResponseStreamEvent{}, err
		}
		return ResponseStreamEvent{}, err
	}

	line = bytes.TrimSpace(line)
	if len(line) == 0 {
		return s.Recv()
	}

	// Skip non-data lines
	if !bytes.HasPrefix(line, []byte("data: ")) {
		return s.Recv()
	}

	data := bytes.TrimPrefix(line, []byte("data: "))
	if string(data) == "[DONE]" {
		return ResponseStreamEvent{}, io.EOF
	}

	var event ResponseStreamEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return ResponseStreamEvent{}, err
	}

	return event, nil
}

// Close closes the stream
func (s *ResponseStream) Close() error {
	if s.response != nil {
		return s.response.Body.Close()
	}
	return nil
}

// ResponseStreamEvent represents an event in a streaming response
type ResponseStreamEvent struct {
	// Common fields
	Type           string `json:"type"`
	EventID        string `json:"event_id,omitempty"`
	ItemID         string `json:"item_id,omitempty"`
	OutputIndex    int    `json:"output_index,omitempty"`
	ContentIndex   int    `json:"content_index,omitempty"`
	SequenceNumber int64  `json:"sequence_number,omitempty"`

	// For output items (response.output_item.added)
	Output []OutputItem `json:"output,omitempty"`
	Item   *OutputItem  `json:"item,omitempty"`

	// For text deltas (response.output_text.delta, response.refusal.delta)
	Delta   string `json:"delta,omitempty"`
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`

	// For function calls
	Arguments string `json:"arguments,omitempty"`
	Name      string `json:"name,omitempty"`

	// For completed event (response.completed)
	Response *APIResponse   `json:"response,omitempty"`
	Usage    *ResponseUsage `json:"usage,omitempty"`

	// For error event
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Param   string `json:"param,omitempty"`

	// For image generation
	PartialImageB64   string `json:"partial_image_b64,omitempty"`
	PartialImageIndex int    `json:"partial_image_index,omitempty"`

	// For annotations
	Annotation      any `json:"annotation,omitempty"`
	AnnotationIndex int `json:"annotation_index,omitempty"`

	// Status
	Status string `json:"status,omitempty"`
}

// Ensure ResponseStream satisfies io.Closer
var _ io.Closer = (*ResponseStream)(nil)

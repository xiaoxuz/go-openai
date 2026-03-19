package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/xiaoxuz/go-openai"
)

// TestCreateResponse_Text_NonStream 测试非流式文本输入
func TestCreateResponse_Text_NonStream(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/responses", handleResponseEndpoint)

	req := openai.ResponseRequest{
		Model: "gpt-4o",
		Input: "Hello, how are you?",
		Store: false,
	}

	resp, err := client.CreateResponse(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateResponse failed: %v", err)
	}

	if resp.Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", resp.Model)
	}

	if resp.Status != openai.ResponseStatusCompleted {
		t.Errorf("expected status completed, got %s", resp.Status)
	}

	if len(resp.Output) == 0 {
		t.Error("expected output to have at least one item")
	}

	// 验证 OutputText 方法
	text := resp.OutputText()
	if text == "" {
		t.Error("expected non-empty output text")
	}
}

// TestCreateResponse_InputItems 测试 InputItems 数组输入
func TestCreateResponse_InputItems(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/responses", handleResponseEndpoint)

	req := openai.ResponseRequest{
		Model: "gpt-4o",
		Input: []openai.InputItem{
			{
				Type: "message",
				Role: "user",
				Content: []openai.ResponseInputContent{
					openai.ResponseInputContent{
						Type: openai.ResponseInputContentTypeText,
						Text: "Hellow",
					},
				},
			},
		},
		Store: false,
	}

	resp, err := client.CreateResponse(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateResponse failed: %v", err)
	}

	if resp.Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", resp.Model)
	}
}

// TestCreateResponse_ImageInput 测试图片输入（多模态）
func TestCreateResponse_ImageInput(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/responses", handleResponseEndpointWithImage)

	req := openai.ResponseRequest{
		Model: "gpt-4o",
		Input: []openai.InputItem{
			{
				Type: "message",
				Role: "user",
				Content: []openai.ResponseInputContent{
					{
						Type: openai.ResponseInputContentTypeText,
						Text: "描述这张图片",
					},
					{
						Type:     openai.ResponseInputContentTypeImage,
						ImageURL: "https://example.com/image.jpg",
						Detail:   openai.ResponseInputImageDetailHigh,
					},
				},
			},
		},
		Store: false,
	}

	resp, err := client.CreateResponse(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateResponse failed: %v", err)
	}

	if resp.Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", resp.Model)
	}

	if len(resp.Output) == 0 {
		t.Error("expected output to have at least one item")
	}
}

// TestCreateResponse_TextFormat 测试结构化输出 (JSON Schema)
func TestCreateResponse_TextFormat(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/responses", handleResponseEndpoint)

	req := openai.ResponseRequest{
		Model: "gpt-4o",
		Input: "Jane, 54 years old",
		Store: false,
		Text: &openai.TextFormatConfig{
			Format: &openai.TextFormat{
				Type:   openai.TextFormatTypeJSONSchema,
				Name:   "person",
				Strict: true,
			},
		},
	}

	resp, err := client.CreateResponse(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateResponse failed: %v", err)
	}

	if resp.Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", resp.Model)
	}
}

// TestCreateResponse_WithInstructions 测试系统指令
func TestCreateResponse_WithInstructions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/responses", handleResponseEndpointWithInstructions)

	req := openai.ResponseRequest{
		Model:        "gpt-4o",
		Input:        "What is 2+2?",
		Instructions: "You are a math tutor. Always explain step by step.",
		Store:        false,
	}

	resp, err := client.CreateResponse(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateResponse failed: %v", err)
	}

	if resp.Instructions == "" {
		t.Error("expected instructions in response")
	}
}

// TestCreateResponseStream_Text 测试流式文本输入
func TestCreateResponseStream_Text(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/responses", handleResponseStreamEndpoint)

	req := openai.ResponseRequest{
		Model:  "gpt-4o",
		Input:  "Tell me a story",
		Stream: true,
	}

	stream, err := client.CreateResponseStream(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateResponseStream failed: %v", err)
	}
	defer stream.Close()

	var fullText strings.Builder
	var eventCount int

	for {
		event, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" || err == io.EOF {
				break
			}
			t.Fatalf("stream.Recv failed: %v", err)
		}
		eventCount++

		switch event.Type {
		case "response.output_text.delta":
			fullText.WriteString(event.Delta)
		case "response.completed":
			if event.Response != nil {
				if event.Response.Usage.InputTokens == 0 {
					t.Error("expected usage to be populated")
				}
			}
		}
	}

	if fullText.Len() == 0 {
		t.Error("expected some text from stream")
	}

	if eventCount == 0 {
		t.Error("expected at least one event")
	}
}

// TestCreateResponseStream_Realtime 模拟实时打字机效果
func TestCreateResponseStream_Realtime(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/responses", handleResponseStreamEndpoint)

	req := openai.ResponseRequest{
		Model:  "gpt-4o",
		Input:  "Hello",
		Stream: true,
	}

	stream, err := client.CreateResponseStream(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateResponseStream failed: %v", err)
	}
	defer stream.Close()

	var fullText strings.Builder

	for {
		event, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("stream.Recv failed: %v", err)
		}

		if event.Type == "response.output_text.delta" {
			fullText.WriteString(event.Delta)
			// 模拟实时处理：每次收到 delta 就可以处理
			_ = fullText.String()
		}
	}

	// 验证最终文本
	if fullText.Len() == 0 {
		t.Error("expected text in stream response")
	}
}

// TestCreateResponseStream_WaitComplete 等待完成事件获取完整响应
func TestCreateResponseStream_WaitComplete(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()

	server.RegisterHandler("/v1/responses", handleResponseStreamEndpoint)

	req := openai.ResponseRequest{
		Model:  "gpt-4o",
		Input:  "What's the weather?",
		Stream: true,
	}

	stream, err := client.CreateResponseStream(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateResponseStream failed: %v", err)
	}
	defer stream.Close()

	var finalText string

	for {
		event, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("stream.Recv failed: %v", err)
		}

		// 从 completed 事件获取完整响应
		if event.Type == "response.completed" && event.Response != nil {
			finalText = event.Response.OutputText()
			break
		}
	}

	if finalText == "" {
		t.Error("expected final text from completed event")
	}
}

// handleResponseEndpoint 是测试用的 Responses API handler
func handleResponseEndpoint(w http.ResponseWriter, r *http.Request) {
	var req openai.ResponseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// 根据请求返回响应
	resp := openai.APIResponse{
		ID:        "resp_123",
		Object:    "response",
		CreatedAt: 1234567890,
		Model:     req.Model,
		Output: []openai.OutputItem{
			{
				Type: "output_text",
				ID:   "msg_1",
				Content: []openai.OutputContentItem{
					{
						Type: "output_text",
						Text: "Hello! I'm doing well, thank you!",
					},
				},
			},
		},
		Status: openai.ResponseStatusCompleted,
		Usage: openai.ResponseUsage{
			InputTokens:  10,
			OutputTokens: 20,
			TotalTokens:  30,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleResponseEndpointWithImage 是测试图片输入的 handler
func handleResponseEndpointWithImage(w http.ResponseWriter, r *http.Request) {
	var req openai.ResponseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	resp := openai.APIResponse{
		ID:        "resp_456",
		Object:    "response",
		CreatedAt: 1234567890,
		Model:     req.Model,
		Output: []openai.OutputItem{
			{
				Type: "output_text",
				ID:   "msg_2",
				Content: []openai.OutputContentItem{
					{
						Type: "output_text",
						Text: "这是一张风景图片，显示了山脉和湖泊。",
					},
				},
			},
		},
		Status: openai.ResponseStatusCompleted,
		Usage: openai.ResponseUsage{
			InputTokens:  100,
			OutputTokens: 50,
			TotalTokens:  150,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleResponseEndpointWithInstructions 是测试指令的 handler
func handleResponseEndpointWithInstructions(w http.ResponseWriter, r *http.Request) {
	var req openai.ResponseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	resp := openai.APIResponse{
		ID:           "resp_789",
		Object:       "response",
		CreatedAt:    1234567890,
		Model:        req.Model,
		Instructions: req.Instructions,
		Output: []openai.OutputItem{
			{
				Type: "output_text",
				ID:   "msg_3",
				Content: []openai.OutputContentItem{
					{
						Type: "output_text",
						Text: "2 + 2 = 4\n\n步骤：\n1. 我们有两个数字：2 和 2\n2. 将它们相加：2 + 2 = 4",
					},
				},
			},
		},
		Status: openai.ResponseStatusCompleted,
		Usage: openai.ResponseUsage{
			InputTokens:  20,
			OutputTokens: 40,
			TotalTokens:  60,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleResponseStreamEndpoint 是测试流式响应的 handler
func handleResponseStreamEndpoint(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		return
	}

	// 发送多个流式事件
	events := []openai.ResponseStreamEvent{
		{
			Type: "response.output_item.added",
			Output: []openai.OutputItem{
				{
					Type: "output_text",
					ID:   "msg_1",
				},
			},
		},
		{
			Type:         "response.output_text.delta",
			ItemID:       "msg_1",
			OutputIndex:  0,
			ContentIndex: 0,
			Delta:        "Hello",
		},
		{
			Type:         "response.output_text.delta",
			ItemID:       "msg_1",
			OutputIndex:  0,
			ContentIndex: 0,
			Delta:        " World",
		},
		{
			Type:         "response.output_text.delta",
			ItemID:       "msg_1",
			OutputIndex:  0,
			ContentIndex: 0,
			Delta:        "!",
		},
		{
			Type: "response.completed",
			Response: &openai.APIResponse{
				ID:        "resp_stream_123",
				Object:    "response",
				CreatedAt: 1234567890,
				Model:     "gpt-4o",
				Output: []openai.OutputItem{
					{
						Type: "output_text",
						ID:   "msg_1",
						Content: []openai.OutputContentItem{
							{
								Type: "output_text",
								Text: "Hello World!",
							},
						},
					},
				},
				Status: openai.ResponseStatusCompleted,
				Usage: openai.ResponseUsage{
					InputTokens:  10,
					OutputTokens: 5,
					TotalTokens:  15,
				},
			},
			Usage: &openai.ResponseUsage{
				InputTokens:  10,
				OutputTokens: 5,
				TotalTokens:  15,
			},
		},
	}

	for _, event := range events {
		data, _ := json.Marshal(event)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

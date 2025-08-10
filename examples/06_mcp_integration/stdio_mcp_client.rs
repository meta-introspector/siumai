//! Stdio MCP Client Example with Siumai LLM Integration
//!
//! This example demonstrates how to create a stdio MCP client that connects to
//! an MCP server via child process, retrieves available tools, and integrates
//! them with siumai for LLM tool calling functionality.
//!
//! Features:
//! - Stdio MCP client implementation via child process
//! - Tool discovery and conversion
//! - LLM integration with tool calling
//! - Complete request-response flow
//! - Process-based communication with MCP server
//!
//! To run this example:
//! 1. Make sure the stdio MCP server is available: `cargo build --example stdio_mcp_server`
//! 2. Then run this client: `cargo run --example stdio_mcp_client`

use rmcp::{
    RoleClient, model::CallToolRequestParam, service::ServiceExt, transport::TokioChildProcess,
};
use serde_json::json;
use siumai::prelude::*;
use tokio::process::Command as TokioCommand;

/// Stdio MCP Client that communicates with MCP server via child process
pub struct StdioMcpClient {
    service: rmcp::service::RunningService<RoleClient, ()>,
}

impl StdioMcpClient {
    /// Create a new stdio MCP client by spawning the server as a child process
    pub async fn new() -> Result<Self, LlmError> {
        // Get the path to the compiled server binary
        let server_binary = if cfg!(windows) {
            "target/debug/examples/stdio_mcp_server.exe"
        } else {
            "target/debug/examples/stdio_mcp_server"
        };

        // Check if server binary exists
        if !std::path::Path::new(server_binary).exists() {
            return Err(LlmError::provider_error(
                "MCP",
                format!(
                    "Server binary not found at {server_binary}. Please run: cargo build --example stdio_mcp_server"
                ),
            ));
        }

        println!("📡 Starting stdio MCP server: {server_binary}");

        // Create child process transport
        let transport = TokioChildProcess::new(TokioCommand::new(server_binary)).map_err(|e| {
            LlmError::provider_error("MCP", format!("Failed to create child process: {e}"))
        })?;

        // Create service
        let service = ().serve(transport).await.map_err(|e| {
            LlmError::provider_error("MCP", format!("Failed to create MCP service: {e}"))
        })?;

        println!("✅ MCP client connected to stdio server");

        Ok(Self { service })
    }

    /// Get server information
    pub fn get_server_info(&self) -> String {
        format!("{:#?}", self.service.peer_info())
    }

    /// Get available tools from MCP server
    pub async fn get_tools(&self) -> Result<Vec<Tool>, LlmError> {
        let tools_result = self
            .service
            .list_tools(Default::default())
            .await
            .map_err(|e| LlmError::provider_error("MCP", format!("Failed to list tools: {e}")))?;

        let mut llm_tools = Vec::new();

        for tool in tools_result.tools {
            // Convert MCP tool to siumai Tool
            let llm_tool = Tool::function(
                tool.name.to_string(),
                tool.description
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "No description".to_string()),
                serde_json::Value::Object((*tool.input_schema).clone()),
            );

            llm_tools.push(llm_tool);
        }

        println!("📋 Retrieved {} tools from MCP server", llm_tools.len());
        Ok(llm_tools)
    }

    /// Execute a tool call via MCP server
    pub async fn call_tool(&self, tool_call: &ToolCall) -> Result<String, LlmError> {
        let function = tool_call
            .function
            .as_ref()
            .ok_or_else(|| LlmError::provider_error("MCP", "Tool call missing function"))?;

        let arguments: serde_json::Value = serde_json::from_str(&function.arguments)
            .map_err(|e| LlmError::provider_error("MCP", format!("Invalid tool arguments: {e}")))?;

        let request = CallToolRequestParam {
            name: function.name.clone().into(),
            arguments: arguments.as_object().cloned(),
        };

        println!(
            "🔧 Calling tool: {} with args: {}",
            function.name, arguments
        );

        let result = self
            .service
            .call_tool(request)
            .await
            .map_err(|e| LlmError::provider_error("MCP", format!("Tool call failed: {e}")))?;

        // Extract text content from MCP response
        let content = result
            .content
            .as_ref()
            .and_then(|content_vec| content_vec.first())
            .and_then(|content| {
                if let rmcp::model::RawContent::Text(text_content) = &content.raw {
                    Some(text_content.text.clone())
                } else {
                    None
                }
            })
            .ok_or_else(|| LlmError::provider_error("MCP", "Invalid tool call response format"))?;

        println!("✅ Tool call result: {content}");
        Ok(content)
    }

    /// Shutdown the MCP client
    pub async fn shutdown(self) -> Result<(), LlmError> {
        self.service.cancel().await.map_err(|e| {
            LlmError::provider_error("MCP", format!("Failed to shutdown MCP service: {e}"))
        })?;
        println!("🔌 MCP client disconnected");
        Ok(())
    }
}

/// Demonstration of Stdio MCP + LLM integration
pub struct StdioMcpLlmDemo {
    mcp_client: StdioMcpClient,
    llm_client: Option<Box<dyn ChatCapability>>,
}

impl StdioMcpLlmDemo {
    pub async fn new() -> Result<Self, LlmError> {
        // Create stdio MCP client
        let mcp_client = StdioMcpClient::new().await?;

        // Try to create LLM client (optional for demo)
        let llm_client = if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if api_key != "demo-key" && !api_key.is_empty() {
                Some(Box::new(
                    LlmBuilder::new()
                        .openai()
                        .api_key(api_key)
                        .model("gpt-3.5-turbo")
                        .build()
                        .await?,
                ) as Box<dyn ChatCapability>)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            mcp_client,
            llm_client,
        })
    }

    /// Run a complete demo showing stdio MCP tool integration with LLM
    pub async fn run_demo(&mut self) -> Result<(), LlmError> {
        println!("🚀 Stdio MCP + LLM Integration Demo");
        println!("===================================");
        println!();

        // Step 1: Show server info
        println!("📡 Step 1: MCP Server Information");
        println!("{}", self.mcp_client.get_server_info());
        println!();

        // Step 2: Get available tools from stdio MCP server
        println!("📋 Step 2: Discovering tools from stdio MCP server...");
        let tools = self.mcp_client.get_tools().await?;
        println!("✅ Found {} tools:", tools.len());
        for tool in &tools {
            println!(
                "   • {} - {}",
                tool.function.name, tool.function.description
            );
        }
        println!();

        // Step 3: Test tool calls directly (simulating LLM behavior)
        println!("🔧 Step 3: Testing tool calls via stdio MCP...");
        let test_tool_calls = vec![
            ToolCall {
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: Some(FunctionCall {
                    name: "add".to_string(),
                    arguments: json!({"a": 25, "b": 17}).to_string(),
                }),
            },
            ToolCall {
                id: "call_2".to_string(),
                r#type: "function".to_string(),
                function: Some(FunctionCall {
                    name: "get_time".to_string(),
                    arguments: json!({"timezone": "UTC"}).to_string(),
                }),
            },
        ];

        let mut tool_results = Vec::new();
        for tool_call in &test_tool_calls {
            let result = self.mcp_client.call_tool(tool_call).await?;
            tool_results.push(result);
        }
        println!();

        // Step 4: Show integration with LLM (if available)
        if self.llm_client.is_some() {
            println!("🤖 Step 4: LLM Integration Examples...");

            // Run non-streaming example
            self.run_non_streaming_example(&tools).await?;

            // Run streaming example
            self.run_streaming_example(&tools).await?;
        } else {
            println!(
                "💡 Step 4: No LLM client available (set OPENAI_API_KEY for full integration)"
            );
        }
        println!();

        // Step 5: Summary
        println!("📊 Demo Summary:");
        println!("   ✅ Stdio MCP server communication: Working");
        println!("   ✅ Tool discovery: {} tools found", tools.len());
        println!(
            "   ✅ Tool execution: {} calls successful",
            tool_results.len()
        );
        if self.llm_client.is_some() {
            println!("   ✅ Non-streaming LLM integration: Complete workflow");
            println!("   ✅ Streaming LLM integration: Complete workflow");
        } else {
            println!("   ⚠️ LLM integration: Not configured");
        }
        println!();

        println!("✨ Stdio MCP integration demo completed!");
        println!("🔄 This demonstrates:");
        println!("   1. Process-based communication with MCP server");
        println!("   2. Real-time tool discovery via JSON-RPC over stdio");
        println!("   3. Tool execution with parameter passing");
        println!("   4. Non-streaming: LLM → Tool Calls → MCP Execution → Final LLM Response");
        println!("   5. Streaming: LLM Stream → Tool Calls → MCP Execution → Final LLM Response");
        println!("   6. Complete integration of siumai streaming with MCP tool execution");

        Ok(())
    }

    /// Demonstrate non-streaming LLM integration with MCP tools
    async fn run_non_streaming_example(&mut self, tools: &[Tool]) -> Result<(), LlmError> {
        let llm_client = self
            .llm_client
            .as_ref()
            .ok_or_else(|| LlmError::provider_error("Demo", "LLM client not available"))?;
        println!("📋 Non-Streaming LLM + MCP Integration");
        println!("=====================================");

        // Create messages for the LLM
        let user_message = "Please add 25 and 17, then tell me the current time in UTC.";
        println!("👤 User: {user_message}");

        let messages = vec![ChatMessage::user(user_message).build()];

        // Test complete chat_with_tools flow with MCP execution
        println!("\n🔧 Testing chat_with_tools with MCP execution:");
        match llm_client
            .chat_with_tools(messages.clone(), Some(tools.to_vec()))
            .await
        {
            Ok(response) => {
                if let Some(tool_calls) = response.get_tool_calls() {
                    println!("📞 LLM requested {} tool calls:", tool_calls.len());

                    // Execute each tool call via MCP
                    let mut tool_results = Vec::new();
                    for tool_call in tool_calls {
                        if let Some(function) = &tool_call.function {
                            println!(
                                "   🔧 Calling: {} with args: {}",
                                function.name, function.arguments
                            );

                            // Execute the tool via MCP
                            match self.mcp_client.call_tool(tool_call).await {
                                Ok(result) => {
                                    println!("   ✅ Result: {result}");
                                    tool_results.push((tool_call.clone(), result));
                                }
                                Err(e) => {
                                    println!("   ❌ Error: {e}");
                                    tool_results.push((tool_call.clone(), format!("Error: {e}")));
                                }
                            }
                        }
                    }

                    // Create follow-up messages with tool results
                    let mut follow_up_messages = messages.clone();

                    // Add assistant message with tool calls
                    let tool_calls = response.get_tool_calls().unwrap_or(&vec![]).clone();
                    follow_up_messages.push(
                        ChatMessage::assistant("")
                            .with_tool_calls(tool_calls)
                            .build(),
                    );

                    // Add tool result messages
                    for (tool_call, result) in tool_results {
                        follow_up_messages.push(ChatMessage::tool(&result, &tool_call.id).build());
                    }

                    // Get final response from LLM
                    println!("\n🤖 Getting final response from LLM...");
                    match llm_client.chat(follow_up_messages).await {
                        Ok(final_response) => {
                            if let MessageContent::Text(content) = &final_response.content {
                                println!("🎯 Final LLM response: {content}");
                            }
                        }
                        Err(e) => {
                            println!("⚠️ Error getting final response: {e}");
                        }
                    }
                } else {
                    println!("📝 LLM response (no tool calls): {:?}", response.content);
                }
            }
            Err(e) => {
                println!("⚠️ Error with chat_with_tools: {e}");
            }
        }

        println!("✅ Non-streaming example completed!\n");
        Ok(())
    }

    /// Demonstrate streaming LLM integration with MCP tools
    async fn run_streaming_example(&mut self, tools: &[Tool]) -> Result<(), LlmError> {
        use futures::StreamExt;
        use siumai::stream::{ChatStreamEvent, StreamProcessor};

        let llm_client = self
            .llm_client
            .as_ref()
            .ok_or_else(|| LlmError::provider_error("Demo", "LLM client not available"))?;

        println!("🌊 Streaming LLM + MCP Integration");
        println!("==================================");

        // Create messages for the LLM
        let user_message = "Please add 25 and 17, then tell me the current time in UTC.";
        println!("👤 User: {user_message}");

        let messages = vec![ChatMessage::user(user_message).build()];

        // Test streaming version with complete MCP execution
        println!("\n🌊 Testing chat_stream with MCP execution:");
        match llm_client
            .chat_stream(messages.clone(), Some(tools.to_vec()))
            .await
        {
            Ok(mut stream) => {
                let mut processor = StreamProcessor::new();
                let mut content_buffer = String::new();

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(ChatStreamEvent::ToolCallDelta {
                            id,
                            function_name,
                            arguments_delta,
                            index,
                        }) => {
                            println!(
                                "🔧 Tool call delta - ID: {id}, Name: {function_name:?}, Args: {arguments_delta:?}, Index: {index:?}"
                            );
                            processor.process_event(ChatStreamEvent::ToolCallDelta {
                                id,
                                function_name,
                                arguments_delta,
                                index,
                            });
                        }
                        Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                            print!("{delta}");
                            content_buffer.push_str(&delta);
                            processor.process_event(ChatStreamEvent::ContentDelta {
                                delta,
                                index: None,
                            });
                        }
                        Ok(ChatStreamEvent::StreamEnd { .. }) => {
                            println!("\n🏁 Stream ended");
                            let final_response = processor.build_final_response();

                            // Execute tool calls if any
                            if let Some(tool_calls) = final_response.get_tool_calls() {
                                println!(
                                    "📋 Executing {} tool calls from stream:",
                                    tool_calls.len()
                                );

                                let mut stream_tool_results = Vec::new();
                                for tool_call in tool_calls {
                                    if let Some(function) = &tool_call.function {
                                        println!(
                                            "   📞 Executing: {} with args: {}",
                                            function.name, function.arguments
                                        );

                                        // Execute the tool via MCP
                                        match self.mcp_client.call_tool(tool_call).await {
                                            Ok(result) => {
                                                println!("   ✅ Result: {result}");
                                                stream_tool_results
                                                    .push((tool_call.clone(), result));
                                            }
                                            Err(e) => {
                                                println!("   ❌ Error: {e}");
                                                stream_tool_results.push((
                                                    tool_call.clone(),
                                                    format!("Error: {e}"),
                                                ));
                                            }
                                        }
                                    }
                                }

                                // Create follow-up messages with tool results for streaming
                                let mut stream_follow_up_messages = messages.clone();

                                // Add assistant message with tool calls
                                let stream_tool_calls =
                                    final_response.get_tool_calls().unwrap_or(&vec![]).clone();
                                stream_follow_up_messages.push(
                                    ChatMessage::assistant("")
                                        .with_tool_calls(stream_tool_calls)
                                        .build(),
                                );

                                // Add tool result messages
                                for (tool_call, result) in stream_tool_results {
                                    stream_follow_up_messages
                                        .push(ChatMessage::tool(&result, &tool_call.id).build());
                                }

                                // Get final response from LLM for streaming
                                println!("\n🤖 Getting final response from LLM (streaming)...");
                                match llm_client.chat(stream_follow_up_messages).await {
                                    Ok(final_response) => {
                                        if let MessageContent::Text(content) =
                                            &final_response.content
                                        {
                                            println!(
                                                "🎯 Final LLM response (streaming): {content}"
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        println!(
                                            "⚠️ Error getting final response (streaming): {e}"
                                        );
                                    }
                                }
                            } else {
                                println!("📝 No tool calls in streaming response");
                                if !content_buffer.is_empty() {
                                    println!("💬 Direct response: {content_buffer}");
                                }
                            }
                            break;
                        }
                        Ok(other) => {
                            println!("📡 Other event: {other:?}");
                        }
                        Err(e) => {
                            println!("❌ Stream error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                println!("⚠️ Error with chat_stream: {e}");
            }
        }

        println!("✅ Streaming example completed!\n");
        Ok(())
    }

    /// Shutdown the demo
    pub async fn shutdown(self) -> Result<(), LlmError> {
        self.mcp_client.shutdown().await
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    siumai::tracing::init_default_tracing().ok();

    println!("🌐 Stdio MCP Client - LLM Integration Demo");
    println!("==========================================");
    println!();

    println!("💡 This demo will:");
    println!("   1. Start the stdio MCP server as a child process");
    println!("   2. Connect to it via stdio transport");
    println!("   3. Discover and test available tools");
    println!("   4. Demonstrate LLM integration (if OPENAI_API_KEY is set)");
    println!();

    // Create and run the demo
    match StdioMcpLlmDemo::new().await {
        Ok(mut demo) => {
            if let Err(e) = demo.run_demo().await {
                eprintln!("❌ Demo failed: {e}");
                let _ = demo.shutdown().await;
                std::process::exit(1);
            }

            // Shutdown gracefully
            if let Err(e) = demo.shutdown().await {
                eprintln!("⚠️ Warning: Failed to shutdown cleanly: {e}");
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize stdio MCP client: {e}");
            eprintln!("💡 Make sure to build the server first:");
            eprintln!("   cargo build --example stdio_mcp_server");
            std::process::exit(1);
        }
    }

    Ok(())
}

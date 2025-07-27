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
            .first()
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
        if let Some(ref llm_client) = self.llm_client {
            println!("🤖 Step 4: Integrating with real LLM...");

            let user_message = "Please add 25 and 17, then tell me the current time in UTC.";
            println!("👤 User: {user_message}");

            let mut messages = vec![ChatMessage::user(user_message).build()];

            // Send request to LLM with tools
            match llm_client
                .chat_with_tools(messages.clone(), Some(tools.clone()))
                .await
            {
                Ok(response) => {
                    if let Some(tool_calls) = response.get_tool_calls() {
                        println!("🔧 LLM requested {} tool calls:", tool_calls.len());

                        let mut tool_results = Vec::new();

                        for tool_call in tool_calls {
                            if let Some(function) = &tool_call.function {
                                println!("   📞 Calling: {}", function.name);
                                let result = self.mcp_client.call_tool(tool_call).await?;
                                println!("   ✅ Result: {result}");

                                tool_results.push(
                                    ChatMessage::tool(result.clone(), tool_call.id.clone()).build(),
                                );
                            }
                        }

                        // Add assistant message with tool calls
                        messages.push(
                            ChatMessage::assistant("")
                                .with_tool_calls(tool_calls.clone())
                                .build(),
                        );

                        // Add tool results
                        messages.extend(tool_results);

                        // Get final response from LLM
                        println!("🔄 Getting final response from LLM...");
                        match llm_client
                            .chat_with_tools(messages, Some(tools.clone()))
                            .await
                        {
                            Ok(final_response) => {
                                println!(
                                    "🤖 LLM Final Response: {}",
                                    final_response
                                        .text()
                                        .unwrap_or_else(|| "No response".to_string())
                                );
                            }
                            Err(e) => {
                                println!("⚠️ Error getting final LLM response: {e}");
                            }
                        }

                        println!("✅ Complete LLM tool calling workflow finished!");
                    } else {
                        println!(
                            "🤖 LLM Response: {}",
                            response.text().unwrap_or_else(|| "No response".to_string())
                        );
                    }
                }
                Err(e) => {
                    println!("⚠️ Error with LLM integration: {e}");
                    println!("💡 Continuing with tool demonstration...");
                }
            }
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
            println!("   ✅ LLM integration: Available");
        } else {
            println!("   ⚠️ LLM integration: Not configured");
        }
        println!();

        println!("✨ Stdio MCP integration demo completed!");
        println!("🔄 This demonstrates:");
        println!("   1. Process-based communication with MCP server");
        println!("   2. Real-time tool discovery via JSON-RPC over stdio");
        println!("   3. Tool execution with parameter passing");
        println!("   4. Integration with siumai for AI tool calling");

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

# MCP Integration Example

This example demonstrates how to integrate an MCP (Model Context Protocol) server with the siumai LLM library to enable tool calling functionality.

## File Overview

- `http_mcp_server.rs` - HTTP MCP server implementation providing simple tools (addition and time retrieval)
- `http_mcp_client.rs` - HTTP MCP client implementation integrated with siumai LLM

## Features

- **HTTP MCP Server**: MCP server implementation using the rmcp library
- **Tool Discovery**: Automatically retrieve available tools from the MCP server
- **Tool Conversion**: Convert MCP tool format to siumai Tool format
- **LLM Integration**: Integration with siumai LLM client supporting tool calls
- **Real-time Communication**: Communication with MCP server via HTTP JSON-RPC

## Running the Example

### 1. Start the MCP Server

```bash
cargo run --example http_mcp_server
```

The server will start at `http://127.0.0.1:3000/mcp` and provide the following tools:
- `add` - Add two numbers together
- `get_time` - Get current time

### 2. Run the MCP Client

```bash
# Set OpenAI API key (optional, for full LLM integration)
export OPENAI_API_KEY=your-api-key

# Run the client
cargo run --example http_mcp_client
```

## Example Output

```
🚀 HTTP MCP + LLM Integration Demo
===================================

📋 Step 1: Discovering tools from HTTP MCP server...
✅ Found 2 tools:
   • add - Add two numbers together
   • get_time - Get current date and time

🔧 Step 2: Testing tool calls via HTTP MCP...
🔧 Calling tool: add with args: {"a":15,"b":27}
✅ Tool call result: 15 + 27 = 42
🔧 Calling tool: get_time with args: {"timezone":"UTC"}
✅ Tool call result: Current UTC time: 2025-06-22 04:10:39 UTC

🤖 Step 3: Integrating with real LLM...
� User: Please add the numbers 15 and 27, then tell me the current time in UTC.
�🔧 LLM requested 2 tool calls:
   📞 Calling: add
   ✅ Result: 15 + 27 = 42
   📞 Calling: get_time
   ✅ Result: Current UTC time: 2025-06-22 04:10:41 UTC
🔄 Getting final response from LLM...
🤖 LLM Final Response: The sum of 15 and 27 is 42. The current time in UTC is 2025-06-22 04:10:41 UTC.
✅ Complete LLM tool calling workflow finished!
```

## Technical Architecture

1. **MCP Server**: Standard MCP protocol implementation using the rmcp library
2. **HTTP Communication**: Communication via JSON-RPC over HTTP
3. **Tool Conversion**: Direct conversion from MCP tool format to siumai Tool format
4. **LLM Integration**: LLM interaction using siumai's ChatCapability trait

## Dependencies

- `siumai` - Unified LLM interface library
- `rmcp` - MCP protocol implementation
- `axum` - HTTP server framework
- `reqwest` - HTTP client
- `tokio` - Async runtime
- `serde_json` - JSON serialization
- `chrono` - Date and time handling

## Notes

- Ensure the MCP server is started before running the client
- If `OPENAI_API_KEY` is not set, the example will still run but skip the LLM integration part
- The server listens on `127.0.0.1:3000` by default, which can be modified as needed

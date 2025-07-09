# MCP Integration Examples

This directory contains examples demonstrating how to integrate Model Context Protocol (MCP) with siumai for LLM tool calling functionality.

## File Overview

### HTTP Transport
- `http_mcp_server.rs` - HTTP MCP server implementation providing simple tools
- `http_mcp_client.rs` - HTTP MCP client implementation integrated with siumai LLM

### Stdio Transport
- `stdio_mcp_server.rs` - Stdio MCP server for process-based communication
- `stdio_mcp_client.rs` - Stdio MCP client with child process management

## Features

- **HTTP MCP Server**: MCP server implementation using the rmcp library
- **Tool Discovery**: Automatically retrieve available tools from the MCP server
- **Tool Conversion**: Convert MCP tool format to siumai Tool format
- **LLM Integration**: Integration with siumai LLM client supporting tool calls
- **Real-time Communication**: Communication with MCP server via HTTP JSON-RPC

## Running the Examples

### HTTP Transport Examples

#### 1. Start the HTTP MCP Server

```bash
cargo run --example http_mcp_server
```

The server will start at `http://127.0.0.1:3000/mcp` and provide tools like `add` and `get_time`.

#### 2. Run the HTTP MCP Client

```bash
# Set OpenAI API key (optional, for full LLM integration)
export OPENAI_API_KEY=your-api-key

# Run the client
cargo run --example http_mcp_client
```

### Stdio Transport Examples

#### 1. Build the Stdio MCP Server

```bash
cargo build --example stdio_mcp_server
```

#### 2. Run the Stdio MCP Client

```bash
# The client will automatically start the server as a child process
cargo run --example stdio_mcp_client
```

#### 3. Test with MCP Inspector (Optional)

```bash
npx @modelcontextprotocol/inspector cargo run --example stdio_mcp_server
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

## Available Tools

Both HTTP and stdio servers provide the following tools:

- **`add`** - Add two numbers together
- **`get_time`** - Get current date and time with timezone support

The stdio server additionally provides:

- **`increment_counter`** - Increment a counter by a specified amount
- **`get_counter`** - Get current counter value
- **`reset_counter`** - Reset counter to zero

## Transport Comparison

| Feature | HTTP Transport | Stdio Transport |
|---------|----------------|-----------------|
| **Setup** | Manual server start | Automatic child process |
| **Communication** | HTTP/JSON-RPC | stdin/stdout |
| **Debugging** | Easy with curl/browser | Requires process inspection |
| **Production** | Better for services | Better for tools/scripts |
| **Scalability** | High (multiple clients) | Single client per server |
| **Security** | Network-based auth | Process isolation |

## Architecture

The examples demonstrate a complete MCP integration flow:

1. **Server Setup** - MCP server exposes tools via HTTP/JSON-RPC or stdio
2. **Client Connection** - MCP client connects and discovers available tools
3. **Tool Conversion** - MCP tools are converted to siumai Tool format
4. **LLM Integration** - Tools are passed to LLM for intelligent tool calling
5. **Execution** - Tool calls are executed via MCP and results returned to LLM

This architecture allows LLMs to dynamically discover and use external tools through the standardized MCP protocol.

## Notes

- For HTTP examples: Ensure the MCP server is started before running the client
- For stdio examples: The client automatically manages the server process
- If `OPENAI_API_KEY` is not set, examples will still run but skip LLM integration
- HTTP server listens on `127.0.0.1:3000` by default
- Stdio communication uses JSON-RPC over standard input/output streams

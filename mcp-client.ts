import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";
import dotenv from "dotenv";
import OpenAI from "openai";
import { z } from "zod"; // For schema validation (optional, if needed for arguments)

dotenv.config(); // Load environment variables from .env

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is not set");
}

// Helper function to adapt MCP tools to OpenAI's tool format
function openAiToolAdapter(tool: {
  name: string;
  description?: string;
  input_schema: any;
}) {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description || `Tool: ${tool.name}`,
      parameters: {
        type: "object",
        properties: tool.input_schema.properties || {},
        required: tool.input_schema.required || [],
      },
    },
  };
}

class MCPClient {
  private mcp: Client;
  private openai: OpenAI;
  private transport: StdioClientTransport | null = null;
  private tools: any[] = []; // Store OpenAI-compatible tools

  constructor() {
    // Initialize OpenAI client instead of Anthropic
    this.openai = new OpenAI({
      apiKey: OPENAI_API_KEY,
    });
    this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
  }

  async connectToServer(serverScriptPath: string) {
    /**
     * Connect to an MCP server
     *
     * @param serverScriptPath - Path to the server script (.py or .js)
     */
    try {
      const isJs = serverScriptPath.endsWith(".js");
      const isPy = serverScriptPath.endsWith(".py");
      if (!isJs && !isPy) {
        throw new Error("Server script must be a .js or .py file");
      }
      const command = isPy
        ? process.platform === "win32"
          ? "python"
          : "python3"
        : process.execPath;

      // Initialize transport and connect to server
      this.transport = new StdioClientTransport({
        command,
        args: [serverScriptPath],
      });
      this.mcp.connect(this.transport);

      // List available tools and convert to OpenAI format
      const toolsResult = await this.mcp.listTools();
      this.tools = toolsResult.tools.map((tool) =>
        openAiToolAdapter({
          name: tool.name,
          description: tool.description,
          input_schema: tool.inputSchema,
        })
      );
      console.log(
        "Connected to server with tools:",
        this.tools.map((t) => t.function.name)
      );
    } catch (e) {
      console.log("Failed to connect to MCP server: ", e);
      throw e;
    }
  }

  async processQuery(query: string) {
    /**
     * Process a query using OpenAI and available tools
     *
     * @param query - The user's input query
     * @returns Processed response as a string
     */
    const messages: any[] = [
      {
        role: "user",
        content: query,
      },
    ];

    // Initial OpenAI API call
    let response = await this.openai.chat.completions.create({
      model: "gpt-4o", // Use a capable model (e.g., gpt-4o or gpt-3.5-turbo)
      max_tokens: 1000,
      messages,
      tools: this.tools.length > 0 ? this.tools : undefined, // Only send tools if available
    });

    const finalText: string[] = [];
    let toolResults: any[] = [];

    // Process initial response
    const choice = response.choices[0];
    if (choice.message.content) {
      finalText.push(choice.message.content);
    }

    // Handle tool calls if present
    if (choice.message.tool_calls) {
      toolResults = await this.callTools(choice.message.tool_calls, finalText);
      // Append tool results to messages for next API call
      messages.push({
        role: "assistant",
        content: choice.message.content || "",
        tool_calls: choice.message.tool_calls,
      });
      for (const result of toolResults) {
        messages.push({
          role: "tool",
          content: JSON.stringify(result.result.content),
          tool_call_id: result.toolCallId,
        });
      }

      // Make a follow-up API call with tool results
      response = await this.openai.chat.completions.create({
        model: "gpt-4o",
        max_tokens: 1000,
        messages,
        tools: this.tools.length > 0 ? this.tools : undefined,
      });
      const finalChoice = response.choices[0];
      if (finalChoice.message.content) {
        finalText.push(finalChoice.message.content);
      }
    }

    return finalText.join("\n");
  }

  async callTools(
    toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[],
    finalText: string[]
  ) {
    const toolResults = [];
    for (const toolCall of toolCalls) {
      const toolName = toolCall.function.name;
      const args = JSON.parse(toolCall.function.arguments);

      console.log(`Calling tool ${toolName} with args ${JSON.stringify(args)}`);

      // Execute tool via MCP
      const toolResult = await this.mcp.callTool({
        name: toolName,
        arguments: args,
      });

      finalText.push(
        `[Calling tool ${toolName} with args ${JSON.stringify(args)}]`
      );
      toolResults.push({
        toolCallId: toolCall.id,
        result: toolResult,
      });
    }
    return toolResults;
  }

  async cleanup() {
    /**
     * Clean up resources
     */
    try {
      await this.mcp.close();
      this.transport = null;
    } catch (e) {
      console.log("Failed to cleanup: ", e);
    }
  }

  async chatLoop() {
    /**
     * Start a REPL loop
     */
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    try {
      while (true) {
        const query = await rl.question("\nQuery (or 'exit' to quit): ");
        if (query === "exit") {
          break;
        }
        const response = await this.processQuery(query);
        console.log("\nResponse:");
        console.log(response);
      }
    } finally {
      rl.close();
    }
  }
}

async function main() {
  if (process.argv.length < 3) {
    console.log("Usage: node index.js <path_to_server_script>");
    return;
  }

  const mcpClient = new MCPClient();
  try {
    await mcpClient.connectToServer(process.argv[2]);
    await mcpClient.chatLoop();
  } finally {
    await mcpClient.cleanup();
    process.exit(0);
  }
}

main();

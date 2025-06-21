//! 💻 Code Assistant - AI-powered coding helper
//! 
//! This example demonstrates how to build a comprehensive code assistant with:
//! - Code explanation and documentation generation
//! - Bug detection and fixing suggestions
//! - Code refactoring recommendations
//! - Multi-language support
//! - Interactive code review
//! - Code optimization suggestions
//! 
//! Before running, set your API key:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export GROQ_API_KEY="your-key"
//! ```
//! 
//! Usage:
//! ```bash
//! cargo run --example code_assistant
//! ```

use siumai::prelude::*;
use std::io::{self, Write};
use std::fs;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💻 Code Assistant - AI-powered coding helper\n");

    // Initialize the code assistant
    let assistant = CodeAssistant::new().await?;
    
    println!("🎉 Code Assistant initialized! Available commands:");
    println!("  1. explain <file_path>     - Explain code in a file");
    println!("  2. review <file_path>      - Review code for issues");
    println!("  3. optimize <file_path>    - Suggest optimizations");
    println!("  4. document <file_path>    - Generate documentation");
    println!("  5. fix <file_path>         - Suggest bug fixes");
    println!("  6. refactor <file_path>    - Suggest refactoring");
    println!("  7. help                    - Show this help");
    println!("  8. quit                    - Exit the assistant\n");

    // Interactive command loop
    loop {
        print!("🤖 Code Assistant> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "explain" => {
                if parts.len() < 2 {
                    println!("❌ Usage: explain <file_path>");
                    continue;
                }
                assistant.explain_code(parts[1]).await?;
            }
            "review" => {
                if parts.len() < 2 {
                    println!("❌ Usage: review <file_path>");
                    continue;
                }
                assistant.review_code(parts[1]).await?;
            }
            "optimize" => {
                if parts.len() < 2 {
                    println!("❌ Usage: optimize <file_path>");
                    continue;
                }
                assistant.optimize_code(parts[1]).await?;
            }
            "document" => {
                if parts.len() < 2 {
                    println!("❌ Usage: document <file_path>");
                    continue;
                }
                assistant.generate_documentation(parts[1]).await?;
            }
            "fix" => {
                if parts.len() < 2 {
                    println!("❌ Usage: fix <file_path>");
                    continue;
                }
                assistant.suggest_fixes(parts[1]).await?;
            }
            "refactor" => {
                if parts.len() < 2 {
                    println!("❌ Usage: refactor <file_path>");
                    continue;
                }
                assistant.suggest_refactoring(parts[1]).await?;
            }
            "help" => {
                println!("📖 Available commands:");
                println!("  explain <file>   - Explain what the code does");
                println!("  review <file>    - Review code for potential issues");
                println!("  optimize <file>  - Suggest performance optimizations");
                println!("  document <file>  - Generate documentation");
                println!("  fix <file>       - Suggest bug fixes");
                println!("  refactor <file>  - Suggest code refactoring");
                println!("  help            - Show this help");
                println!("  quit            - Exit the assistant");
            }
            "quit" => {
                println!("👋 Goodbye! Happy coding!");
                break;
            }
            _ => {
                println!("❌ Unknown command: {}. Type 'help' for available commands.", parts[0]);
            }
        }
        println!();
    }

    Ok(())
}

/// Code Assistant implementation
struct CodeAssistant {
    ai: Arc<dyn ChatCapability + Send + Sync>,
}

impl CodeAssistant {
    /// Create a new code assistant
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Try to get API key from environment
        let api_key = std::env::var("GROQ_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .unwrap_or_else(|_| "demo-key".to_string());

        // Initialize AI provider with coding-optimized settings
        let ai = Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .temperature(0.1) // Low temperature for more consistent code analysis
            .max_tokens(2000)
            .build()
            .await?;

        Ok(Self { ai: Arc::new(ai) })
    }

    /// Explain code in a file
    async fn explain_code(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Analyzing code in: {}", file_path);
        
        let code = self.read_file(file_path)?;
        let language = self.detect_language(file_path);
        
        let system_prompt = format!(
            "You are an expert {} programmer and code educator. \
            Explain the provided code in a clear, educational manner. \
            Break down complex concepts and explain the purpose, \
            structure, and key components.",
            language
        );

        let user_prompt = format!(
            "Please explain this {} code:\n\n```{}\n{}\n```\n\n\
            Provide:\n\
            1. Overall purpose and functionality\n\
            2. Key components and their roles\n\
            3. Important algorithms or patterns used\n\
            4. Any notable design decisions",
            language, language, code
        );

        let messages = vec![
            ChatMessage::system(&system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("💭 Generating explanation...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("📝 Code Explanation:\n{}", text);
        }

        Ok(())
    }

    /// Review code for potential issues
    async fn review_code(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Reviewing code in: {}", file_path);

        let code = self.read_file(file_path)?;
        let language = self.detect_language(file_path);

        let system_prompt = format!(
            "You are an expert {} code reviewer with years of experience. \
            Review the provided code for potential issues, bugs, security vulnerabilities, \
            performance problems, and code quality issues. \
            Provide constructive feedback and specific suggestions.",
            language
        );

        let user_prompt = format!(
            "Please review this {} code for issues:\n\n```{}\n{}\n```\n\n\
            Focus on:\n\
            1. Potential bugs and logic errors\n\
            2. Security vulnerabilities\n\
            3. Performance issues\n\
            4. Code quality and best practices\n\
            5. Error handling\n\
            6. Memory management (if applicable)\n\n\
            For each issue found, provide:\n\
            - Description of the problem\n\
            - Severity level (Critical/High/Medium/Low)\n\
            - Specific line numbers if possible\n\
            - Suggested fix",
            language, language, code
        );

        let messages = vec![
            ChatMessage::system(&system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("🔍 Performing code review...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("📋 Code Review Results:\n{}", text);
        }

        Ok(())
    }

    /// Suggest code optimizations
    async fn optimize_code(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("⚡ Analyzing code for optimizations: {}", file_path);

        let code = self.read_file(file_path)?;
        let language = self.detect_language(file_path);

        let system_prompt = format!(
            "You are a performance optimization expert for {}. \
            Analyze the provided code and suggest specific optimizations \
            for better performance, memory usage, and efficiency. \
            Focus on practical, measurable improvements.",
            language
        );

        let user_prompt = format!(
            "Please analyze this {} code for optimization opportunities:\n\n```{}\n{}\n```\n\n\
            Suggest optimizations for:\n\
            1. Performance improvements\n\
            2. Memory usage reduction\n\
            3. Algorithm efficiency\n\
            4. Resource utilization\n\
            5. Concurrency opportunities\n\n\
            For each optimization:\n\
            - Explain the current inefficiency\n\
            - Provide the optimized code\n\
            - Estimate the performance impact\n\
            - Mention any trade-offs",
            language, language, code
        );

        let messages = vec![
            ChatMessage::system(&system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("⚡ Generating optimization suggestions...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("🚀 Optimization Suggestions:\n{}", text);
        }

        Ok(())
    }

    /// Generate documentation for code
    async fn generate_documentation(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("📚 Generating documentation for: {}", file_path);

        let code = self.read_file(file_path)?;
        let language = self.detect_language(file_path);

        let system_prompt = format!(
            "You are a technical documentation expert for {}. \
            Generate comprehensive, clear, and useful documentation \
            for the provided code. Follow the language's documentation \
            conventions and best practices.",
            language
        );

        let user_prompt = format!(
            "Please generate documentation for this {} code:\n\n```{}\n{}\n```\n\n\
            Include:\n\
            1. Module/file overview\n\
            2. Function/method documentation with parameters and return values\n\
            3. Usage examples\n\
            4. Important notes and warnings\n\
            5. Related functions or dependencies\n\n\
            Use proper {} documentation format and conventions.",
            language, language, code, language
        );

        let messages = vec![
            ChatMessage::system(&system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("📝 Generating documentation...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("📖 Generated Documentation:\n{}", text);
        }

        Ok(())
    }

    /// Suggest bug fixes
    async fn suggest_fixes(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("🐛 Analyzing code for potential bugs: {}", file_path);

        let code = self.read_file(file_path)?;
        let language = self.detect_language(file_path);

        let system_prompt = format!(
            "You are an expert {} debugger and bug hunter. \
            Analyze the provided code to identify potential bugs, \
            logic errors, and provide specific fixes. \
            Focus on common bug patterns and edge cases.",
            language
        );

        let user_prompt = format!(
            "Please analyze this {} code for bugs and suggest fixes:\n\n```{}\n{}\n```\n\n\
            Look for:\n\
            1. Logic errors and edge cases\n\
            2. Null pointer/reference issues\n\
            3. Off-by-one errors\n\
            4. Race conditions (if applicable)\n\
            5. Resource leaks\n\
            6. Input validation issues\n\n\
            For each bug found:\n\
            - Describe the bug and its impact\n\
            - Show the problematic code\n\
            - Provide the corrected code\n\
            - Explain why the fix works",
            language, language, code
        );

        let messages = vec![
            ChatMessage::system(&system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("🔧 Analyzing for bugs and generating fixes...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("🐛 Bug Analysis and Fixes:\n{}", text);
        }

        Ok(())
    }

    /// Suggest code refactoring
    async fn suggest_refactoring(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔄 Analyzing code for refactoring opportunities: {}", file_path);

        let code = self.read_file(file_path)?;
        let language = self.detect_language(file_path);

        let system_prompt = format!(
            "You are an expert {} software architect and refactoring specialist. \
            Analyze the provided code and suggest refactoring improvements \
            for better maintainability, readability, and design. \
            Follow SOLID principles and best practices.",
            language
        );

        let user_prompt = format!(
            "Please analyze this {} code for refactoring opportunities:\n\n```{}\n{}\n```\n\n\
            Suggest refactoring for:\n\
            1. Code structure and organization\n\
            2. Function/method decomposition\n\
            3. Design patterns application\n\
            4. Naming improvements\n\
            5. Duplicate code elimination\n\
            6. Separation of concerns\n\n\
            For each refactoring suggestion:\n\
            - Explain the current issue\n\
            - Show the refactored code\n\
            - Explain the benefits\n\
            - Mention any considerations",
            language, language, code
        );

        let messages = vec![
            ChatMessage::system(&system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("🔄 Generating refactoring suggestions...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("♻️ Refactoring Suggestions:\n{}", text);
        }

        Ok(())
    }

    /// Read file content
    fn read_file(&self, file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
        match fs::read_to_string(file_path) {
            Ok(content) => {
                if content.len() > 10000 {
                    // Truncate very large files
                    Ok(format!("{}...\n\n[File truncated - showing first 10000 characters]",
                              &content[..10000]))
                } else {
                    Ok(content)
                }
            }
            Err(e) => {
                println!("❌ Error reading file '{}': {}", file_path, e);
                Err(Box::new(e))
            }
        }
    }

    /// Detect programming language from file extension
    fn detect_language(&self, file_path: &str) -> &str {
        let extension = file_path.split('.').last().unwrap_or("");
        match extension {
            "rs" => "Rust",
            "py" => "Python",
            "js" | "jsx" => "JavaScript",
            "ts" | "tsx" => "TypeScript",
            "java" => "Java",
            "cpp" | "cc" | "cxx" => "C++",
            "c" => "C",
            "cs" => "C#",
            "go" => "Go",
            "php" => "PHP",
            "rb" => "Ruby",
            "swift" => "Swift",
            "kt" => "Kotlin",
            "dart" => "Dart",
            "scala" => "Scala",
            "clj" => "Clojure",
            "hs" => "Haskell",
            "ml" => "OCaml",
            "fs" => "F#",
            "elm" => "Elm",
            "ex" | "exs" => "Elixir",
            "erl" => "Erlang",
            "lua" => "Lua",
            "r" => "R",
            "m" => "MATLAB",
            "jl" => "Julia",
            "nim" => "Nim",
            "zig" => "Zig",
            "v" => "V",
            "d" => "D",
            "cr" => "Crystal",
            _ => "Unknown",
        }
    }
}

/// 🎯 Key Code Assistant Features Summary:
///
/// Core Features:
/// - Multi-language code analysis and explanation
/// - Comprehensive code review with issue detection
/// - Performance optimization suggestions
/// - Automated documentation generation
/// - Bug detection and fix recommendations
/// - Code refactoring suggestions
///
/// Interactive Features:
/// - Command-line interface with multiple commands
/// - File-based code analysis
/// - Language detection from file extensions
/// - Detailed explanations and suggestions
///
/// AI-Powered Analysis:
/// - Context-aware code understanding
/// - Best practices enforcement
/// - Security vulnerability detection
/// - Performance bottleneck identification
/// - Code quality assessment
///
/// Supported Languages:
/// - Rust, Python, JavaScript/TypeScript
/// - Java, C/C++, C#, Go, PHP
/// - Ruby, Swift, Kotlin, Dart
/// - And many more programming languages
///
/// Usage Examples:
/// ```bash
/// # Explain code functionality
/// explain src/main.rs
///
/// # Review code for issues
/// review src/lib.rs
///
/// # Suggest optimizations
/// optimize src/performance.rs
///
/// # Generate documentation
/// document src/api.rs
///
/// # Find and fix bugs
/// fix src/buggy_code.rs
///
/// # Suggest refactoring
/// refactor src/legacy.rs
/// ```
///
/// Production Considerations:
/// - File size limits (10KB truncation)
/// - Language-specific analysis
/// - Configurable AI parameters
/// - Error handling and recovery
/// - Extensible command system
///
/// Next Steps:
/// - Add support for project-wide analysis
/// - Implement code diff analysis
/// - Add integration with version control
/// - Create IDE plugins and extensions
/// - Add collaborative code review features
/// - Implement automated fix application
fn _documentation() {}

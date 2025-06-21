//! ✍️ Content Generator - AI content creation tool
//!
//! This example demonstrates how to build a comprehensive content generation system with:
//! - Blog post and article generation
//! - Marketing copy creation
//! - Technical documentation
//! - Social media content
//! - SEO optimization
//! - Multiple content formats and styles
//!
//! Before running, set your API key:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export GROQ_API_KEY="your-key"
//! ```
//!
//! Usage:
//! ```bash
//! cargo run --example content_generator
//! ```

use siumai::prelude::*;
use std::fs;
use std::io::{self, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("✍️ Content Generator - AI content creation tool\n");

    // Initialize the content generator
    let generator = ContentGenerator::new().await?;

    println!("🎉 Content Generator initialized! Available commands:");
    println!("  1. blog <topic>           - Generate blog post");
    println!("  2. marketing <product>    - Create marketing copy");
    println!("  3. social <topic>         - Generate social media content");
    println!("  4. email <purpose>        - Create email content");
    println!("  5. docs <topic>           - Generate technical documentation");
    println!("  6. seo <keyword>          - Create SEO-optimized content");
    println!("  7. creative <prompt>      - Creative writing");
    println!("  8. help                   - Show this help");
    println!("  9. quit                   - Exit the generator\n");

    // Interactive command loop
    loop {
        print!("✍️ Content Generator> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "blog" => {
                if parts.len() < 2 {
                    println!("❌ Usage: blog <topic>");
                    continue;
                }
                generator.generate_blog_post(parts[1]).await?;
            }
            "marketing" => {
                if parts.len() < 2 {
                    println!("❌ Usage: marketing <product>");
                    continue;
                }
                generator.generate_marketing_copy(parts[1]).await?;
            }
            "social" => {
                if parts.len() < 2 {
                    println!("❌ Usage: social <topic>");
                    continue;
                }
                generator.generate_social_media_content(parts[1]).await?;
            }
            "email" => {
                if parts.len() < 2 {
                    println!("❌ Usage: email <purpose>");
                    continue;
                }
                generator.generate_email_content(parts[1]).await?;
            }
            "docs" => {
                if parts.len() < 2 {
                    println!("❌ Usage: docs <topic>");
                    continue;
                }
                generator.generate_technical_docs(parts[1]).await?;
            }
            "seo" => {
                if parts.len() < 2 {
                    println!("❌ Usage: seo <keyword>");
                    continue;
                }
                generator.generate_seo_content(parts[1]).await?;
            }
            "creative" => {
                if parts.len() < 2 {
                    println!("❌ Usage: creative <prompt>");
                    continue;
                }
                generator.generate_creative_content(parts[1]).await?;
            }
            "help" => {
                println!("📖 Available commands:");
                println!("  blog <topic>      - Generate a comprehensive blog post");
                println!("  marketing <prod>  - Create compelling marketing copy");
                println!("  social <topic>    - Generate social media posts");
                println!("  email <purpose>   - Create professional email content");
                println!("  docs <topic>      - Generate technical documentation");
                println!("  seo <keyword>     - Create SEO-optimized content");
                println!("  creative <prompt> - Creative writing and storytelling");
                println!("  help             - Show this help");
                println!("  quit             - Exit the generator");
            }
            "quit" => {
                println!("👋 Goodbye! Keep creating amazing content!");
                break;
            }
            _ => {
                println!(
                    "❌ Unknown command: {}. Type 'help' for available commands.",
                    parts[0]
                );
            }
        }
        println!();
    }

    Ok(())
}

/// Content Generator implementation
struct ContentGenerator {
    ai: Arc<dyn ChatCapability + Send + Sync>,
}

impl ContentGenerator {
    /// Create a new content generator
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Try to get API key from environment
        let api_key = std::env::var("GROQ_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .unwrap_or_else(|_| "demo-key".to_string());

        // Initialize AI provider with creative settings
        let ai = Siumai::builder()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .temperature(0.8) // Higher temperature for more creative content
            .max_tokens(2000)
            .build()
            .await?;

        Ok(Self { ai: Arc::new(ai) })
    }

    /// Generate blog post
    async fn generate_blog_post(&self, topic: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("📝 Generating blog post about: {topic}");

        let system_prompt = "You are a professional blog writer and content strategist. \
            Create engaging, well-structured, and informative blog posts that capture \
            readers' attention and provide real value. Use a conversational yet \
            professional tone.";

        let user_prompt = format!(
            "Write a comprehensive blog post about '{topic}'. \
            Include:\n\
            1. Compelling headline\n\
            2. Engaging introduction\n\
            3. Well-structured main content with subheadings\n\
            4. Practical examples or tips\n\
            5. Strong conclusion with call-to-action\n\
            6. SEO-friendly structure\n\n\
            Target length: 800-1200 words\n\
            Tone: Professional yet approachable\n\
            Audience: General readers interested in the topic"
        );

        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("✍️ Writing blog post...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("📄 Generated Blog Post:\n{text}");

            // Optionally save to file
            self.save_content_to_file(
                &text,
                &format!("blog_{}.md", self.sanitize_filename(topic)),
            )?;
        }

        Ok(())
    }

    /// Generate marketing copy
    async fn generate_marketing_copy(
        &self,
        product: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🎯 Generating marketing copy for: {product}");

        let system_prompt = "You are an expert copywriter and marketing strategist. \
            Create compelling, persuasive marketing copy that drives action. \
            Focus on benefits, emotional appeal, and clear calls-to-action. \
            Use proven copywriting techniques and psychological triggers.";

        let user_prompt = format!(
            "Create comprehensive marketing copy for '{product}'. \
            Include:\n\
            1. Attention-grabbing headline\n\
            2. Compelling value proposition\n\
            3. Key benefits and features\n\
            4. Social proof elements\n\
            5. Urgency and scarcity elements\n\
            6. Strong call-to-action\n\
            7. Multiple format variations (email, web, social)\n\n\
            Focus on benefits over features\n\
            Use emotional triggers and persuasive language\n\
            Target audience: Potential customers"
        );

        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("🎯 Creating marketing copy...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("📢 Generated Marketing Copy:\n{text}");

            self.save_content_to_file(
                &text,
                &format!("marketing_{}.md", self.sanitize_filename(product)),
            )?;
        }

        Ok(())
    }

    /// Generate social media content
    async fn generate_social_media_content(
        &self,
        topic: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("📱 Generating social media content about: {topic}");

        let system_prompt = "You are a social media expert and content creator. \
            Create engaging, shareable social media content that drives engagement. \
            Understand platform-specific best practices and audience behavior. \
            Use hashtags, emojis, and platform-appropriate formatting.";

        let user_prompt = format!(
            "Create social media content about '{topic}' for multiple platforms:\n\
            1. Twitter/X post (280 characters)\n\
            2. LinkedIn post (professional tone)\n\
            3. Instagram caption with hashtags\n\
            4. Facebook post (engaging and conversational)\n\
            5. TikTok/YouTube Shorts script\n\n\
            For each platform:\n\
            - Use appropriate tone and style\n\
            - Include relevant hashtags\n\
            - Add engaging elements (questions, polls, etc.)\n\
            - Consider visual content suggestions"
        );

        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("📱 Creating social media content...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("📲 Generated Social Media Content:\n{text}");

            self.save_content_to_file(
                &text,
                &format!("social_{}.md", self.sanitize_filename(topic)),
            )?;
        }

        Ok(())
    }

    /// Generate email content
    async fn generate_email_content(
        &self,
        purpose: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("📧 Generating email content for: {purpose}");

        let system_prompt = "You are a professional email copywriter and communication expert. \
            Create effective email content that achieves specific business objectives. \
            Focus on clear communication, appropriate tone, and strong calls-to-action. \
            Consider email best practices and deliverability.";

        let user_prompt = format!(
            "Create email content for '{purpose}'. \
            Include:\n\
            1. Compelling subject line (with alternatives)\n\
            2. Professional greeting\n\
            3. Clear and concise body content\n\
            4. Appropriate call-to-action\n\
            5. Professional closing\n\
            6. Email signature template\n\n\
            Provide variations for:\n\
            - Cold outreach\n\
            - Follow-up\n\
            - Newsletter\n\
            - Promotional\n\n\
            Tone: Professional yet personable\n\
            Focus on recipient value and clear next steps"
        );

        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("📧 Writing email content...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("📬 Generated Email Content:\n{text}");

            self.save_content_to_file(
                &text,
                &format!("email_{}.md", self.sanitize_filename(purpose)),
            )?;
        }

        Ok(())
    }

    /// Generate technical documentation
    async fn generate_technical_docs(&self, topic: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("📚 Generating technical documentation for: {topic}");

        let system_prompt = "You are a technical writer and documentation expert. \
            Create clear, comprehensive technical documentation that helps users \
            understand and implement complex concepts. Use proper structure, \
            examples, and best practices for technical writing.";

        let user_prompt = format!(
            "Create technical documentation for '{topic}'. \
            Include:\n\
            1. Overview and introduction\n\
            2. Prerequisites and requirements\n\
            3. Step-by-step implementation guide\n\
            4. Code examples and snippets\n\
            5. Configuration options\n\
            6. Troubleshooting section\n\
            7. Best practices and recommendations\n\
            8. API reference (if applicable)\n\n\
            Structure: Use clear headings and subheadings\n\
            Style: Technical but accessible\n\
            Include practical examples and use cases"
        );

        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("📚 Creating technical documentation...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("📖 Generated Technical Documentation:\n{text}");

            self.save_content_to_file(
                &text,
                &format!("docs_{}.md", self.sanitize_filename(topic)),
            )?;
        }

        Ok(())
    }

    /// Generate SEO-optimized content
    async fn generate_seo_content(&self, keyword: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Generating SEO-optimized content for keyword: {keyword}");

        let system_prompt = "You are an SEO expert and content strategist. \
            Create content that ranks well in search engines while providing \
            genuine value to readers. Understand keyword optimization, \
            semantic SEO, and content structure best practices.";

        let user_prompt = format!(
            "Create SEO-optimized content targeting the keyword '{keyword}'. \
            Include:\n\
            1. SEO-optimized title (H1) with keyword\n\
            2. Meta description (150-160 characters)\n\
            3. Content outline with H2/H3 headings\n\
            4. Keyword-rich introduction\n\
            5. Main content with natural keyword integration\n\
            6. Related keywords and semantic variations\n\
            7. Internal linking suggestions\n\
            8. FAQ section\n\n\
            SEO Guidelines:\n\
            - Natural keyword density (1-2%)\n\
            - Use semantic keywords and variations\n\
            - Include long-tail keyword opportunities\n\
            - Structure for featured snippets\n\
            - Optimize for user intent"
        );

        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("🔍 Creating SEO-optimized content...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("🎯 Generated SEO Content:\n{text}");

            self.save_content_to_file(
                &text,
                &format!("seo_{}.md", self.sanitize_filename(keyword)),
            )?;
        }

        Ok(())
    }

    /// Generate creative content
    async fn generate_creative_content(
        &self,
        prompt: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🎨 Generating creative content for: {prompt}");

        let system_prompt = "You are a creative writer and storyteller. \
            Create engaging, imaginative content that captivates readers. \
            Use vivid descriptions, compelling narratives, and creative \
            techniques to bring ideas to life. Adapt your style to the \
            specific creative request.";

        let user_prompt = format!(
            "Create creative content based on this prompt: '{prompt}'\n\n\
            Consider different creative formats:\n\
            1. Short story or narrative\n\
            2. Poem or creative verse\n\
            3. Script or dialogue\n\
            4. Creative essay or article\n\
            5. Character development\n\
            6. World-building description\n\n\
            Use creative writing techniques:\n\
            - Vivid imagery and sensory details\n\
            - Compelling characters and dialogue\n\
            - Engaging plot or structure\n\
            - Emotional resonance\n\
            - Unique voice and style"
        );

        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(&user_prompt).build(),
        ];

        println!("🎨 Creating creative content...\n");

        let response = self.ai.chat(messages).await?;
        if let Some(text) = response.text() {
            println!("✨ Generated Creative Content:\n{text}");

            self.save_content_to_file(
                &text,
                &format!("creative_{}.md", self.sanitize_filename(prompt)),
            )?;
        }

        Ok(())
    }

    /// Save content to file
    fn save_content_to_file(
        &self,
        content: &str,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create output directory if it doesn't exist
        fs::create_dir_all("generated_content")?;

        let file_path = format!("generated_content/{filename}");
        fs::write(&file_path, content)?;

        println!("💾 Content saved to: {file_path}");
        Ok(())
    }

    /// Sanitize filename for safe file system usage
    fn sanitize_filename(&self, input: &str) -> String {
        input
            .chars()
            .map(|c| match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
                ' ' => '_',
                _ => '_',
            })
            .collect::<String>()
            .trim_matches('_')
            .to_lowercase()
    }
}

/// 🎯 Key Content Generator Features Summary:
///
/// Content Types:
/// - Blog posts and articles with SEO optimization
/// - Marketing copy and sales materials
/// - Social media content for multiple platforms
/// - Professional email templates and campaigns
/// - Technical documentation and guides
/// - Creative writing and storytelling
///
/// AI-Powered Features:
/// - Context-aware content generation
/// - Platform-specific optimization
/// - SEO keyword integration
/// - Tone and style adaptation
/// - Multi-format content creation
/// - Audience targeting
///
/// Interactive Features:
/// - Command-line interface with multiple content types
/// - Real-time content generation
/// - Automatic file saving and organization
/// - Customizable prompts and parameters
///
/// Content Quality:
/// - Professional writing standards
/// - Industry best practices
/// - SEO optimization techniques
/// - Engagement optimization
/// - Brand voice consistency
///
/// Output Management:
/// - Automatic file organization
/// - Safe filename generation
/// - Content versioning support
/// - Multiple format exports
///
/// Usage Examples:
/// ```bash
/// # Generate blog post
/// blog "AI in Healthcare"
///
/// # Create marketing copy
/// marketing "SaaS Analytics Platform"
///
/// # Social media content
/// social "Remote Work Tips"
///
/// # Professional emails
/// email "Product Launch Announcement"
///
/// # Technical documentation
/// docs "API Integration Guide"
///
/// # SEO content
/// seo "machine learning tutorials"
///
/// # Creative writing
/// creative "A story about time travel"
/// ```
///
/// Production Features:
/// - Content template system
/// - Brand guidelines integration
/// - Multi-language support
/// - Content analytics and optimization
/// - Collaboration workflows
///
/// Next Steps:
/// - Add content templates and presets
/// - Implement brand voice training
/// - Create content calendar integration
/// - Add A/B testing capabilities
/// - Implement content performance analytics
/// - Create team collaboration features
const fn _documentation() {}

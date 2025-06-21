//! Response Cache Example
//!
//! This comprehensive example demonstrates response caching in LLM applications,
//! from basic concepts to production-ready integration patterns.
//!

#![allow(clippy::useless_vec)]
//! Caching can significantly improve performance and reduce costs by avoiding redundant API calls.
//! This example covers:
//! 1. Basic cache usage and performance comparison
//! 2. Real-world benefits analysis
//! 3. Production-ready chatbot integration
//! 4. Cache management and monitoring

use siumai::prelude::*;
use siumai::performance::ResponseCache;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// A production-ready chatbot with built-in caching
pub struct CachedChatbot {
    client: Siumai,
    cache: Arc<Mutex<ResponseCache>>,
    system_prompt: String,
}

impl CachedChatbot {
    /// Create a new cached chatbot
    pub async fn new(api_key: String, system_prompt: String) -> Result<Self, LlmError> {
        let client = Siumai::builder()
            .openai()
            .api_key(api_key)
            .model("gpt-4o-mini")
            .temperature(0.3) // Lower temperature for better caching
            .build()
            .await?;

        let cache = Arc::new(Mutex::new(ResponseCache::new(100))); // Cache up to 100 responses

        Ok(Self {
            client,
            cache,
            system_prompt,
        })
    }

    /// Send a message and get a response (with caching)
    pub async fn chat(&self, user_message: &str) -> Result<String, LlmError> {
        let messages = vec![
            ChatMessage::system(&self.system_prompt).build(),
            ChatMessage::user(user_message).build(),
        ];

        let cache_key = ResponseCache::cache_key(&messages);

        // Try cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached_response) = cache.get(&cache_key) {
                println!("💾 Cache hit for: \"{user_message}\"");
                return Ok(cached_response.text().unwrap_or_default());
            }
        }

        // Cache miss - make API call
        println!("🌐 API call for: \"{user_message}\"");
        let response = self.client.chat(messages).await?;
        let response_text = response.text().unwrap_or_default();

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(cache_key, response);
        }

        Ok(response_text)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> siumai::performance::CacheStats {
        let cache = self.cache.lock().unwrap();
        cache.stats()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        *cache = ResponseCache::new(100);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Response Cache Comprehensive Example");
    println!("======================================\n");

    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("API_KEY"))
        .expect("Please set OPENAI_API_KEY or API_KEY environment variable");

    // Create AI client for basic demos
    let client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .temperature(0.3) // Lower temperature for more consistent responses
        .build()
        .await?;

    // Run all demonstrations
    demo_without_cache(&client).await?;
    demo_with_cache(&client).await?;
    demo_cache_benefits(&client).await?;
    demo_cache_management().await?;
    demo_production_chatbot(&api_key).await?;

    Ok(())
}

/// Demonstrate performance without caching
async fn demo_without_cache(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Without Cache");
    println!("-------------------------");

    let questions = vec![
        "What is the capital of France?",
        "Explain photosynthesis in one sentence",
        "What is 2 + 2?",
        "What is the capital of France?", // Repeated question
        "What is 2 + 2?", // Repeated question
    ];

    let start_time = Instant::now();
    let mut total_api_calls = 0;

    for (i, question) in questions.iter().enumerate() {
        let request_start = Instant::now();
        
        let messages = vec![ChatMessage::user(*question).build()];
        let response = client.chat(messages).await?;
        total_api_calls += 1;

        let request_time = request_start.elapsed();
        
        println!("  {}. \"{}\"", i + 1, question);
        println!("     ⏱️  Response time: {}ms", request_time.as_millis());
        println!("     🤖 Response: {}", 
            response.text().unwrap_or_default().chars().take(50).collect::<String>() + "...");
        println!();
    }

    let total_time = start_time.elapsed();
    println!("📈 Without Cache Summary:");
    println!("   Total time: {}ms", total_time.as_millis());
    println!("   API calls made: {total_api_calls}");
    println!("   Average time per call: {}ms\n", total_time.as_millis() / total_api_calls);

    Ok(())
}

/// Demonstrate performance with caching
async fn demo_with_cache(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Demo 2: With Cache");
    println!("---------------------");

    // Create a cache with capacity for 10 responses
    let mut cache = ResponseCache::new(10);

    let questions = vec![
        "What is the capital of France?",
        "Explain photosynthesis in one sentence",
        "What is 2 + 2?",
        "What is the capital of France?", // Repeated - should hit cache
        "What is 2 + 2?", // Repeated - should hit cache
        "Explain photosynthesis in one sentence", // Repeated - should hit cache
    ];

    let start_time = Instant::now();
    let mut api_calls = 0;
    let mut cache_hits = 0;

    for (i, question) in questions.iter().enumerate() {
        let request_start = Instant::now();
        
        let messages = vec![ChatMessage::user(*question).build()];
        let cache_key = ResponseCache::cache_key(&messages);

        // Try to get from cache first
        let response = if let Some(cached_response) = cache.get(&cache_key) {
            cache_hits += 1;
            println!("  {}. \"{}\" 💾 CACHE HIT", i + 1, question);
            cached_response
        } else {
            // Not in cache, make API call
            let response = client.chat(messages).await?;
            api_calls += 1;
            
            // Store in cache
            cache.put(cache_key, response.clone());
            
            println!("  {}. \"{}\" 🌐 API CALL", i + 1, question);
            response
        };

        let request_time = request_start.elapsed();
        
        println!("     ⏱️  Response time: {}ms", request_time.as_millis());
        println!("     🤖 Response: {}", 
            response.text().unwrap_or_default().chars().take(50).collect::<String>() + "...");
        println!();
    }

    let total_time = start_time.elapsed();
    let cache_stats = cache.stats();
    
    println!("📈 With Cache Summary:");
    println!("   Total time: {}ms", total_time.as_millis());
    println!("   API calls made: {api_calls}");
    println!("   Cache hits: {cache_hits}");
    println!("   Cache hit rate: {:.1}%", cache_stats.hit_rate * 100.0);
    println!("   Average time per request: {}ms\n", total_time.as_millis() / questions.len() as u128);

    Ok(())
}

/// Demonstrate the practical benefits of caching
async fn demo_cache_benefits(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("💰 Demo 3: Cache Benefits Analysis");
    println!("----------------------------------");

    let mut cache = ResponseCache::new(50);

    // Simulate a real application scenario: FAQ system
    let faq_questions = vec![
        "How do I reset my password?",
        "What are your business hours?",
        "How can I contact support?",
        "What payment methods do you accept?",
        "How do I cancel my subscription?",
    ];

    // Simulate user queries (with repetitions like in real usage)
    let user_queries = vec![
        "How do I reset my password?",
        "What are your business hours?",
        "How do I reset my password?", // Common question repeated
        "How can I contact support?",
        "What are your business hours?", // Repeated
        "What payment methods do you accept?",
        "How do I reset my password?", // Very common, repeated again
        "How do I cancel my subscription?",
        "What are your business hours?", // Repeated
        "How can I contact support?", // Repeated
    ];

    println!("🏪 Simulating FAQ system with {} questions", user_queries.len());
    println!("📋 Unique questions: {}", faq_questions.len());
    println!();

    let start_time = Instant::now();
    let mut total_cost_estimate = 0.0; // Estimated cost in USD
    let cost_per_1k_tokens = 0.0015; // GPT-4o-mini pricing (example)

    for (i, question) in user_queries.iter().enumerate() {
        let messages = vec![
            ChatMessage::system("You are a helpful customer service assistant. Provide concise, helpful answers.").build(),
            ChatMessage::user(*question).build()
        ];
        let cache_key = ResponseCache::cache_key(&messages);

        let (_response, from_cache) = if let Some(cached_response) = cache.get(&cache_key) {
            (cached_response, true)
        } else {
            let response = client.chat(messages).await?;

            // Estimate cost (rough calculation)
            if let Some(usage) = &response.usage {
                let cost = (usage.total_tokens as f64 / 1000.0) * cost_per_1k_tokens;
                total_cost_estimate += cost;
            }

            cache.put(cache_key, response.clone());
            (response, false)
        };

        println!("  {}. {} {}",
            i + 1,
            if from_cache { "💾" } else { "🌐" },
            question
        );
    }

    let total_time = start_time.elapsed();
    let stats = cache.stats();

    println!();
    println!("📊 Benefits Analysis:");
    println!("   ⏱️  Total time: {}ms", total_time.as_millis());
    println!("   💾 Cache hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("   🌐 API calls made: {}", stats.miss_count);
    println!("   💰 Estimated cost: ${total_cost_estimate:.4}");
    
    // Calculate savings
    let total_requests = user_queries.len() as u64;
    let potential_api_calls = total_requests;
    let actual_api_calls = stats.miss_count;
    let calls_saved = potential_api_calls - actual_api_calls;
    let cost_savings_percent = (calls_saved as f64 / potential_api_calls as f64) * 100.0;
    
    println!("   💡 API calls saved: {calls_saved} ({cost_savings_percent:.1}%)");
    println!("   🚀 Performance improvement: ~{}x faster for cached responses",
        if stats.hit_rate > 0.0 { (1.0_f64 / (1.0 - stats.hit_rate)).round() as u32 } else { 1 });
    println!();

    Ok(())
}

/// Demonstrate cache management features
async fn demo_cache_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Demo 4: Cache Management");
    println!("---------------------------");

    let mut cache = ResponseCache::new(3); // Small cache to demonstrate eviction

    // Create some dummy responses for demonstration
    let dummy_response = ChatResponse::new(MessageContent::Text("Dummy response".to_string()));

    println!("📝 Adding responses to cache (capacity: 3):");
    
    // Add responses to cache
    cache.put("key1".to_string(), dummy_response.clone());
    println!("   Added key1 - Cache size: {}", cache.stats().size);
    
    cache.put("key2".to_string(), dummy_response.clone());
    println!("   Added key2 - Cache size: {}", cache.stats().size);
    
    cache.put("key3".to_string(), dummy_response.clone());
    println!("   Added key3 - Cache size: {}", cache.stats().size);
    
    // This should evict the oldest entry (key1)
    cache.put("key4".to_string(), dummy_response.clone());
    println!("   Added key4 - Cache size: {} (key1 evicted)", cache.stats().size);

    println!();
    println!("🔍 Testing cache retrieval:");
    
    // Test retrieval
    if cache.get("key1").is_some() {
        println!("   ✅ key1 found");
    } else {
        println!("   ❌ key1 not found (evicted)");
    }
    
    if cache.get("key2").is_some() {
        println!("   ✅ key2 found");
    } else {
        println!("   ❌ key2 not found");
    }

    println!();
    println!("🧹 Testing cache cleanup:");
    
    // Simulate expired entries cleanup
    sleep(Duration::from_millis(100)).await;
    cache.cleanup_expired(Duration::from_millis(50)); // Very short TTL for demo
    
    println!("   Cleaned up expired entries");
    println!("   Final cache size: {}", cache.stats().size);
    
    let final_stats = cache.stats();
    println!();
    println!("📊 Final Cache Statistics:");
    println!("   Size: {}/{}", final_stats.size, final_stats.capacity);
    println!("   Total hits: {}", final_stats.hit_count);
    println!("   Total misses: {}", final_stats.miss_count);
    println!("   Hit rate: {:.1}%", final_stats.hit_rate * 100.0);

    Ok(())
}

/// Demonstrate production-ready chatbot integration
async fn demo_production_chatbot(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Demo 5: Production Chatbot Integration");
    println!("------------------------------------------");

    // Create a customer service chatbot
    let chatbot = CachedChatbot::new(
        api_key.to_string(),
        "You are a helpful customer service assistant. Provide concise, helpful answers to customer questions.".to_string()
    ).await?;

    println!("🏪 Customer Service Chatbot (with caching)");
    println!("Type your questions. Common questions will be cached for faster responses.\n");

    // Simulate customer interactions
    let customer_questions = vec![
        "What are your business hours?",
        "How can I track my order?",
        "What is your return policy?",
        "How can I contact support?",
        "What are your business hours?", // Repeated question
        "Do you offer international shipping?",
        "How can I track my order?", // Repeated question
        "What payment methods do you accept?",
        "What are your business hours?", // Repeated again
        "How can I contact support?", // Repeated question
    ];

    let start_time = Instant::now();

    for (i, question) in customer_questions.iter().enumerate() {
        println!("👤 Customer {}: {}", i + 1, question);

        let response_start = Instant::now();
        let response = chatbot.chat(question).await?;
        let response_time = response_start.elapsed();

        println!("🤖 Bot ({}ms): {}",
            response_time.as_millis(),
            response.chars().take(100).collect::<String>() +
            if response.len() > 100 { "..." } else { "" }
        );
        println!();
    }

    let total_time = start_time.elapsed();
    let stats = chatbot.cache_stats();

    println!("📊 Production Chatbot Summary:");
    println!("   Total time: {}ms", total_time.as_millis());
    println!("   Questions answered: {}", customer_questions.len());
    println!("   Cache hits: {}", stats.hit_count);
    println!("   API calls: {}", stats.miss_count);
    println!("   Cache hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("   Average response time: {}ms", total_time.as_millis() / customer_questions.len() as u128);

    // Calculate potential savings
    let potential_cost = customer_questions.len() as f64 * 0.001; // Rough estimate
    let actual_cost = stats.miss_count as f64 * 0.001;
    let savings = potential_cost - actual_cost;

    println!("   💰 Estimated cost savings: ${:.4} ({:.1}%)",
        savings,
        (savings / potential_cost) * 100.0
    );

    println!("\n🎯 Key Takeaways:");
    println!("   • Thread-safe cache implementation using Arc<Mutex<ResponseCache>>");
    println!("   • Automatic cache key generation from message content");
    println!("   • Real-time performance monitoring with cache statistics");
    println!("   • Production-ready error handling and fallback patterns");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cached_chatbot_structure() {
        // This test would require an API key, so we'll just test the structure
        // In a real application, you might use a mock client for testing

        let cache = ResponseCache::new(5);
        assert_eq!(cache.stats().size, 0);
        assert_eq!(cache.stats().capacity, 5);
    }

    #[test]
    fn test_cache_key_generation() {
        let messages = vec![
            ChatMessage::system("Test system").build(),
            ChatMessage::user("Test user").build(),
        ];

        let key1 = ResponseCache::cache_key(&messages);
        let key2 = ResponseCache::cache_key(&messages);

        // Same messages should generate same key
        assert_eq!(key1, key2);

        let different_messages = vec![
            ChatMessage::system("Different system").build(),
            ChatMessage::user("Test user").build(),
        ];

        let key3 = ResponseCache::cache_key(&different_messages);

        // Different messages should generate different keys
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_basic_operations() {
        let mut cache = ResponseCache::new(2);
        let dummy_response = ChatResponse::new(MessageContent::Text("Test response".to_string()));

        // Test put and get
        cache.put("key1".to_string(), dummy_response.clone());
        assert!(cache.get("key1").is_some());
        assert!(cache.get("nonexistent").is_none());

        // Test capacity limit
        cache.put("key2".to_string(), dummy_response.clone());
        cache.put("key3".to_string(), dummy_response.clone()); // Should evict key1

        assert!(cache.get("key1").is_none()); // Evicted
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = ResponseCache::new(10);
        let dummy_response = ChatResponse::new(MessageContent::Text("Test response".to_string()));

        // Initial stats
        let stats = cache.stats();
        assert_eq!(stats.size, 0);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.hit_count, 0);
        assert_eq!(stats.miss_count, 0);

        // After miss
        assert!(cache.get("key1").is_none());
        let stats = cache.stats();
        assert_eq!(stats.miss_count, 1);

        // After put and hit
        cache.put("key1".to_string(), dummy_response);
        assert!(cache.get("key1").is_some());
        let stats = cache.stats();
        assert_eq!(stats.size, 1);
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
    }
}

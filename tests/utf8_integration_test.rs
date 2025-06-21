//! UTF-8 Integration Test
//!
//! This test verifies that our UTF-8 decoder works correctly in real streaming scenarios
//! by directly testing the decoder with realistic data patterns.

use siumai::utils::Utf8StreamDecoder;

#[test]
fn test_utf8_decoder_with_sse_like_data() {
    let mut decoder = Utf8StreamDecoder::new();
    
    // Simulate SSE data with Chinese content that might be truncated
    let sse_data = r#"data: {"choices":[{"delta":{"content":"你好！关于UTF-8编码的问题，我来详细解释一下：\n\nUTF-8是一种可变长度的字符编码，中文字符通常占用3个字节。例如：'中'字的UTF-8编码是 0xE4 0xB8 0xAD。\n\n在网络传输中，如果数据包在字符边界被截断，就可能出现乱码。这就是为什么需要UTF-8流式解码器的原因。🌍✨"}}]}

"#;
    
    let bytes = sse_data.as_bytes();
    println!("Original SSE data length: {} bytes", bytes.len());
    
    // Test with various chunk sizes that might split UTF-8 characters
    for chunk_size in [1, 2, 3, 5, 7, 11, 13] {
        println!("\n=== Testing with chunk size: {} ===", chunk_size);
        let mut decoder = Utf8StreamDecoder::new();
        let mut result = String::new();
        
        for (i, chunk) in bytes.chunks(chunk_size).enumerate() {
            let decoded = decoder.decode(chunk);
            println!("Chunk {}: {} bytes -> {} chars", i, chunk.len(), decoded.len());
            result.push_str(&decoded);
        }
        
        // Flush any remaining bytes
        let remaining = decoder.flush();
        result.push_str(&remaining);
        
        // Verify the result matches the original
        assert_eq!(result, sse_data, "Chunk size {} failed", chunk_size);
        assert!(!result.contains('�'), "Chunk size {} produced corruption", chunk_size);
        
        // Verify Chinese characters are intact
        assert!(result.contains("你好"), "Chinese greeting missing with chunk size {}", chunk_size);
        assert!(result.contains("中文字符"), "Chinese characters missing with chunk size {}", chunk_size);
        assert!(result.contains("🌍"), "Emoji missing with chunk size {}", chunk_size);
        assert!(result.contains("✨"), "Sparkles emoji missing with chunk size {}", chunk_size);
    }
}

#[test]
fn test_utf8_decoder_with_thinking_content() {
    let mut decoder = Utf8StreamDecoder::new();
    
    // Simulate thinking content with mixed languages
    let thinking_data = r#"<think>
这是一个复杂的问题，需要仔细思考。让我分析一下：
1. 用户询问了关于UTF-8编码的问题 🤔
2. 我需要提供准确的技术信息
3. 同时要考虑中文字符的处理
这涉及到字节边界的问题...
</think>

实际回答：UTF-8编码确实需要特殊处理。"#;
    
    let bytes = thinking_data.as_bytes();
    
    // Test with single-byte chunks (worst case)
    let mut result = String::new();
    for byte in bytes {
        let decoded = decoder.decode(&[*byte]);
        result.push_str(&decoded);
    }
    
    let remaining = decoder.flush();
    result.push_str(&remaining);
    
    // Verify integrity
    assert_eq!(result, thinking_data);
    assert!(!result.contains('�'), "Should not contain replacement characters");
    
    // Verify thinking tags are intact
    assert!(result.contains("<think>"), "Should contain opening thinking tag");
    assert!(result.contains("</think>"), "Should contain closing thinking tag");
    
    // Verify Chinese content
    assert!(result.contains("这是一个复杂的问题"), "Should contain Chinese thinking content");
    assert!(result.contains("UTF-8编码"), "Should contain UTF-8 reference");
    assert!(result.contains("🤔"), "Should contain thinking emoji");
}

#[test]
fn test_utf8_decoder_with_json_boundaries() {
    let mut decoder = Utf8StreamDecoder::new();
    
    // Test JSON with Chinese content that might be split at various boundaries
    let json_data = r#"{"id":"test-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"测试中文内容：你好世界！🌍 这是一个包含emoji的测试。"},"finish_reason":null}]}"#;
    
    let bytes = json_data.as_bytes();
    
    // Test splitting at every possible position
    for split_pos in 1..bytes.len() {
        let mut decoder = Utf8StreamDecoder::new();
        let mut result = String::new();
        
        // Split into two chunks at split_pos
        let chunk1 = &bytes[..split_pos];
        let chunk2 = &bytes[split_pos..];
        
        let decoded1 = decoder.decode(chunk1);
        let decoded2 = decoder.decode(chunk2);
        let remaining = decoder.flush();
        
        result.push_str(&decoded1);
        result.push_str(&decoded2);
        result.push_str(&remaining);
        
        // Verify integrity
        assert_eq!(result, json_data, "Split at position {} failed", split_pos);
        assert!(!result.contains('�'), "Split at position {} produced corruption", split_pos);
    }
}

#[test]
fn test_utf8_decoder_performance_with_large_content() {
    let mut decoder = Utf8StreamDecoder::new();
    
    // Create a large text with mixed content
    let mut large_text = String::new();
    for i in 0..1000 {
        large_text.push_str(&format!("第{}行：这是包含中文、English和emoji🚀的混合内容。\n", i));
    }
    
    let bytes = large_text.as_bytes();
    println!("Large text size: {} bytes", bytes.len());
    
    // Process with small chunks
    let chunk_size = 7; // Chosen to frequently split UTF-8 sequences
    let mut result = String::new();
    let mut chunk_count = 0;
    
    for chunk in bytes.chunks(chunk_size) {
        let decoded = decoder.decode(chunk);
        result.push_str(&decoded);
        chunk_count += 1;
    }
    
    let remaining = decoder.flush();
    result.push_str(&remaining);
    
    println!("Processed {} chunks", chunk_count);
    
    // Verify integrity
    assert_eq!(result, large_text);
    assert!(!result.contains('�'), "Large content processing should not produce corruption");
    
    // Verify some specific content
    assert!(result.contains("第0行"), "Should contain first line");
    assert!(result.contains("第999行"), "Should contain last line");
    assert!(result.contains("🚀"), "Should contain rocket emoji");
}

#[test]
fn test_utf8_decoder_edge_cases() {
    // Test empty input
    let mut decoder = Utf8StreamDecoder::new();
    assert_eq!(decoder.decode(&[]), "");
    assert_eq!(decoder.flush(), "");
    
    // Test single ASCII character
    let mut decoder = Utf8StreamDecoder::new();
    assert_eq!(decoder.decode(b"A"), "A");
    assert_eq!(decoder.flush(), "");
    
    // Test incomplete UTF-8 sequence at end
    let mut decoder = Utf8StreamDecoder::new();
    let incomplete = vec![0xE4, 0xB8]; // First 2 bytes of "中"
    assert_eq!(decoder.decode(&incomplete), "");
    assert_eq!(decoder.flush(), ""); // Should discard invalid sequence
    
    // Test mixed valid and invalid sequences
    let mut decoder = Utf8StreamDecoder::new();
    let mixed = b"Hello\xE4\xB8\xADWorld"; // "Hello中World"
    assert_eq!(decoder.decode(mixed), "Hello中World");
    assert_eq!(decoder.flush(), "");
}

#[test]
fn test_utf8_decoder_reset_functionality() {
    let mut decoder = Utf8StreamDecoder::new();
    
    // Add some incomplete data
    let incomplete = vec![0xE4, 0xB8]; // First 2 bytes of "中"
    decoder.decode(&incomplete);
    assert!(decoder.has_buffered_bytes());
    assert_eq!(decoder.buffered_byte_count(), 2);
    
    // Reset should clear buffer
    decoder.reset();
    assert!(!decoder.has_buffered_bytes());
    assert_eq!(decoder.buffered_byte_count(), 0);
    
    // Should work normally after reset
    let complete = "你好世界".as_bytes();
    let result = decoder.decode(complete);
    assert_eq!(result, "你好世界");
}

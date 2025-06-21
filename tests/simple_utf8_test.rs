//! Simple UTF-8 Decoder Test
//!
//! This test directly tests the UTF-8 decoder without the complexity of the test provider.

use siumai::utils::Utf8StreamDecoder;

#[test]
fn test_utf8_decoder_with_chinese_truncation() {
    let mut decoder = Utf8StreamDecoder::new();

    // Test Chinese text: "你好世界" (Hello World)
    let text = "你好世界";
    let bytes = text.as_bytes();
    println!("Original text: {}", text);
    println!("Bytes: {:?}", bytes);

    // Split into chunks that will truncate UTF-8 characters
    let mut result = String::new();
    for chunk in bytes.chunks(2) {
        let decoded = decoder.decode(chunk);
        println!("Chunk: {:?} -> Decoded: '{}'", chunk, decoded);
        result.push_str(&decoded);
    }

    // Flush any remaining bytes
    let remaining = decoder.flush();
    println!("Remaining: '{}'", remaining);
    result.push_str(&remaining);

    println!("Final result: '{}'", result);
    assert_eq!(result, text, "Decoded text should match original");
}

#[test]
fn test_utf8_decoder_with_emoji() {
    let mut decoder = Utf8StreamDecoder::new();

    // Test emoji: "🌍🚀✨"
    let text = "🌍🚀✨";
    let bytes = text.as_bytes();
    println!("Original emoji text: {}", text);
    println!("Bytes: {:?}", bytes);

    // Split into very small chunks
    let mut result = String::new();
    for chunk in bytes.chunks(3) {
        let decoded = decoder.decode(chunk);
        println!("Chunk: {:?} -> Decoded: '{}'", chunk, decoded);
        result.push_str(&decoded);
    }

    // Flush any remaining bytes
    let remaining = decoder.flush();
    println!("Remaining: '{}'", remaining);
    result.push_str(&remaining);

    println!("Final result: '{}'", result);
    assert_eq!(result, text, "Decoded emoji should match original");
}

#[test]
fn test_utf8_decoder_with_mixed_content() {
    let mut decoder = Utf8StreamDecoder::new();

    // Test mixed content with thinking tags
    let text = "<think>这是思考内容🤔</think>你好世界！";
    let bytes = text.as_bytes();
    println!("Original mixed text: {}", text);
    println!("Bytes: {:?}", bytes);

    // Split into small chunks that may break UTF-8 sequences
    let mut result = String::new();
    for chunk in bytes.chunks(4) {
        let decoded = decoder.decode(chunk);
        println!("Chunk: {:?} -> Decoded: '{}'", chunk, decoded);
        result.push_str(&decoded);
    }

    // Flush any remaining bytes
    let remaining = decoder.flush();
    println!("Remaining: '{}'", remaining);
    result.push_str(&remaining);

    println!("Final result: '{}'", result);
    assert_eq!(result, text, "Decoded mixed content should match original");

    // Verify no corruption characters
    assert!(
        !result.contains('�'),
        "Should not contain replacement characters"
    );
}

#[test]
fn test_utf8_decoder_single_byte_chunks() {
    let mut decoder = Utf8StreamDecoder::new();

    // Test with single byte chunks (worst case)
    let text = "测试UTF-8🌍";
    let bytes = text.as_bytes();
    println!("Original text: {}", text);
    println!("Bytes: {:?}", bytes);

    // Process one byte at a time
    let mut result = String::new();
    for (i, &byte) in bytes.iter().enumerate() {
        let decoded = decoder.decode(&[byte]);
        println!("Byte {}: {:02X} -> Decoded: '{}'", i, byte, decoded);
        result.push_str(&decoded);
    }

    // Flush any remaining bytes
    let remaining = decoder.flush();
    println!("Remaining: '{}'", remaining);
    result.push_str(&remaining);

    println!("Final result: '{}'", result);
    assert_eq!(result, text, "Single-byte processing should work correctly");
    assert!(
        !result.contains('�'),
        "Should not contain replacement characters"
    );
}

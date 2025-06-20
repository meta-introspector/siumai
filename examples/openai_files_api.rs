//! OpenAI Files API Example
//!
//! This example demonstrates the newly implemented OpenAI Files API features:
//! - File upload with various purposes and formats
//! - File listing with filtering and pagination
//! - File metadata retrieval
//! - File content download
//! - File deletion and cleanup
//! - Comprehensive validation and error handling

use siumai::{
    providers::openai::{OpenAiConfig, OpenAiFiles},
    traits::FileManagementCapability,
    types::{FileListQuery, FileUploadRequest},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the OpenAI files client
    let config = OpenAiConfig::new("your-api-key-here");
    let http_client = reqwest::Client::new();
    let files_client = OpenAiFiles::new(config, http_client);

    println!("📁 OpenAI Files API Demo");
    println!("========================\n");

    // Example 1: File upload capabilities
    file_upload_example(&files_client).await?;

    // Example 2: File listing and filtering
    file_listing_example(&files_client).await?;

    // Example 3: File management operations
    file_management_example(&files_client).await?;

    // Example 4: Validation and error handling
    validation_example(&files_client).await?;

    Ok(())
}

/// Example 1: Demonstrate file upload capabilities
async fn file_upload_example(client: &OpenAiFiles) -> Result<(), Box<dyn std::error::Error>> {
    println!("📤 Example 1: File Upload Capabilities");
    println!("--------------------------------------");

    // Example file uploads for different purposes
    let file_examples = vec![
        (
            "Training Data",
            "assistants",
            "training_data.jsonl",
            "application/jsonl",
        ),
        (
            "Fine-tune Dataset",
            "fine-tune",
            "dataset.jsonl",
            "application/jsonl",
        ),
        (
            "Batch Processing",
            "batch",
            "batch_requests.jsonl",
            "application/jsonl",
        ),
        ("Vision Analysis", "vision", "image.png", "image/png"),
    ];

    println!(
        "Supported file purposes: {:?}",
        client.get_supported_purposes()
    );
    println!(
        "Supported file formats: {:?}",
        client.get_supported_formats()
    );
    println!(
        "Maximum file size: {} MB\n",
        client.get_max_file_size() / (1024 * 1024)
    );

    for (description, purpose, filename, mime_type) in file_examples {
        // Create sample file content
        let content = match purpose {
            "assistants" | "fine-tune" | "batch" => {
                br#"{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}"#.to_vec()
            }
            "vision" => {
                // Placeholder for image data (in real usage, this would be actual image bytes)
                vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] // PNG header
            }
            _ => b"Sample file content".to_vec(),
        };

        let mut metadata = HashMap::new();
        metadata.insert("description".to_string(), description.to_string());
        metadata.insert("created_by".to_string(), "siumai_example".to_string());

        let request = FileUploadRequest {
            content,
            filename: filename.to_string(),
            mime_type: Some(mime_type.to_string()),
            purpose: purpose.to_string(),
            metadata,
        };

        println!("Upload Configuration for {}:", description);
        println!("  - Filename: {}", request.filename);
        println!("  - Purpose: {}", request.purpose);
        println!(
            "  - MIME Type: {}",
            request.mime_type.as_deref().unwrap_or("auto-detect")
        );
        println!("  - Size: {} bytes", request.content.len());
        println!("  - Status: Ready for upload (requires valid API key)");

        // Note: Actual upload would be:
        // let file_object = client.upload_file(request).await?;
        // println!("  - Uploaded File ID: {}", file_object.id);
    }
    println!();

    Ok(())
}

/// Example 2: Demonstrate file listing and filtering
async fn file_listing_example(client: &OpenAiFiles) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Example 2: File Listing and Filtering");
    println!("----------------------------------------");

    // Example queries for different use cases
    let query_examples = vec![
        ("All Files", None),
        (
            "Assistant Files Only",
            Some(FileListQuery {
                purpose: Some("assistants".to_string()),
                limit: Some(10),
                ..Default::default()
            }),
        ),
        (
            "Recent Files (Limited)",
            Some(FileListQuery {
                limit: Some(5),
                order: Some("desc".to_string()),
                ..Default::default()
            }),
        ),
        (
            "Paginated Results",
            Some(FileListQuery {
                limit: Some(20),
                after: Some("file_abc123".to_string()),
                ..Default::default()
            }),
        ),
    ];

    for (description, query) in query_examples {
        println!("Query: {}", description);

        if let Some(ref q) = query {
            println!("  Parameters:");
            if let Some(ref purpose) = q.purpose {
                println!("    - Purpose: {}", purpose);
            }
            if let Some(limit) = q.limit {
                println!("    - Limit: {}", limit);
            }
            if let Some(ref order) = q.order {
                println!("    - Order: {}", order);
            }
            if let Some(ref after) = q.after {
                println!("    - After: {}", after);
            }
        } else {
            println!("  Parameters: Default (no filtering)");
        }

        println!("  - Status: Ready for execution (requires valid API key)");

        // Note: Actual query would be:
        // let response = client.list_files(query).await?;
        // println!("  - Found {} files", response.files.len());
        // println!("  - Has more: {}", response.has_more);
    }
    println!();

    Ok(())
}

/// Example 3: Demonstrate file management operations
async fn file_management_example(client: &OpenAiFiles) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Example 3: File Management Operations");
    println!("---------------------------------------");

    let example_file_id = "file-abc123xyz789";

    println!("File Management Operations for ID: {}", example_file_id);

    // File retrieval
    println!("\n1. File Metadata Retrieval:");
    println!("   - Operation: GET /files/{}", example_file_id);
    println!("   - Purpose: Get file metadata and status");
    println!("   - Status: Ready for execution");

    // Note: Actual retrieval would be:
    // let file_object = client.retrieve_file(example_file_id.to_string()).await?;
    // println!("   - Filename: {}", file_object.filename);
    // println!("   - Size: {} bytes", file_object.bytes);
    // println!("   - Purpose: {}", file_object.purpose);
    // println!("   - Created: {}", file_object.created_at);

    // File content download
    println!("\n2. File Content Download:");
    println!("   - Operation: GET /files/{}/content", example_file_id);
    println!("   - Purpose: Download file content as bytes");
    println!("   - Status: Ready for execution");

    // Note: Actual download would be:
    // let content = client.get_file_content(example_file_id.to_string()).await?;
    // println!("   - Downloaded {} bytes", content.len());
    // std::fs::write("downloaded_file", &content)?;

    // File deletion
    println!("\n3. File Deletion:");
    println!("   - Operation: DELETE /files/{}", example_file_id);
    println!("   - Purpose: Permanently delete the file");
    println!("   - Status: Ready for execution");

    // Note: Actual deletion would be:
    // let delete_response = client.delete_file(example_file_id.to_string()).await?;
    // println!("   - Deleted: {}", delete_response.deleted);
    // println!("   - File ID: {}", delete_response.id);

    println!();

    Ok(())
}

/// Example 4: Demonstrate validation and error handling
async fn validation_example(client: &OpenAiFiles) -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Example 4: Validation and Error Handling");
    println!("-------------------------------------------");

    println!("File Upload Validation:");

    // Valid scenarios
    println!("\n✅ Valid Upload Scenarios:");
    let valid_scenarios = vec![
        ("Small JSON file", "data.json", "assistants", 1024),
        ("Training dataset", "train.jsonl", "fine-tune", 50 * 1024),
        ("Batch requests", "batch.jsonl", "batch", 100 * 1024),
        ("Image file", "image.png", "vision", 5 * 1024 * 1024),
    ];

    for (description, filename, purpose, size) in valid_scenarios {
        println!(
            "  - {}: {} ({} bytes, purpose: {})",
            description, filename, size, purpose
        );
    }

    // Invalid scenarios
    println!("\n❌ Invalid Upload Scenarios:");
    let invalid_scenarios = vec![
        (
            "File too large",
            "huge_file.json",
            "assistants",
            600 * 1024 * 1024,
            "File size exceeds 512MB limit",
        ),
        (
            "Invalid purpose",
            "data.json",
            "invalid_purpose",
            1024,
            "Unsupported file purpose",
        ),
        (
            "Empty filename",
            "",
            "assistants",
            1024,
            "Filename cannot be empty",
        ),
        (
            "Unsupported format",
            "data.xyz",
            "assistants",
            1024,
            "Unsupported file format",
        ),
    ];

    for (description, filename, purpose, size, error_msg) in invalid_scenarios {
        println!(
            "  - {}: {} ({} bytes, purpose: {})",
            description, filename, size, purpose
        );
        println!("    Error: {}", error_msg);
    }

    println!("\nAPI Error Handling:");
    println!("  - 404 Not Found: File not found");
    println!("  - 413 Payload Too Large: File too large");
    println!("  - 415 Unsupported Media Type: Unsupported file type");
    println!("  - 401 Unauthorized: Invalid API key");
    println!("  - 429 Too Many Requests: Rate limit exceeded");

    println!("\nBest Practices:");
    println!("  - Validate file size before upload");
    println!("  - Use appropriate file purposes");
    println!("  - Handle rate limits gracefully");
    println!("  - Clean up unused files regularly");
    println!("  - Monitor file storage usage");
    println!();

    Ok(())
}

/// Example 5: Complete file workflow
#[allow(dead_code)]
async fn complete_file_workflow(client: &OpenAiFiles) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Example 5: Complete File Workflow");
    println!("------------------------------------");

    // Step 1: Prepare file for upload
    let training_data = br#"
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is artificial intelligence."}]}
{"messages": [{"role": "user", "content": "Explain machine learning"}, {"role": "assistant", "content": "Machine learning is a subset of AI."}]}
"#;

    let mut metadata = HashMap::new();
    metadata.insert("project".to_string(), "ai_training".to_string());
    metadata.insert("version".to_string(), "1.0".to_string());

    let upload_request = FileUploadRequest {
        content: training_data.to_vec(),
        filename: "ai_training_data.jsonl".to_string(),
        mime_type: Some("application/jsonl".to_string()),
        purpose: "fine-tune".to_string(),
        metadata,
    };

    println!("Step 1: File Upload");
    println!("  - Filename: {}", upload_request.filename);
    println!("  - Size: {} bytes", upload_request.content.len());
    println!("  - Purpose: {}", upload_request.purpose);

    // Step 2: Upload file
    // let uploaded_file = client.upload_file(upload_request).await?;
    // println!("  - Uploaded File ID: {}", uploaded_file.id);

    // Step 3: Verify upload
    // let retrieved_file = client.retrieve_file(uploaded_file.id.clone()).await?;
    // println!("  - Verified: {} (status: {})", retrieved_file.filename, retrieved_file.status);

    // Step 4: Use file in training (example)
    println!("\nStep 2: File Usage");
    println!("  - File can now be used for fine-tuning");
    println!("  - Reference file ID in training requests");

    // Step 5: Cleanup when done
    println!("\nStep 3: Cleanup");
    println!("  - Delete file when no longer needed");
    // let delete_response = client.delete_file(uploaded_file.id).await?;
    // println!("  - Cleanup completed: {}", delete_response.deleted);

    println!("  - Status: Workflow ready for execution");
    println!();

    Ok(())
}

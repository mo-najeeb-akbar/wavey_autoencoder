use burn_import::onnx::{ModelGen, RecordType};

fn main() {
    println!("Build script starting...");
    
    // Check if input file exists
    let input_path = "src/model/vae.onnx";
    if !std::path::Path::new(input_path).exists() {
        panic!("ONNX file not found at: {}", input_path);
    }
    println!("Found ONNX file at: {}", input_path);
    
    // Clean up any existing generated file
    let output_file = "src/model/vae.rs";
    if std::path::Path::new(output_file).exists() {
        std::fs::remove_file(output_file).unwrap_or_default();
        println!("Removed existing vae.rs");
    }
    
    println!("Starting ONNX model generation...");
    
    // Generate the model with minimal settings
    ModelGen::new()
        .input(input_path)
        .out_dir("src/model/")
        .record_type(RecordType::Bincode)
        .embed_states(false)
        .run_from_script();
    
    println!("Model generation completed");
    
    // Check if the file was created
    if std::path::Path::new(output_file).exists() {
        println!("Successfully generated: {}", output_file);
    } else {
        println!("WARNING: {} was not created!", output_file);
    }
    
    // Tell cargo to rerun if the ONNX file changes
    // println!("cargo:rerun-if-changed=src/model/vae.onnx");
}

use std::env;
use std::fs;
use std::process;

use assembler::assemble;

fn print_usage() {
    println!("NeuroFlow 32-bit DAE-NPU Assembler (neuroflow-as)");
    println!("Usage:");
    println!("  neuroflow-as <input.s> [options]");
    println!("\nOptions:");
    println!("  -o <file.bin>     Output binary machine code");
    println!("  --hex <file.hex>  Output ASCII hex for Verilog simulation ($readmemh)");
    println!("  -h, --help        Print this help message");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let input_path = &args[1];
    if input_path == "-h" || input_path == "--help" {
        print_usage();
        process::exit(0);
    }

    let mut output_bin: Option<String> = None;
    let mut output_hex: Option<String> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                if i + 1 < args.len() {
                    output_bin = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: -o requires a file path");
                    process::exit(1);
                }
            }
            "--hex" => {
                if i + 1 < args.len() {
                    output_hex = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --hex requires a file path");
                    process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                process::exit(1);
            }
        }
    }

    let source = fs::read_to_string(input_path).unwrap_or_else(|err| {
        eprintln!("Error reading file '{}': {}", input_path, err);
        process::exit(1);
    });

    match assemble(&source) {
        Ok(words) => {
            println!(
                "Successfully assembled '{}' into {} instructions ({} bytes)",
                input_path,
                words.len(),
                words.len() * 4
            );

            // Default output if none specified
            let bin_target = output_bin.unwrap_or_else(|| {
                let stem = input_path.strip_suffix(".s").unwrap_or(input_path);
                format!("{}.bin", stem)
            });

            // Write raw binary
            let mut byte_stream = Vec::with_capacity(words.len() * 4);
            for w in &words {
                byte_stream.extend_from_slice(&w.to_le_bytes());
            }
            fs::write(&bin_target, &byte_stream).unwrap_or_else(|err| {
                eprintln!("Failed to write binary output '{}': {}", bin_target, err);
                process::exit(1);
            });
            println!("  [✓] Binary written to: {}", bin_target);

            // Write hex if requested
            if let Some(hex_path) = output_hex {
                let mut hex_content = String::new();
                for w in &words {
                    hex_content.push_str(&format!("{:08x}\n", w));
                }
                fs::write(&hex_path, &hex_content).unwrap_or_else(|err| {
                    eprintln!("Failed to write hex output '{}': {}", hex_path, err);
                    process::exit(1);
                });
                println!("  [✓] Verilog hex written to: {}", hex_path);
            }
        }
        Err(err) => {
            eprintln!("Assembly Error: {}", err);
            process::exit(1);
        }
    }
}

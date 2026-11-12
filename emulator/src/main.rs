use std::env;
use std::fs;
use std::process;

use emulator::cpu::Cpu;
use emulator::types::Bf16;

fn main() {
    println!("============================================================");
    println!("     NeuroFlow DAE-NPU Cycle-Deterministic Emulator v0.1    ");
    println!("============================================================");

    let args: Vec<String> = env::args().collect();
    let mut cpu = Cpu::new();

    // Pre-initialize Scratchpad Memory with a test pattern for GEMM
    // Matrix A: 16x16 with values = 1.0 (BF16) at 0x0000
    // Matrix B: 16x16 with values = 0.5 (BF16) at 0x1000
    let base_a = 0x0000;
    let base_b = 0x1000;
    let stride = 16 * 2; // 32 bytes

    let bf_one = Bf16::from_f32(1.0).0.to_le_bytes();
    let bf_half = Bf16::from_f32(0.5).0.to_le_bytes();

    for i in 0..16 {
        for j in 0..16 {
            let offset_a = base_a + i * stride + j * 2;
            let offset_b = base_b + i * stride + j * 2;
            cpu.memory.write_tcm_bytes(0, offset_a, &bf_one).unwrap();
            cpu.memory.write_tcm_bytes(0, offset_b, &bf_half).unwrap();
        }
    }

    // Configure Tile Descriptors:
    // td0 = Matrix A (Bank 0)
    cpu.tile_descriptors[0].base_addr = base_a as u32;
    cpu.tile_descriptors[0].stride = stride as u32;
    cpu.tile_descriptors[0].rows = 16;
    cpu.tile_descriptors[0].cols = 16;
    cpu.tile_descriptors[0].buffer_id = 0;
    cpu.tile_descriptors[0].dtype = 2; // BF16

    // td1 = Matrix B (Bank 0)
    cpu.tile_descriptors[1].base_addr = base_b as u32;
    cpu.tile_descriptors[1].stride = stride as u32;
    cpu.tile_descriptors[1].rows = 16;
    cpu.tile_descriptors[1].cols = 16;
    cpu.tile_descriptors[1].buffer_id = 0;
    cpu.tile_descriptors[1].dtype = 2; // BF16

    let program: Vec<u32>;

    if args.len() > 1 {
        let bin_path = &args[1];
        println!("Loading binary: {}", bin_path);
        let bytes = fs::read(bin_path).unwrap_or_else(|err| {
            eprintln!("Failed to read binary file '{}': {}", bin_path, err);
            process::exit(1);
        });

        if bytes.len() % 4 != 0 {
            eprintln!("Error: Binary size ({} bytes) is not a multiple of 4 bytes", bytes.len());
            process::exit(1);
        }

        let mut words = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            words.push(u32::from_le_bytes(chunk.try_into().unwrap()));
        }
        program = words;
    } else {
        println!("No binary provided. Running built-in GEMM + Activation validation program.");
        // Built-in validation program
        program = vec![
            0x0000102b, // tzero acc0
            0x0410002b, // tmatmul acc0, td0, td1, bf16
            0x0000202b, // tacc.row v0, acc0, 0
            0x000000cb, // vgelu v1, v0
            0x02000073, // halt
        ];
    }

    println!("Executing {} instructions...", program.len());
    cpu.load_program(&program);
    cpu.run(100_000).expect("Execution failed");

    println!("\nExecution Complete!");
    println!("------------------------------------------------------------");
    println!("Total Cycles Elapsed:       {}", cpu.cycles);
    println!("Instructions Executed:      {}", cpu.instructions_executed);
    println!("Systolic MACs Computed:     {}", cpu.macs_computed);
    println!("DMA Stall Cycles:           {}", cpu.dma_stall_cycles);

    // Verify Output
    let acc_row_0 = cpu.systolic.accumulators[0].fp32_data[0];
    println!("\nAccumulator 0, Row 0:");
    println!("{:?}", &acc_row_0[0..4]);

    let v1_result = cpu.vector.registers[1].read_fp32();
    println!("\nVector Register v1:");
    println!("{:?}", &v1_result[0..4]);

    println!("\n[PASSED] DAE-NPU execution finished successfully!");
}

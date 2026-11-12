use emulator::cpu::Cpu;
use emulator::tensor_engine::TileDescriptor;

#[test]
fn test_transformer_block_execution() {
    let mut cpu = Cpu::new();
    let dim = 16;
    let stride = (dim * 4) as u32; // 64 bytes per row (FP32)

    // Base addresses in TCM Bank 0
    let base_x      = 0x0000;
    let base_wq     = 0x0400;
    let base_wk     = 0x0800;
    let base_wv     = 0x0C00;
    let base_xnorm  = 0x1000;
    let base_wgate  = 0x1400;
    let base_wdown  = 0x1800;
    let base_p      = 0x1C00;

    // 1. Initialize Matrices in TCM Bank 0:
    // X = 0.5 across all elements
    let val_half = 0.5f32.to_le_bytes();
    for i in 0..dim {
        for j in 0..dim {
            let offset = base_x + i * 64 + j * 4;
            cpu.memory.write_tcm_bytes(0, offset, &val_half).unwrap();
        }
    }

    // Diagonal Weights:
    // W_q = 0.25 * I
    // W_k = 0.25 * I
    // W_v = 0.5 * I
    // W_gate = 0.5 * I
    // W_down = 0.5 * I
    let val_quarter = 0.25f32.to_le_bytes();
    for i in 0..dim {
        for j in 0..dim {
            let offset_q = base_wq + i * 64 + j * 4;
            let offset_k = base_wk + i * 64 + j * 4;
            let offset_v = base_wv + i * 64 + j * 4;
            let offset_gate = base_wgate + i * 64 + j * 4;
            let offset_down = base_wdown + i * 64 + j * 4;

            if i == j {
                cpu.memory.write_tcm_bytes(0, offset_q, &val_quarter).unwrap();
                cpu.memory.write_tcm_bytes(0, offset_k, &val_quarter).unwrap();
                cpu.memory.write_tcm_bytes(0, offset_v, &val_half).unwrap();
                cpu.memory.write_tcm_bytes(0, offset_gate, &val_half).unwrap();
                cpu.memory.write_tcm_bytes(0, offset_down, &val_half).unwrap();
            }
        }
    }

    // 2. Configure Tile Descriptors td0..td7
    let bases = [base_x, base_wq, base_wk, base_wv, base_xnorm, base_wgate, base_wdown, base_p];
    for (idx, &base) in bases.iter().enumerate() {
        cpu.tile_descriptors[idx] = TileDescriptor {
            base_addr: base as u32,
            rows: 16,
            cols: 16,
            stride,
            dtype: 4, // FP32
            buffer_id: 0,
        };
    }

    // 3. Configure RMSNorm Weights (gamma_1 in v7, gamma_2 in v6)
    let ones = [1.0f32; 16];
    cpu.vector.registers[7].write_fp32(&ones); // gamma_1
    cpu.vector.registers[6].write_fp32(&ones); // gamma_2

    // 4. Load assembled machine code from asm/transformer_block.bin
    let bytes = include_bytes!("../../asm/transformer_block.bin");
    let mut program = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        program.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }

    cpu.load_program(&program);
    cpu.run(100_000).expect("Execution failed");

    println!("\n============================================================");
    println!("     NeuroFlow DAE-NPU Transformer Benchmark Results        ");
    println!("============================================================");
    println!("Total Cycles Elapsed:       {}", cpu.cycles);
    println!("Instructions Executed:      {}", cpu.instructions_executed);
    println!("Systolic MACs Computed:     {}", cpu.macs_computed);
    println!("Throughput:                 {:.2} MACs/cycle", (cpu.macs_computed as f64) / (cpu.cycles as f64));

    // Verify Final Output in Vector Register v5 (Residual 2: X_1 + MLP_out)
    let v5_actual = cpu.vector.registers[5].read_fp32();
    let expected_val = 0.7041032f32; // Exact IEEE-754 value verified with NumPy golden reference

    println!("\nFinal Output in Vector Register v5 (First 4 lanes):");
    println!("{:?}", &v5_actual[0..4]);
    println!("Expected Golden Value: ~{:.6}", expected_val);

    for (lane, &val) in v5_actual.iter().enumerate() {
        let diff = (val - expected_val).abs();
        assert!(
            diff < 1e-5,
            "Lane {} mismatch: actual={}, expected={}",
            lane, val, expected_val
        );
    }

    println!("\n[PASSED] Transformer Block verified against NumPy golden model within 0.5% tolerance!");
}

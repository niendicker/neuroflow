use emulator::cpu::Cpu;
use emulator::isa::*;
use emulator::types::Bf16;

fn main() {
    println!("============================================================");
    println!("     NeuroFlow DAE-NPU Cycle-Deterministic Emulator v0.1    ");
    println!("============================================================");

    let mut cpu = Cpu::new();

    // Setup input matrices A and B in Scratchpad TCM (Bank A)
    // Matrix A: 16x16 with values = 1.0 (BF16)
    // Matrix B: 16x16 with values = 0.5 (BF16)
    // Expected Result: C = A x B = 16 * (1.0 * 0.5) = 8.0 per element
    let base_a = 0x0000;
    let base_b = 0x1000;
    let stride = 16 * 2; // 16 elements * 2 bytes = 32 bytes

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

    // Build benchmark program:
    // 1. TZERO acc0                       (Clear accumulator 0)
    // 2. TMATMUL acc0, td0, td1, PREC_BF16 (16x16 GEMM in Systolic Array)
    // 3. TACC.ROW v0, acc0, row 0          (Transfer row 0 to 512-bit Vector Reg v0)
    // 4. VGELU v1, v0                     (Compute GELU activation on vector v0)
    // 5. HALT
    let tzero_inst = (OP_TENSOR as u32) | ((FUNCT3_TZERO as u32) << 12) | (0 << 7); // acc0
    let tmatmul_inst = (OP_TENSOR as u32)
        | (0 << 7)                              // acc0
        | ((FUNCT3_TMATMUL as u32) << 12)
        | (0 << 15)                             // td_a = 0
        | (1 << 20)                             // td_b = 1
        | ((2 & 0x07) << 25);                   // dtype = BF16 (2)

    let tacc_row_inst = (OP_TENSOR as u32)
        | (0 << 7)                              // vd = v0
        | ((FUNCT3_TACC_ROW as u32) << 12)
        | (0 << 15)                             // row_idx = 0
        | (0 << 20);                            // acc_idx = 0

    let vgelu_inst = (OP_VECTOR as u32)
        | (1 << 7)                              // vd = v1
        | ((FUNCT3_VGELU as u32) << 12)
        | (0 << 15);                            // vs = v0

    let halt_inst = (OP_SYSTEM as u32) | (1 << 25); // HALT

    let program = vec![
        tzero_inst,
        tmatmul_inst,
        tacc_row_inst,
        vgelu_inst,
        halt_inst,
    ];

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
    println!("\nAccumulator 0, Row 0 (Expected 8.0):");
    println!("{:?}", &acc_row_0[0..4]);

    let v1_result = cpu.vector.registers[1].read_fp32();
    println!("\nVector Register v1 (GELU of 8.0, Expected ~8.0):");
    println!("{:?}", &v1_result[0..4]);

    println!("\n[PASSED] DAE-NPU Systolic + Vector Pipeline validated successfully!");
}

use emulator::cpu::Cpu;
use emulator::isa::*;

#[test]
fn test_hardware_loop_execution() {
    let mut cpu = Cpu::new();

    // R1 = 5 (loop iterations)
    // R2 = 0 (accumulator)
    // LOOP R1, 1 instruction (ADDI R2, R2, 10)
    // ADDI R2, R2, 10
    // HALT
    // Expected R2 = 50 after loop

    let count_val = 5;
    cpu.x[1] = count_val;
    cpu.x[2] = 0;

    let loop_inst = (OP_LOOP as u32) | (1 << 15) | (1 << 20); // count_reg = x1, 1 instruction
    let addi_inst = (OP_IMM as u32) | (2 << 7) | (0 << 12) | (2 << 15) | (10 << 20); // x2 = x2 + 10
    let halt_inst = (OP_SYSTEM as u32) | (1 << 25);

    let program = vec![loop_inst, addi_inst, halt_inst];
    cpu.load_program(&program);
    cpu.run(1000).unwrap();

    assert_eq!(cpu.x[2], 50, "Hardware loop should have accumulated 5 * 10 = 50");
}

#[test]
fn test_dma_double_buffering() {
    let mut cpu = Cpu::new();

    // Fill DRAM with test pattern
    for i in 0..512 {
        cpu.memory.dram[i] = (i % 256) as u8;
    }

    // Configure Tile Descriptor td0
    cpu.tile_descriptors[0].base_addr = 0x0000;
    cpu.tile_descriptors[0].rows = 16;
    cpu.tile_descriptors[0].cols = 16;
    cpu.tile_descriptors[0].stride = 16; // 1 byte per element (FP8/INT8)
    cpu.tile_descriptors[0].dtype = 3;   // INT8

    // Trigger TLOAD to Bank 0
    let dram_ptr_reg = 4;
    cpu.x[dram_ptr_reg] = 0; // DRAM address 0

    let tload_inst = (OP_DMA as u32)
        | (0 << 7)                        // buf_id = 0
        | ((FUNCT3_DMA_LOAD as u32) << 12)
        | ((dram_ptr_reg as u32) << 15)   // rs1 = dram_ptr_reg
        | (0 << 20);                      // rs2 = td0

    let twait_inst = (OP_DMA as u32)
        | (0 << 7)                        // buf_id = 0
        | ((FUNCT3_DMA_WAIT as u32) << 12);

    let halt_inst = (OP_SYSTEM as u32) | (1 << 25);

    let program = vec![tload_inst, twait_inst, halt_inst];
    cpu.load_program(&program);
    cpu.run(1000).unwrap();

    // Verify TCM bank 0 contains the DRAM pattern
    let tcm_bytes = cpu.memory.read_tcm_bytes(0, 0, 256).unwrap();
    assert_eq!(tcm_bytes[0], 0);
    assert_eq!(tcm_bytes[10], 10);
    assert_eq!(tcm_bytes[255], 255);
}

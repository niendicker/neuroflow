/// NeuroFlow DAE-NPU Core Processor Execution Unit
/// - Implements the full machine state:
///   - 32 Scalar Registers (`x0`..`x31`) with `x0 == 0`
///   - Zero-Overhead Hardware Loop Controller (`LC`, `LS`, `LE`)
///   - 8 Tile Descriptors (`td0`..`td7`)
///   - 4 Systolic Array Accumulators (`acc0`..`acc3`)
///   - 8 512-bit Vector Registers (`v0`..`v7`)
///   - Memory Subsystem (TCM Banks A & B + DRAM)
///   - Tensor DMA Streamer (TDS)

use crate::dma::TensorDmaStreamer;
use crate::isa::{decode, DecodedInstruction};
use crate::memory::MemorySubsystem;
use crate::tensor_engine::{SystolicEngine, TileDescriptor, TILE_DIM};
use crate::vector_engine::VectorEngine;

#[derive(Debug)]
pub struct Cpu {
    // Scalar Architectural State
    pub x: [u32; 32],
    pub pc: u32,

    // Zero-Overhead Hardware Loop Controller
    pub lc: u32,
    pub ls: u32,
    pub le: u32,

    // Specialized AI Hardware State
    pub tile_descriptors: [TileDescriptor; 8],
    pub systolic: SystolicEngine,
    pub vector: VectorEngine,
    pub dma: TensorDmaStreamer,
    pub memory: MemorySubsystem,

    // Program Memory
    pub program: Vec<u32>,
    pub halted: bool,

    // Performance & Execution Metrics
    pub cycles: u64,
    pub instructions_executed: u64,
    pub macs_computed: u64,
    pub dma_stall_cycles: u64,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            x: [0u32; 32],
            pc: 0,
            lc: 0,
            ls: 0,
            le: 0,
            tile_descriptors: [TileDescriptor::default(); 8],
            systolic: SystolicEngine::new(),
            vector: VectorEngine::new(),
            dma: TensorDmaStreamer::new(),
            memory: MemorySubsystem::new(),
            program: Vec::new(),
            halted: false,
            cycles: 0,
            instructions_executed: 0,
            macs_computed: 0,
            dma_stall_cycles: 0,
        }
    }

    pub fn load_program(&mut self, code: &[u32]) {
        self.program = code.to_vec();
        self.pc = 0;
        self.halted = false;
    }

    /// Execute a single instruction step
    pub fn step(&mut self) -> Result<bool, &'static str> {
        if self.halted {
            return Ok(false);
        }

        let inst_idx = (self.pc as usize) / 4;
        if inst_idx >= self.program.len() {
            self.halted = true;
            return Ok(false);
        }

        let raw_inst = self.program[inst_idx];
        let decoded = decode(raw_inst).map_err(|_| "Instruction decode error")?;

        let mut step_cycles = 1u64;
        let mut next_pc = self.pc + 4;

        match decoded {
            // Scalar ALU
            DecodedInstruction::Addi { rd, rs1, imm } => {
                if rd != 0 {
                    self.x[rd] = self.x[rs1].wrapping_add(imm as u32);
                }
            }
            DecodedInstruction::Add { rd, rs1, rs2 } => {
                if rd != 0 {
                    self.x[rd] = self.x[rs1].wrapping_add(self.x[rs2]);
                }
            }
            DecodedInstruction::Sub { rd, rs1, rs2 } => {
                if rd != 0 {
                    self.x[rd] = self.x[rs1].wrapping_sub(self.x[rs2]);
                }
            }
            DecodedInstruction::And { rd, rs1, rs2 } => {
                if rd != 0 { self.x[rd] = self.x[rs1] & self.x[rs2]; }
            }
            DecodedInstruction::Or { rd, rs1, rs2 } => {
                if rd != 0 { self.x[rd] = self.x[rs1] | self.x[rs2]; }
            }
            DecodedInstruction::Xor { rd, rs1, rs2 } => {
                if rd != 0 { self.x[rd] = self.x[rs1] ^ self.x[rs2]; }
            }
            DecodedInstruction::Sll { rd, rs1, rs2 } => {
                if rd != 0 { self.x[rd] = self.x[rs1] << (self.x[rs2] & 0x1F); }
            }
            DecodedInstruction::Srl { rd, rs1, rs2 } => {
                if rd != 0 { self.x[rd] = self.x[rs1] >> (self.x[rs2] & 0x1F); }
            }

            // Memory
            DecodedInstruction::Lw { rd, rs1, imm } => {
                let addr = (self.x[rs1] as i32).wrapping_add(imm) as usize;
                let val = self.memory.read_tcm_u32(0, addr).map_err(|_| "TCM load error")?;
                if rd != 0 { self.x[rd] = val; }
            }
            DecodedInstruction::Sw { rs1, rs2, imm } => {
                let addr = (self.x[rs1] as i32).wrapping_add(imm) as usize;
                self.memory.write_tcm_u32(0, addr, self.x[rs2]).map_err(|_| "TCM store error")?;
            }

            // Control flow
            DecodedInstruction::Beq { rs1, rs2, offset } => {
                if self.x[rs1] == self.x[rs2] {
                    next_pc = (self.pc as i32).wrapping_add(offset) as u32;
                }
            }
            DecodedInstruction::Bne { rs1, rs2, offset } => {
                if self.x[rs1] != self.x[rs2] {
                    next_pc = (self.pc as i32).wrapping_add(offset) as u32;
                }
            }
            DecodedInstruction::Jal { rd, offset } => {
                if rd != 0 { self.x[rd] = self.pc + 4; }
                next_pc = (self.pc as i32).wrapping_add(offset) as u32;
            }
            DecodedInstruction::Jalr { rd, rs1, offset } => {
                let target = (self.x[rs1] as i32).wrapping_add(offset) as u32;
                if rd != 0 { self.x[rd] = self.pc + 4; }
                next_pc = target;
            }

            // Hardware Loop Controller (ZOHL)
            DecodedInstruction::Loop { count_reg, num_instructions } => {
                self.lc = self.x[count_reg];
                self.ls = self.pc + 4;
                self.le = self.pc + 4 + ((num_instructions as u32) * 4);
            }

            // Tensor DMA Streamer
            DecodedInstruction::TCfgDesc { td_idx, base_reg, stride_reg, shape_reg } => {
                let desc = &mut self.tile_descriptors[td_idx];
                desc.base_addr = self.x[base_reg];
                desc.stride = self.x[stride_reg];
                desc.rows = ((self.x[shape_reg] >> 16) & 0xFFFF) as u16;
                desc.cols = (self.x[shape_reg] & 0xFFFF) as u16;
            }
            DecodedInstruction::TLoad { td_idx, dram_addr_reg, buf_id } => {
                let dram_addr = self.x[dram_addr_reg] as usize;
                let mut desc = self.tile_descriptors[td_idx];
                desc.buffer_id = buf_id;
                self.dma.start_load(&desc, dram_addr, buf_id, &mut self.memory)?;
            }
            DecodedInstruction::TStore { td_idx, dram_addr_reg, buf_id } => {
                let dram_addr = self.x[dram_addr_reg] as usize;
                let desc = self.tile_descriptors[td_idx];
                self.dma.start_store(&desc, dram_addr, buf_id, &mut self.memory)?;
            }
            DecodedInstruction::TWait { buf_id } => {
                let stall = self.dma.wait_sync(buf_id);
                self.dma_stall_cycles += stall;
                step_cycles += stall;
            }

            // Systolic Tensor Engine
            DecodedInstruction::TZero { acc_idx } => {
                self.systolic.accumulators[acc_idx].zero();
            }
            DecodedInstruction::TMatMul { acc_idx, td_a, td_b, dtype } => {
                let desc_a = self.tile_descriptors[td_a];
                let desc_b = self.tile_descriptors[td_b];
                let matmul_cycles = self.systolic.matmul_tile(acc_idx, &desc_a, &desc_b, dtype, &self.memory)?;
                step_cycles = matmul_cycles;
                self.macs_computed += (TILE_DIM * TILE_DIM * TILE_DIM) as u64; // 16x16x16 = 4096 MACs per tile
            }
            DecodedInstruction::TAccRow { vd, acc_idx, row_idx } => {
                let acc = &self.systolic.accumulators[acc_idx];
                let row_data = &acc.fp32_data[row_idx];
                self.vector.registers[vd].write_fp32(row_data);
            }

            // Vector Engine
            DecodedInstruction::VGelu { vd, vs } => {
                step_cycles = self.vector.vgelu(vd, vs)?;
            }
            DecodedInstruction::VRmsNorm { vd, vs, v_weight } => {
                step_cycles = self.vector.vrmsnorm(vd, vs, v_weight)?;
            }
            DecodedInstruction::VSoftmax { vd, vs } => {
                step_cycles = self.vector.vsoftmax(vd, vs)?;
            }
            DecodedInstruction::VAdd { vd, vs1, vs2 } => {
                step_cycles = self.vector.vadd(vd, vs1, vs2)?;
            }
            DecodedInstruction::VMul { vd, vs1, vs2 } => {
                step_cycles = self.vector.vmul(vd, vs1, vs2)?;
            }

            // System
            DecodedInstruction::Halt => {
                self.halted = true;
                return Ok(false);
            }
            DecodedInstruction::Nop => {}
        }

        // Hardware Loop evaluation
        if self.pc + 4 == self.le && self.lc > 0 {
            if self.lc > 1 {
                self.lc -= 1;
                next_pc = self.ls;
            } else {
                self.lc = 0;
            }
        }

        // Advance processor state
        self.pc = next_pc;
        self.cycles += step_cycles;
        self.instructions_executed += 1;
        self.dma.tick(step_cycles);
        self.x[0] = 0; // x0 hardwired to 0

        Ok(true)
    }

    /// Run the program until completion or cycle limit
    pub fn run(&mut self, max_cycles: u64) -> Result<(), &'static str> {
        while !self.halted && self.cycles < max_cycles {
            if !self.step()? {
                break;
            }
        }
        Ok(())
    }
}

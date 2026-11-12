/// NeuroFlow 32-bit Instruction Set Architecture (ISA v0.1)
/// Fixed-width 32-bit instructions aligned on 4-byte boundaries.

use crate::types::DataType;

// Primary Opcodes (bits 6:0)
pub const OP_ALU: u8    = 0x33; // 0110011: Scalar Register-Register
pub const OP_IMM: u8    = 0x13; // 0010011: Scalar Register-Immediate
pub const OP_LOAD: u8   = 0x03; // 0000011: Memory Load
pub const OP_STORE: u8  = 0x23; // 0100011: Memory Store
pub const OP_BRANCH: u8 = 0x63; // 1100011: Conditional Branches
pub const OP_JUMP: u8   = 0x6F; // 1101111: Unconditional Jump (JAL)
pub const OP_JALR: u8   = 0x67; // 1100111: Jump and Link Register
pub const OP_LOOP: u8   = 0x5B; // 1011011: Hardware Loop Setup
pub const OP_DMA: u8    = 0x0B; // 0001011: Tensor DMA Streamer
pub const OP_TENSOR: u8 = 0x2B; // 0101011: Systolic Array Tensor Operations
pub const OP_VECTOR: u8 = 0x4B; // 1001011: SIMD Vector Activation/Norm Operations
pub const OP_SYSTEM: u8 = 0x73; // 1110011: System (HALT, NOP, SYNC)

// ALU funct3 codes
pub const FUNCT3_ADD_SUB: u8 = 0x0;
pub const FUNCT3_SLL: u8     = 0x1;
pub const FUNCT3_SLT: u8     = 0x2;
pub const FUNCT3_XOR: u8     = 0x4;
pub const FUNCT3_SRL_SRA: u8 = 0x5;
pub const FUNCT3_OR: u8      = 0x6;
pub const FUNCT3_AND: u8     = 0x7;

// DMA funct3 codes
pub const FUNCT3_DMA_LOAD: u8  = 0x0;
pub const FUNCT3_DMA_STORE: u8 = 0x1;
pub const FUNCT3_DMA_WAIT: u8  = 0x2;
pub const FUNCT3_DMA_CFG: u8   = 0x3;

// Tensor funct3 codes
pub const FUNCT3_TMATMUL: u8  = 0x0;
pub const FUNCT3_TZERO: u8    = 0x1;
pub const FUNCT3_TACC_ROW: u8 = 0x2;

// Vector funct3 codes
pub const FUNCT3_VGELU: u8      = 0x0;
pub const FUNCT3_VRMSNORM: u8   = 0x1;
pub const FUNCT3_VSOFTMAX: u8   = 0x2;
pub const FUNCT3_VADD: u8       = 0x3;
pub const FUNCT3_VMUL: u8       = 0x4;
pub const FUNCT3_VROW_LOAD: u8  = 0x5;
pub const FUNCT3_VROW_STORE: u8 = 0x6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedInstruction {
    // Scalar ALU
    Add { rd: usize, rs1: usize, rs2: usize },
    Sub { rd: usize, rs1: usize, rs2: usize },
    And { rd: usize, rs1: usize, rs2: usize },
    Or  { rd: usize, rs1: usize, rs2: usize },
    Xor { rd: usize, rs1: usize, rs2: usize },
    Sll { rd: usize, rs1: usize, rs2: usize },
    Srl { rd: usize, rs1: usize, rs2: usize },
    Addi { rd: usize, rs1: usize, imm: i32 },

    // Memory (TCM / Scratchpad)
    Lw { rd: usize, rs1: usize, imm: i32 },
    Sw { rs1: usize, rs2: usize, imm: i32 },

    // Control Flow
    Beq { rs1: usize, rs2: usize, offset: i32 },
    Bne { rs1: usize, rs2: usize, offset: i32 },
    Jal { rd: usize, offset: i32 },
    Jalr { rd: usize, rs1: usize, offset: i32 },

    // Hardware Loop Controller (ZOHL)
    Loop { count_reg: usize, num_instructions: u16 },

    // Tensor DMA Streamer
    TLoad { td_idx: usize, dram_addr_reg: usize, buf_id: u8 },
    TStore { td_idx: usize, dram_addr_reg: usize, buf_id: u8 },
    TWait { buf_id: u8 },
    TCfgDesc { td_idx: usize, base_reg: usize, stride_reg: usize, shape_reg: usize },

    // Systolic Tensor Engine
    TMatMul { acc_idx: usize, td_a: usize, td_b: usize, dtype: DataType },
    TZero { acc_idx: usize },
    TAccRow { vd: usize, acc_idx: usize, row_idx: usize },

    // Vector Engine
    VGelu { vd: usize, vs: usize },
    VRmsNorm { vd: usize, vs: usize, v_weight: usize },
    VSoftmax { vd: usize, vs: usize },
    VAdd { vd: usize, vs1: usize, vs2: usize },
    VMul { vd: usize, vs1: usize, vs2: usize },
    VRowLoad { vd: usize, td_idx: usize, row_idx: usize },
    VRowStore { vs: usize, td_idx: usize, row_idx: usize },

    // System
    Halt,
    Nop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    InvalidOpcode(u8),
    UnknownFunct3 { opcode: u8, funct3: u8 },
    UnknownDataType(u8),
}

pub fn decode(inst: u32) -> Result<DecodedInstruction, DecodeError> {
    let opcode = (inst & 0x7F) as u8;
    let rd = ((inst >> 7) & 0x1F) as usize;
    let funct3 = ((inst >> 12) & 0x07) as u8;
    let rs1 = ((inst >> 15) & 0x1F) as usize;
    let rs2 = ((inst >> 20) & 0x1F) as usize;
    let funct7 = ((inst >> 25) & 0x7F) as u8;

    match opcode {
        OP_IMM => {
            let imm = (inst as i32) >> 20; // sign-extended 12-bit
            match funct3 {
                FUNCT3_ADD_SUB => Ok(DecodedInstruction::Addi { rd, rs1, imm }),
                _ => Err(DecodeError::UnknownFunct3 { opcode, funct3 }),
            }
        }
        OP_ALU => {
            match (funct3, funct7) {
                (FUNCT3_ADD_SUB, 0x00) => Ok(DecodedInstruction::Add { rd, rs1, rs2 }),
                (FUNCT3_ADD_SUB, 0x20) => Ok(DecodedInstruction::Sub { rd, rs1, rs2 }),
                (FUNCT3_AND, 0x00)     => Ok(DecodedInstruction::And { rd, rs1, rs2 }),
                (FUNCT3_OR, 0x00)      => Ok(DecodedInstruction::Or  { rd, rs1, rs2 }),
                (FUNCT3_XOR, 0x00)     => Ok(DecodedInstruction::Xor { rd, rs1, rs2 }),
                (FUNCT3_SLL, 0x00)     => Ok(DecodedInstruction::Sll { rd, rs1, rs2 }),
                (FUNCT3_SRL_SRA, 0x00) => Ok(DecodedInstruction::Srl { rd, rs1, rs2 }),
                _ => Err(DecodeError::UnknownFunct3 { opcode, funct3 }),
            }
        }
        OP_LOAD => {
            let imm = (inst as i32) >> 20;
            Ok(DecodedInstruction::Lw { rd, rs1, imm })
        }
        OP_STORE => {
            let imm = (((inst >> 25) as i32) << 5) | (((inst >> 7) & 0x1F) as i32);
            let sign_ext_imm = (imm << 20) >> 20;
            Ok(DecodedInstruction::Sw { rs1, rs2, imm: sign_ext_imm })
        }
        OP_LOOP => {
            let count_reg = rs1;
            let num_instructions = (inst >> 20) as u16;
            Ok(DecodedInstruction::Loop { count_reg, num_instructions })
        }
        OP_DMA => {
            let td_idx = rs2 & 0x07;
            let dram_addr_reg = rs1;
            let buf_id = (rd & 0x01) as u8;
            match funct3 {
                FUNCT3_DMA_LOAD => Ok(DecodedInstruction::TLoad { td_idx, dram_addr_reg, buf_id }),
                FUNCT3_DMA_STORE => Ok(DecodedInstruction::TStore { td_idx, dram_addr_reg, buf_id }),
                FUNCT3_DMA_WAIT => Ok(DecodedInstruction::TWait { buf_id }),
                FUNCT3_DMA_CFG => {
                    let base_reg = rs1;
                    let stride_reg = rs2;
                    let shape_reg = (inst >> 25) as usize & 0x1F;
                    Ok(DecodedInstruction::TCfgDesc { td_idx, base_reg, stride_reg, shape_reg })
                }
                _ => Err(DecodeError::UnknownFunct3 { opcode, funct3 }),
            }
        }
        OP_TENSOR => {
            let acc_idx = rd & 0x03;
            let td_a = rs1 & 0x07;
            let td_b = rs2 & 0x07;
            match funct3 {
                FUNCT3_TMATMUL => {
                    let dtype_raw = (inst >> 25) as u8 & 0x07;
                    let dtype = DataType::from_u8(dtype_raw)
                        .ok_or(DecodeError::UnknownDataType(dtype_raw))?;
                    Ok(DecodedInstruction::TMatMul { acc_idx, td_a, td_b, dtype })
                }
                FUNCT3_TZERO => Ok(DecodedInstruction::TZero { acc_idx }),
                FUNCT3_TACC_ROW => {
                    let vd = rd & 0x07;
                    let row_idx = rs1 & 0x0F;
                    let acc_idx = rs2 & 0x03;
                    Ok(DecodedInstruction::TAccRow { vd, acc_idx, row_idx })
                }
                _ => Err(DecodeError::UnknownFunct3 { opcode, funct3 }),
            }
        }
        OP_VECTOR => {
            let vd = rd & 0x07;
            let vs1 = rs1 & 0x07;
            let vs2 = rs2 & 0x07;
            match funct3 {
                FUNCT3_VGELU => Ok(DecodedInstruction::VGelu { vd, vs: vs1 }),
                FUNCT3_VRMSNORM => Ok(DecodedInstruction::VRmsNorm { vd, vs: vs1, v_weight: vs2 }),
                FUNCT3_VSOFTMAX => Ok(DecodedInstruction::VSoftmax { vd, vs: vs1 }),
                FUNCT3_VADD => Ok(DecodedInstruction::VAdd { vd, vs1, vs2 }),
                FUNCT3_VMUL => Ok(DecodedInstruction::VMul { vd, vs1, vs2 }),
                FUNCT3_VROW_LOAD => {
                    let td_idx = rs1 & 0x07;
                    let row_idx = rs2 & 0x0F;
                    Ok(DecodedInstruction::VRowLoad { vd, td_idx, row_idx })
                }
                FUNCT3_VROW_STORE => {
                    let vs = vd;
                    let td_idx = rs1 & 0x07;
                    let row_idx = rs2 & 0x0F;
                    Ok(DecodedInstruction::VRowStore { vs, td_idx, row_idx })
                }
                _ => Err(DecodeError::UnknownFunct3 { opcode, funct3 }),
            }
        }
        OP_SYSTEM => {
            if funct3 == 0 && funct7 == 0 {
                Ok(DecodedInstruction::Nop)
            } else if funct3 == 0 && funct7 == 1 {
                Ok(DecodedInstruction::Halt)
            } else {
                Ok(DecodedInstruction::Nop)
            }
        }
        _ => Err(DecodeError::InvalidOpcode(opcode)),
    }
}

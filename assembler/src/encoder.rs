/// Two-Pass Assembler & Instruction Encoder for NeuroFlow

use std::collections::HashMap;
use crate::parser::{Operand, ParsedInstruction, Statement};

// Opcodes
const OP_ALU: u32    = 0x33;
const OP_IMM: u32    = 0x13;
const OP_LOAD: u32   = 0x03;
const OP_STORE: u32  = 0x23;
const OP_BRANCH: u32 = 0x63;
const OP_JUMP: u32   = 0x6F;
const OP_LOOP: u32   = 0x5B;
const OP_DMA: u32    = 0x0B;
const OP_TENSOR: u32 = 0x2B;
const OP_VECTOR: u32 = 0x4B;
const OP_SYSTEM: u32 = 0x73;

#[derive(Debug, Clone)]
pub struct EncodeError {
    pub line: usize,
    pub message: String,
}

pub struct Assembler {
    labels: HashMap<String, u32>,
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
        }
    }

    pub fn assemble(&mut self, statements: &[Statement]) -> Result<Vec<u32>, EncodeError> {
        self.labels.clear();

        // Pass 1: Collect labels and calculate instruction addresses
        let mut pc = 0u32;
        for stmt in statements {
            match stmt {
                Statement::Label(name) => {
                    if self.labels.contains_key(name) {
                        return Err(EncodeError {
                            line: 0,
                            message: format!("Duplicate label definition: '{}'", name),
                        });
                    }
                    self.labels.insert(name.clone(), pc);
                }
                Statement::Instruction(_) => {
                    pc += 4;
                }
                Statement::Directive { name, args: _ } => {
                    if name == ".word" {
                        pc += 4;
                    }
                }
            }
        }

        // Pass 2: Encode instructions into 32-bit machine code
        let mut machine_code = Vec::new();
        pc = 0;

        for stmt in statements {
            match stmt {
                Statement::Label(_) => {}
                Statement::Directive { name, args } => {
                    if name == ".word" {
                        if let Some(Operand::Imm(val)) = args.first() {
                            machine_code.push(*val as u32);
                            pc += 4;
                        }
                    }
                }
                Statement::Instruction(inst) => {
                    let encoded = self.encode_instruction(inst, pc)?;
                    machine_code.push(encoded);
                    pc += 4;
                }
            }
        }

        Ok(machine_code)
    }

    fn encode_instruction(&self, inst: &ParsedInstruction, pc: u32) -> Result<u32, EncodeError> {
        let m = inst.mnemonic.as_str();

        match m {
            // Scalar Immediate
            "addi" => {
                let rd = self.get_scalar_reg(inst, 0)?;
                let rs1 = self.get_scalar_reg(inst, 1)?;
                let imm = self.get_imm(inst, 2, pc)? as i32;
                Ok(OP_IMM | (rd << 7) | (0 << 12) | (rs1 << 15) | (((imm as u32) & 0xFFF) << 20))
            }

            // Scalar ALU
            "add" => self.encode_r_type(inst, OP_ALU, 0x0, 0x00),
            "sub" => self.encode_r_type(inst, OP_ALU, 0x0, 0x20),
            "and" => self.encode_r_type(inst, OP_ALU, 0x7, 0x00),
            "or"  => self.encode_r_type(inst, OP_ALU, 0x6, 0x00),
            "xor" => self.encode_r_type(inst, OP_ALU, 0x4, 0x00),
            "sll" => self.encode_r_type(inst, OP_ALU, 0x1, 0x00),
            "srl" => self.encode_r_type(inst, OP_ALU, 0x5, 0x00),

            // Memory
            "lw" => {
                let rd = self.get_scalar_reg(inst, 0)?;
                let rs1 = self.get_scalar_reg(inst, 1)?;
                let imm = if inst.operands.len() > 2 { self.get_imm(inst, 2, pc)? as i32 } else { 0 };
                Ok(OP_LOAD | (rd << 7) | (0x2 << 12) | (rs1 << 15) | (((imm as u32) & 0xFFF) << 20))
            }
            "sw" => {
                let rs1 = self.get_scalar_reg(inst, 0)?;
                let rs2 = self.get_scalar_reg(inst, 1)?;
                let imm = if inst.operands.len() > 2 { self.get_imm(inst, 2, pc)? as i32 } else { 0 };
                let imm_u = (imm as u32) & 0xFFF;
                let imm_low = imm_u & 0x1F;
                let imm_high = (imm_u >> 5) & 0x7F;
                Ok(OP_STORE | (imm_low << 7) | (0x2 << 12) | (rs1 << 15) | (rs2 << 20) | (imm_high << 25))
            }

            // Control Flow
            "beq" => self.encode_branch(inst, pc, 0x0),
            "bne" => self.encode_branch(inst, pc, 0x1),
            "jal" => {
                let rd = self.get_scalar_reg(inst, 0)?;
                let target = self.get_label_addr(inst, 1)?;
                let offset = (target as i32) - (pc as i32);
                let imm = (offset as u32) & 0xFFFFF;
                Ok(OP_JUMP | (rd << 7) | (imm << 12))
            }
            "j" => {
                let target = self.get_label_addr(inst, 0)?;
                let offset = (target as i32) - (pc as i32);
                let imm = (offset as u32) & 0xFFFFF;
                Ok(OP_JUMP | (0 << 7) | (imm << 12)) // rd = x0
            }

            // Hardware Loop (ZOHL)
            "loop" => {
                let count_reg = self.get_scalar_reg(inst, 0)?;
                let num_insts = self.get_imm(inst, 1, pc)? as u32;
                Ok(OP_LOOP | (count_reg << 15) | ((num_insts & 0xFFF) << 20))
            }

            // Tensor DMA
            "tload" => {
                let td = self.get_td_reg(inst, 0)?;
                let dram_reg = self.get_scalar_reg(inst, 1)?;
                let buf_id = self.get_imm(inst, 2, pc)? as u32 & 0x1;
                Ok(OP_DMA | (buf_id << 7) | (0x0 << 12) | (dram_reg << 15) | (td << 20))
            }
            "tstore" => {
                let td = self.get_td_reg(inst, 0)?;
                let dram_reg = self.get_scalar_reg(inst, 1)?;
                let buf_id = self.get_imm(inst, 2, pc)? as u32 & 0x1;
                Ok(OP_DMA | (buf_id << 7) | (0x1 << 12) | (dram_reg << 15) | (td << 20))
            }
            "twait" => {
                let buf_id = self.get_imm(inst, 0, pc)? as u32 & 0x1;
                Ok(OP_DMA | (buf_id << 7) | (0x2 << 12))
            }
            "tcfg.desc" => {
                let td = self.get_td_reg(inst, 0)?;
                let base_reg = self.get_scalar_reg(inst, 1)?;
                let stride_reg = self.get_scalar_reg(inst, 2)?;
                let shape_reg = self.get_scalar_reg(inst, 3)?;
                Ok(OP_DMA | (td << 7) | (0x3 << 12) | (base_reg << 15) | (stride_reg << 20) | (shape_reg << 25))
            }

            // Systolic Tensor Operations
            "tzero" => {
                let acc = self.get_acc_reg(inst, 0)?;
                Ok(OP_TENSOR | (acc << 7) | (0x1 << 12))
            }
            "tmatmul" => {
                let acc = self.get_acc_reg(inst, 0)?;
                let td_a = self.get_td_reg(inst, 1)?;
                let td_b = self.get_td_reg(inst, 2)?;
                let dtype = self.get_dtype(inst, 3)?;
                Ok(OP_TENSOR | (acc << 7) | (0x0 << 12) | (td_a << 15) | (td_b << 20) | (dtype << 25))
            }
            "tacc.row" => {
                let vd = self.get_vec_reg(inst, 0)?;
                let acc = self.get_acc_reg(inst, 1)?;
                let row_idx = self.get_imm(inst, 2, pc)? as u32 & 0x0F;
                Ok(OP_TENSOR | (vd << 7) | (0x2 << 12) | (row_idx << 15) | (acc << 20))
            }

            // Vector Engine
            "vgelu" => {
                let vd = self.get_vec_reg(inst, 0)?;
                let vs = self.get_vec_reg(inst, 1)?;
                Ok(OP_VECTOR | (vd << 7) | (0x0 << 12) | (vs << 15))
            }
            "vrmsnorm" => {
                let vd = self.get_vec_reg(inst, 0)?;
                let vs = self.get_vec_reg(inst, 1)?;
                let v_weight = self.get_vec_reg(inst, 2)?;
                Ok(OP_VECTOR | (vd << 7) | (0x1 << 12) | (vs << 15) | (v_weight << 20))
            }
            "vsoftmax" => {
                let vd = self.get_vec_reg(inst, 0)?;
                let vs = self.get_vec_reg(inst, 1)?;
                Ok(OP_VECTOR | (vd << 7) | (0x2 << 12) | (vs << 15))
            }
            "vadd" => {
                let vd = self.get_vec_reg(inst, 0)?;
                let vs1 = self.get_vec_reg(inst, 1)?;
                let vs2 = self.get_vec_reg(inst, 2)?;
                Ok(OP_VECTOR | (vd << 7) | (0x3 << 12) | (vs1 << 15) | (vs2 << 20))
            }
            "vmul" => {
                let vd = self.get_vec_reg(inst, 0)?;
                let vs1 = self.get_vec_reg(inst, 1)?;
                let vs2 = self.get_vec_reg(inst, 2)?;
                Ok(OP_VECTOR | (vd << 7) | (0x4 << 12) | (vs1 << 15) | (vs2 << 20))
            }

            // System
            "halt" => Ok(OP_SYSTEM | (0x1 << 25)),
            "nop" => Ok(OP_SYSTEM),

            _ => Err(EncodeError {
                line: inst.line,
                message: format!("Unknown mnemonic: '{}'", m),
            }),
        }
    }

    fn encode_r_type(&self, inst: &ParsedInstruction, opcode: u32, funct3: u32, funct7: u32) -> Result<u32, EncodeError> {
        let rd = self.get_scalar_reg(inst, 0)?;
        let rs1 = self.get_scalar_reg(inst, 1)?;
        let rs2 = self.get_scalar_reg(inst, 2)?;
        Ok(opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25))
    }

    fn encode_branch(&self, inst: &ParsedInstruction, pc: u32, funct3: u32) -> Result<u32, EncodeError> {
        let rs1 = self.get_scalar_reg(inst, 0)?;
        let rs2 = self.get_scalar_reg(inst, 1)?;
        let target = self.get_label_addr(inst, 2)?;
        let offset = (target as i32) - (pc as i32);
        let imm = (offset as u32) & 0xFFF;
        let imm_low = (imm >> 1) & 0x1F;
        let imm_high = (imm >> 6) & 0x7F;
        Ok(OP_BRANCH | (imm_low << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (imm_high << 25))
    }

    fn get_scalar_reg(&self, inst: &ParsedInstruction, idx: usize) -> Result<u32, EncodeError> {
        match inst.operands.get(idx) {
            Some(Operand::ScalarReg(r)) => Ok(*r as u32),
            _ => Err(EncodeError {
                line: inst.line,
                message: format!("Operand {} must be a scalar register (x0-x31)", idx + 1),
            }),
        }
    }

    fn get_vec_reg(&self, inst: &ParsedInstruction, idx: usize) -> Result<u32, EncodeError> {
        match inst.operands.get(idx) {
            Some(Operand::VectorReg(v)) => Ok(*v as u32),
            _ => Err(EncodeError {
                line: inst.line,
                message: format!("Operand {} must be a vector register (v0-v7)", idx + 1),
            }),
        }
    }

    fn get_td_reg(&self, inst: &ParsedInstruction, idx: usize) -> Result<u32, EncodeError> {
        match inst.operands.get(idx) {
            Some(Operand::TileDescReg(td)) => Ok(*td as u32),
            _ => Err(EncodeError {
                line: inst.line,
                message: format!("Operand {} must be a tile descriptor register (td0-td7)", idx + 1),
            }),
        }
    }

    fn get_acc_reg(&self, inst: &ParsedInstruction, idx: usize) -> Result<u32, EncodeError> {
        match inst.operands.get(idx) {
            Some(Operand::AccReg(acc)) => Ok(*acc as u32),
            _ => Err(EncodeError {
                line: inst.line,
                message: format!("Operand {} must be an accumulator register (acc0-acc3)", idx + 1),
            }),
        }
    }

    fn get_imm(&self, inst: &ParsedInstruction, idx: usize, _pc: u32) -> Result<i64, EncodeError> {
        match inst.operands.get(idx) {
            Some(Operand::Imm(n)) => Ok(*n),
            _ => Err(EncodeError {
                line: inst.line,
                message: format!("Operand {} must be an immediate number", idx + 1),
            }),
        }
    }

    fn get_label_addr(&self, inst: &ParsedInstruction, idx: usize) -> Result<u32, EncodeError> {
        match inst.operands.get(idx) {
            Some(Operand::Label(lbl)) => {
                self.labels.get(lbl).copied().ok_or(EncodeError {
                    line: inst.line,
                    message: format!("Undefined label: '{}'", lbl),
                })
            }
            Some(Operand::Imm(n)) => Ok(*n as u32),
            _ => Err(EncodeError {
                line: inst.line,
                message: format!("Operand {} must be a label or address", idx + 1),
            }),
        }
    }

    fn get_dtype(&self, inst: &ParsedInstruction, idx: usize) -> Result<u32, EncodeError> {
        match inst.operands.get(idx) {
            Some(Operand::Label(name)) => {
                match name.to_lowercase().as_str() {
                    "fp8_e4m3" | "fp8" => Ok(0),
                    "fp8_e5m2" => Ok(1),
                    "bf16" => Ok(2),
                    "int8" => Ok(3),
                    "fp32" => Ok(4),
                    _ => Err(EncodeError {
                        line: inst.line,
                        message: format!("Invalid precision datatype: '{}'", name),
                    }),
                }
            }
            Some(Operand::Imm(n)) => Ok(*n as u32 & 0x07),
            _ => Err(EncodeError {
                line: inst.line,
                message: format!("Operand {} must be a data type (bf16, fp8, int8)", idx + 1),
            }),
        }
    }
}

/// NeuroFlow Systolic Array Tensor Engine (TE)
/// - Simulates a 16x16 2D Systolic Array with Output-Stationary accumulation.
/// - Operates on low-precision inputs (BF16, FP8-E4M3, INT8) and accumulates in FP32 / INT32.

use crate::memory::MemorySubsystem;
use crate::types::{Bf16, DataType, Fp8E4M3};

pub const TILE_DIM: usize = 16; // 16x16 Processing Elements

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileDescriptor {
    pub base_addr: u32,
    pub rows: u16,
    pub cols: u16,
    pub stride: u32,
    pub dtype: u8,
    pub buffer_id: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct AccumulatorTile {
    pub fp32_data: [[f32; TILE_DIM]; TILE_DIM],
    pub int32_data: [[i32; TILE_DIM]; TILE_DIM],
}

impl Default for AccumulatorTile {
    fn default() -> Self {
        Self {
            fp32_data: [[0.0f32; TILE_DIM]; TILE_DIM],
            int32_data: [[0i32; TILE_DIM]; TILE_DIM],
        }
    }
}

impl AccumulatorTile {
    pub fn zero(&mut self) {
        self.fp32_data = [[0.0f32; TILE_DIM]; TILE_DIM];
        self.int32_data = [[0i32; TILE_DIM]; TILE_DIM];
    }
}

#[derive(Debug, Default)]
pub struct SystolicEngine {
    pub accumulators: [AccumulatorTile; 4],
}

impl SystolicEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Multiply tile A by tile B from TCM and accumulate into accumulator `acc_idx`.
    /// Returns the number of simulated clock cycles spent (e.g. 16+16+16 = 48 cycles).
    pub fn matmul_tile(
        &mut self,
        acc_idx: usize,
        desc_a: &TileDescriptor,
        desc_b: &TileDescriptor,
        dtype: DataType,
        memory: &MemorySubsystem,
    ) -> Result<u64, &'static str> {
        let acc = self.accumulators.get_mut(acc_idx).ok_or("Invalid accumulator index")?;

        match dtype {
            DataType::Bf16 => {
                for i in 0..TILE_DIM {
                    for j in 0..TILE_DIM {
                        let mut sum = 0.0f32;
                        for k in 0..TILE_DIM {
                            let offset_a = (desc_a.base_addr as usize)
                                + i * (desc_a.stride as usize)
                                + k * 2;
                            let offset_b = (desc_b.base_addr as usize)
                                + k * (desc_b.stride as usize)
                                + j * 2;

                            let bytes_a = memory.read_tcm_bytes(desc_a.buffer_id, offset_a, 2)
                                .map_err(|_| "TCM Out of Bounds reading tile A")?;
                            let bytes_b = memory.read_tcm_bytes(desc_b.buffer_id, offset_b, 2)
                                .map_err(|_| "TCM Out of Bounds reading tile B")?;

                            let a_val = Bf16(u16::from_le_bytes([bytes_a[0], bytes_a[1]])).to_f32();
                            let b_val = Bf16(u16::from_le_bytes([bytes_b[0], bytes_b[1]])).to_f32();

                            sum += a_val * b_val;
                        }
                        acc.fp32_data[i][j] += sum;
                    }
                }
            }
            DataType::Fp8E4M3 => {
                for i in 0..TILE_DIM {
                    for j in 0..TILE_DIM {
                        let mut sum = 0.0f32;
                        for k in 0..TILE_DIM {
                            let offset_a = (desc_a.base_addr as usize)
                                + i * (desc_a.stride as usize)
                                + k;
                            let offset_b = (desc_b.base_addr as usize)
                                + k * (desc_b.stride as usize)
                                + j;

                            let bytes_a = memory.read_tcm_bytes(desc_a.buffer_id, offset_a, 1)
                                .map_err(|_| "TCM Out of Bounds reading tile A")?;
                            let bytes_b = memory.read_tcm_bytes(desc_b.buffer_id, offset_b, 1)
                                .map_err(|_| "TCM Out of Bounds reading tile B")?;

                            let a_val = Fp8E4M3(bytes_a[0]).to_f32();
                            let b_val = Fp8E4M3(bytes_b[0]).to_f32();

                            sum += a_val * b_val;
                        }
                        acc.fp32_data[i][j] += sum;
                    }
                }
            }
            DataType::Int8 => {
                for i in 0..TILE_DIM {
                    for j in 0..TILE_DIM {
                        let mut sum = 0i32;
                        for k in 0..TILE_DIM {
                            let offset_a = (desc_a.base_addr as usize)
                                + i * (desc_a.stride as usize)
                                + k;
                            let offset_b = (desc_b.base_addr as usize)
                                + k * (desc_b.stride as usize)
                                + j;

                            let bytes_a = memory.read_tcm_bytes(desc_a.buffer_id, offset_a, 1)
                                .map_err(|_| "TCM Out of Bounds reading tile A")?;
                            let bytes_b = memory.read_tcm_bytes(desc_b.buffer_id, offset_b, 1)
                                .map_err(|_| "TCM Out of Bounds reading tile B")?;

                            let a_val = bytes_a[0] as i8 as i32;
                            let b_val = bytes_b[0] as i8 as i32;

                            sum += a_val * b_val;
                        }
                        acc.int32_data[i][j] += sum;
                    }
                }
            }
            DataType::Fp32 => {
                for i in 0..TILE_DIM {
                    for j in 0..TILE_DIM {
                        let mut sum = 0.0f32;
                        for k in 0..TILE_DIM {
                            let offset_a = (desc_a.base_addr as usize)
                                + i * (desc_a.stride as usize)
                                + k * 4;
                            let offset_b = (desc_b.base_addr as usize)
                                + k * (desc_b.stride as usize)
                                + j * 4;

                            let bytes_a = memory.read_tcm_bytes(desc_a.buffer_id, offset_a, 4)
                                .map_err(|_| "TCM Out of Bounds reading tile A")?;
                            let bytes_b = memory.read_tcm_bytes(desc_b.buffer_id, offset_b, 4)
                                .map_err(|_| "TCM Out of Bounds reading tile B")?;

                            let a_val = f32::from_le_bytes(bytes_a.try_into().unwrap());
                            let b_val = f32::from_le_bytes(bytes_b.try_into().unwrap());

                            sum += a_val * b_val;
                        }
                        acc.fp32_data[i][j] += sum;
                    }
                }
            }
            _ => return Err("Unsupported GEMM data type"),
        }

        // A 16x16 2D systolic array with skewing and draining takes ~48 cycles
        Ok(48)
    }
}

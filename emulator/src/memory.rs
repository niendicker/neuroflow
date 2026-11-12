/// NeuroFlow Memory Subsystem:
/// 1. Tightly-Coupled Scratchpad SRAM (TCM):
///    - Double-buffered ping-pong architecture (Bank A and Bank B).
///    - 100% deterministic, zero wait-state access for Tensor and Vector engines.
/// 2. External Memory (DRAM):
///    - Modeled external memory space accessed exclusively through the Tensor DMA Streamer (TDS).

pub const SCRATCHPAD_BANK_SIZE: usize = 256 * 1024; // 256 KB per bank (512 KB total on-chip TCM)
pub const DRAM_SIZE: usize = 16 * 1024 * 1024;       // 16 MB simulated external DRAM

#[derive(Debug)]
pub struct MemorySubsystem {
    pub tcm_bank_a: Box<[u8]>,
    pub tcm_bank_b: Box<[u8]>,
    pub dram: Box<[u8]>,
}

impl Default for MemorySubsystem {
    fn default() -> Self {
        Self::new()
    }
}

impl MemorySubsystem {
    pub fn new() -> Self {
        Self {
            tcm_bank_a: vec![0u8; SCRATCHPAD_BANK_SIZE].into_boxed_slice(),
            tcm_bank_b: vec![0u8; SCRATCHPAD_BANK_SIZE].into_boxed_slice(),
            dram: vec![0u8; DRAM_SIZE].into_boxed_slice(),
        }
    }

    /// Read a 32-bit word from a specific Scratchpad bank (little-endian)
    pub fn read_tcm_u32(&self, bank_id: u8, offset: usize) -> Result<u32, MemoryError> {
        let bank = if bank_id == 0 {
            &self.tcm_bank_a[..]
        } else {
            &self.tcm_bank_b[..]
        };

        if offset + 4 > SCRATCHPAD_BANK_SIZE {
            return Err(MemoryError::OutOfBounds {
                target: "TCM",
                offset,
                size: 4,
                capacity: SCRATCHPAD_BANK_SIZE,
            });
        }

        let slice = &bank[offset..offset + 4];
        Ok(u32::from_le_bytes(slice.try_into().unwrap()))
    }

    /// Write a 32-bit word to a specific Scratchpad bank (little-endian)
    pub fn write_tcm_u32(&mut self, bank_id: u8, offset: usize, val: u32) -> Result<(), MemoryError> {
        let bank = if bank_id == 0 {
            &mut self.tcm_bank_a[..]
        } else {
            &mut self.tcm_bank_b[..]
        };

        if offset + 4 > SCRATCHPAD_BANK_SIZE {
            return Err(MemoryError::OutOfBounds {
                target: "TCM",
                offset,
                size: 4,
                capacity: SCRATCHPAD_BANK_SIZE,
            });
        }

        bank[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        Ok(())
    }

    /// Read a slice of bytes from TCM bank
    pub fn read_tcm_bytes(&self, bank_id: u8, offset: usize, len: usize) -> Result<&[u8], MemoryError> {
        let bank = if bank_id == 0 {
            &self.tcm_bank_a[..]
        } else {
            &self.tcm_bank_b[..]
        };

        if offset + len > SCRATCHPAD_BANK_SIZE {
            return Err(MemoryError::OutOfBounds {
                target: "TCM",
                offset,
                size: len,
                capacity: SCRATCHPAD_BANK_SIZE,
            });
        }

        Ok(&bank[offset..offset + len])
    }

    /// Write bytes into TCM bank
    pub fn write_tcm_bytes(&mut self, bank_id: u8, offset: usize, data: &[u8]) -> Result<(), MemoryError> {
        let bank = if bank_id == 0 {
            &mut self.tcm_bank_a[..]
        } else {
            &mut self.tcm_bank_b[..]
        };

        if offset + data.len() > SCRATCHPAD_BANK_SIZE {
            return Err(MemoryError::OutOfBounds {
                target: "TCM",
                offset,
                size: data.len(),
                capacity: SCRATCHPAD_BANK_SIZE,
            });
        }

        bank[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    OutOfBounds {
        target: &'static str,
        offset: usize,
        size: usize,
        capacity: usize,
    },
    MisalignedAccess {
        offset: usize,
        alignment: usize,
    },
}

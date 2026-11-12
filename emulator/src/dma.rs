/// NeuroFlow Tensor DMA Streamer (TDS)
/// - Asynchronously streams 2D/3D tensor tiles between DRAM and Scratchpad TCM.
/// - Supports double-buffering (Bank A = 0, Bank B = 1) to hide DRAM latency.

use crate::memory::{MemorySubsystem, DRAM_SIZE, SCRATCHPAD_BANK_SIZE};
use crate::tensor_engine::TileDescriptor;

#[derive(Debug, Clone, Copy, Default)]
pub struct DmaChannel {
    pub busy: bool,
    pub cycles_remaining: u64,
}

#[derive(Debug, Default)]
pub struct TensorDmaStreamer {
    pub channels: [DmaChannel; 2], // Channel 0 for Bank A, Channel 1 for Bank B
}

impl TensorDmaStreamer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance DMA timers by given clock cycles
    pub fn tick(&mut self, cycles: u64) {
        for chan in &mut self.channels {
            if chan.busy {
                if cycles >= chan.cycles_remaining {
                    chan.busy = false;
                    chan.cycles_remaining = 0;
                } else {
                    chan.cycles_remaining -= cycles;
                }
            }
        }
    }

    /// Asynchronous TLOAD: schedules a 2D tile copy from DRAM to TCM bank
    pub fn start_load(
        &mut self,
        desc: &TileDescriptor,
        dram_addr: usize,
        buf_id: u8,
        memory: &mut MemorySubsystem,
    ) -> Result<(), &'static str> {
        let channel = buf_id as usize & 0x01;
        let bytes_per_element = match desc.dtype {
            0 | 1 | 3 => 1, // FP8, INT8
            2 => 2,         // BF16
            4 | 5 => 4,     // FP32, INT32
            _ => 2,
        };

        let row_bytes = (desc.cols as usize) * bytes_per_element;
        let total_bytes = (desc.rows as usize) * row_bytes;

        if dram_addr + total_bytes > DRAM_SIZE {
            return Err("DRAM read address out of bounds");
        }
        if (desc.base_addr as usize) + total_bytes > SCRATCHPAD_BANK_SIZE {
            return Err("TCM write address out of bounds");
        }

        // Perform copy with a small reusable buffer to satisfy borrow rules
        let mut row_buf = vec![0u8; row_bytes];
        for r in 0..(desc.rows as usize) {
            let dram_offset = dram_addr + r * row_bytes;
            let tcm_offset = (desc.base_addr as usize) + r * (desc.stride as usize);

            row_buf.copy_from_slice(&memory.dram[dram_offset..dram_offset + row_bytes]);
            memory.write_tcm_bytes(buf_id, tcm_offset, &row_buf)
                .map_err(|_| "Failed to write into TCM")?;
        }

        // Latency model: 200 GB/s bandwidth assumption (approx 1 cycle per 64 bytes + 40 cycle overhead)
        let simulated_latency = 40 + (total_bytes as u64 / 64).max(1);

        self.channels[channel].busy = true;
        self.channels[channel].cycles_remaining = simulated_latency;
        Ok(())
    }

    /// Asynchronous TSTORE: schedules a 2D tile copy from TCM bank to DRAM
    pub fn start_store(
        &mut self,
        desc: &TileDescriptor,
        dram_addr: usize,
        buf_id: u8,
        memory: &mut MemorySubsystem,
    ) -> Result<(), &'static str> {
        let channel = buf_id as usize & 0x01;
        let bytes_per_element = match desc.dtype {
            0 | 1 | 3 => 1,
            2 => 2,
            4 | 5 => 4,
            _ => 2,
        };

        let row_bytes = (desc.cols as usize) * bytes_per_element;
        let total_bytes = (desc.rows as usize) * row_bytes;

        if dram_addr + total_bytes > DRAM_SIZE {
            return Err("DRAM write address out of bounds");
        }

        let mut row_buf = vec![0u8; row_bytes];
        for r in 0..(desc.rows as usize) {
            let tcm_offset = (desc.base_addr as usize) + r * (desc.stride as usize);
            let dram_offset = dram_addr + r * row_bytes;

            let slice = memory.read_tcm_bytes(buf_id, tcm_offset, row_bytes)
                .map_err(|_| "Failed to read from TCM")?;
            row_buf.copy_from_slice(slice);
            memory.dram[dram_offset..dram_offset + row_bytes].copy_from_slice(&row_buf);
        }

        let simulated_latency = 40 + (total_bytes as u64 / 64).max(1);
        self.channels[channel].busy = true;
        self.channels[channel].cycles_remaining = simulated_latency;
        Ok(())
    }

    /// TWAIT: Blocks if channel is busy, returning the number of cycles waited
    pub fn wait_sync(&mut self, buf_id: u8) -> u64 {
        let channel = buf_id as usize & 0x01;
        if self.channels[channel].busy {
            let wait_cycles = self.channels[channel].cycles_remaining;
            self.channels[channel].busy = false;
            self.channels[channel].cycles_remaining = 0;
            // Also tick other channels
            self.tick(wait_cycles);
            wait_cycles
        } else {
            0 // Latency was completely hidden!
        }
    }
}

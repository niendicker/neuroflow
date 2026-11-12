# NeuroFlow ISA Specification (v0.1)

---

## 1. Core Architectural Tenets & Design Principles

All future development, extensions, and microarchitectural choices in NeuroFlow **must** strictly adhere to the following five fundamental tenets. Any proposed feature that violates these principles must be rejected:

### Tenet I: The Compute-to-Energy Symmetry Principle (Physics First)
* *The Rule:* In deep learning hardware, moving 1 byte of data across silicon or off-chip DRAM consumes 10× to 100× more energy than performing a floating-point Multiply-Accumulate (MAC) operation.
* *The Mandate:* Never introduce intermediate memory roundtrips. Intermediate partial sums must remain stationary in local accumulator registers (*Output-Stationary Systolic Array*). Data movement must be minimized at all costs.

### Tenet II: Radical Determinism Over Speculative Hardware
* *The Rule:* In neural networks, tensor shapes, layer dimensions, and iteration bounds are 99.9% deterministic and known prior to kernel dispatch.
* *The Mandate:* Do not burn silicon die area or power budget on speculative out-of-order execution (OoO), branch prediction history tables, or speculative prefetchers. Instead, use zero-overhead hardware loops, deterministic DMA streamers, and software-managed scratchpads.

### Tenet III: Strict Decoupling of Access and Execute (DAE)
* *The Rule:* Compute engines (Systolic Arrays, Vector Units) must never stall waiting on address arithmetic, memory fetches, or DRAM latency.
* *The Mandate:* Memory transfers (Access) run strictly asynchronously from mathematical operations (Execute). The architecture relies on double-buffering (ping-pong buffers) so memory latency is 100% masked behind computation:
  $$\text{Latency} = \max(\text{Compute}_{\text{tile}}, \text{Memory}_{\text{tile+1}})$$

### Tenet IV: Dimensional Alignment (Tile & Vector Symmetry)
* *The Rule:* Data widths, vector lanes, and systolic dimensions must be powers of two and physically aligned to avoid data realignment and permutation overhead.
* *The Mandate:* A systolic array tile dimension ($16 \times 16$) in FP32 requires $16 \times 4\text{ bytes} = 64\text{ bytes} = 512\text{ bits}$. Therefore, vector registers are precisely 512 bits wide. One matrix row maps directly to one vector register in a single cycle with zero shuffling.

### Tenet V: Anti-Bloat / Lean 32-bit Encoding
* *The Rule:* Complex CISC instructions create slow, power-hungry instruction decoders and unpredictable pipeline stages.
* *The Mandate:* Every instruction in NeuroFlow is strictly **32 bits wide, fixed-length**, and aligned to 4 bytes. Instructions are clean, composable, and express high-level domain operations (GEMM, activations, DMA) without bloat.

---

## 2. Architectural Decisions & Rationales ("The Why")

| Architectural Decision | Chosen Implementation | Alternative Rejected | Why This Choice? (Technical & Physical Rationale) |
| :--- | :--- | :--- | :--- |
| **Instruction Width** | **Fixed 32-bit** | Variable (16/32-bit) or VLIW (256-bit+) | Eliminates decoder multiplexer cascades and pipeline alignment stalls. Keeps decoder area minimal on FPGA/ASIC. |
| **Scalar Register Count** | **32 registers (`x0`–`x31`)** | 16 registers or 64+ registers | 16 registers cause frequent *register spilling* to RAM when tracking multi-tensor pointers and quant scales. 64 registers consume 6 bits per operand (18 bits total), leaving insufficient space in a 32-bit instruction. 32 registers (5 bits per operand) is optimal. |
| **Loop Control** | **Zero-Overhead Hardware Loops (`LC`, `LS`, `LE`)** | Standard conditional branch (`BNE`/`BEQ`) | Avoids 2 instructions of loop overhead per iteration and branch misprediction penalties. Hardware updates PC and counter in the execution cycle. |
| **Tensor Addressing** | **Tile Descriptors (`td0`–`td7`)** | Raw 32-bit memory pointers | Tensores are 2D/3D matrices with strides. Tile descriptors encode base address, dimensions ($M, K$), stride, and precision in hardware, offloading address math from the scalar ALU. |
| **Accumulator Storage** | **Stationary Tile Buffers (`acc0`–`acc3`)** | Writing partial sums to Scratchpad | Prevents catastrophic precision loss (FP8/BF16 multiplication requires FP32 accumulation). Keeping partial sums stationary inside the array avoids burning memory bandwidth. |
| **Vector Register Width** | **512 bits (`v0`–`v7`)** | 128-bit (NEON) or 256-bit (AVX2) | Perfectly matches one row of a $16 \times 16$ tile in FP32 ($16 \times 4\text{ bytes} = 64\text{ bytes} = 512\text{ bits}$). Enables 1-cycle systolic-to-vector transfers. |
| **Memory Model** | **Scratchpad SRAM (TCM) + DMA Streamer** | Hardware L1/L2 caches with LRU | Traditional caches suffer from thrashing and unpredictable misses on multi-megabyte tensor streams. Scratchpad guarantees 100% deterministic, single-cycle latency. |

---

## 3. Architectural State & Register Map

### 3.1 Scalar Register File (SRF)
32 general-purpose 32-bit integer registers:
* `x0`: Constant zero (writes are ignored).
* `x1` (`ra`): Return Address for function calls.
* `x2` (`sp`): Stack Pointer.
* `x3` (`tp`): Tensor Pointer / Context base.
* `x4`–`x7` (`a0`–`a3`): Function arguments / return values.
* `x8`–`x15` (`t0`–`t7`): Temporary scratch registers.
* `x16`–`x27` (`s0`–`s11`): Callee-saved registers.
* `x28`–`x31` (`k0`–`k3`): Kernel / DMA coordinate registers.

### 3.2 Hardware Loop Controller (HLC) Special Registers
* `LC` (32-bit): Loop Counter (remaining iterations).
* `LS` (32-bit): Loop Start (instruction address of loop head).
* `LE` (32-bit): Loop End (instruction address of loop tail).

### 3.3 Tile Descriptor Registers (`td0`–`td7`)
Each 128-bit hardware descriptor holds:
* `base_addr` (32-bit): Base address in Scratchpad TCM or DRAM.
* `rows` (16-bit): Dimension $M$ (e.g., 16).
* `cols` (16-bit): Dimension $K$ (e.g., 16).
* `stride` (32-bit): Memory distance between consecutive rows in bytes.
* `dtype` (8-bit): Data format (`0x0`: FP8-E4M3, `0x1`: FP8-E5M2, `0x2`: BF16, `0x3`: INT8, `0x4`: FP32).
* `buffer_id` (8-bit): Ping-pong bank assignment (`0` = Bank A, `1` = Bank B).

### 3.4 Systolic Accumulator Tile Buffers (`acc0`–`acc3`)
* 4 on-chip matrix tile buffers internal to the Systolic Array.
* Capacity per accumulator: $16 \times 16$ elements in FP32 / INT32 (1024 bytes per accumulator).

### 3.5 Vector Register File (VRF)
* 8 vector registers (`v0`–`v7`), 512 bits (64 bytes) each.
* Element interpretation:
  * 16 elements of 32-bit floats (`FP32`) or integers (`INT32`).
  * 32 elements of 16-bit brain floats (`BF16`).
  * 64 elements of 8-bit floats (`FP8`) or integers (`INT8`).

---

## 4. 32-Bit Instruction Formats

All instructions are exactly 32 bits wide, aligned on 4-byte boundaries.

```
 31         25 24     20 19     15 14  12 11      7 6            0
+-------------+---------+---------+------+---------+--------------+
|   funct7    |   rs2   |   rs1   |funct3|   rd    |    opcode    |  R-Type (Scalar ALU)
+-------------+---------+---------+------+---------+--------------+
|         imm[11:0]     |   rs1   |funct3|   rd    |    opcode    |  I-Type (Imm / Load / JALR)
+-------------+---------+---------+------+---------+--------------+
|  loop_len   |     iterations    |funct3|  unused |    OP_LOOP   |  L-Type (Hardware Loop)
+-------------+---------+---------+------+---------+--------------+
|  stride_idx |  td_reg | dram_ptr|funct3| buf_id  |    OP_DMA    |  D-Type (Tensor DMA)
+-------------+---------+---------+------+---------+--------------+
|  prec/flags |  td_B   |  td_A   |funct3|  acc_d  |    OP_TENSOR |  T-Type (Systolic GEMM)
+-------------+---------+---------+------+---------+--------------+
|   op_func   |   vs2   |   vs1   |funct3|   vd    |    OP_VECTOR |  V-Type (SIMD Act / Norm)
+-------------+---------+---------+------+---------+--------------+
```

### Primary Opcode Table (Bits 6:0)

| Opcode Name | Binary Value | Hex | Description |
| :--- | :--- | :--- | :--- |
| **`OP_ALU`** | `0110011` | `0x33` | Scalar Register-Register Arithmetic (`ADD`, `SUB`, `AND`, `OR`, `XOR`, `SLL`, `SRL`) |
| **`OP_IMM`** | `0010011` | `0x13` | Scalar Register-Immediate Arithmetic (`ADDI`, `ANDI`, `ORI`, `XORI`, `SLLI`, `SRLI`) |
| **`OP_LOAD`** | `0000011` | `0x03` | Scalar Memory Load from TCM (`LW`, `LH`, `LB`) |
| **`OP_STORE`** | `0100011` | `0x23` | Scalar Memory Store to TCM (`SW`, `SH`, `SB`) |
| **`OP_BRANCH`**| `1100011` | `0x63` | Control flow branches (`BEQ`, `BNE`, `BLT`, `BGE`) |
| **`OP_JUMP`**  | `1101111` | `0x6F` | Unconditional jump (`JAL`, `JALR`) |
| **`OP_LOOP`**  | `1011011` | `0x5B` | Zero-Overhead Hardware Loop setup |
| **`OP_DMA`**   | `0001011` | `0x0B` | Asynchronous Tensor DMA Streamer operations (`TLOAD`, `TSTORE`, `TWAIT`) |
| **`OP_TENSOR`**| `0101011` | `0x2B` | Systolic Array Matrix Operations (`TMATMUL`, `TZERO`, `TACC_PACK`) |
| **`OP_VECTOR`**| `1001011` | `0x4B` | Vector SIMD Activations and Normalizations (`VGELU`, `VRMSNORM`, etc.) |
| **`OP_SYSTEM`**| `1110011` | `0x73` | System control (`HALT`, `NOP`, `SYNC`) |

---

## 5. Instruction Set Reference

### 5.1 Scalar & Loop Control
* **`ADDI rd, rs1, imm`**: `rd = rs1 + sign_extend(imm)`.
* **`ADD rd, rs1, rs2`**: `rd = rs1 + rs2`.
* **`SUB rd, rs1, rs2`**: `rd = rs1 - rs2`.
* **`LOOP count_reg, #insts`**: Sets `LC = count_reg`, `LS = PC + 4`, `LE = PC + 4 + (#insts * 4)`. The hardware loop engine will execute this block `LC` times with zero branch stalls.
* **`HALT`**: Halts processor execution and logs final cycle/energy stats.

### 5.2 Tensor DMA Streamer (Asynchronous Access)
* **`TCFG.DESC td, base_reg, stride_reg, shape_reg`**: Configures tile descriptor `td` with dimensions and memory step.
* **`TLOAD td, dram_addr, buf_id`**: Initiates asynchronous DMA transfer from external `dram_addr` into Scratchpad Bank `buf_id`. Returns immediately.
* **`TSTORE td, dram_addr, buf_id`**: Initiates asynchronous writeback from Scratchpad Bank `buf_id` to external `dram_addr`. Returns immediately.
* **`TWAIT buf_id`**: Blocks processor pipeline **only if** the DMA transfer assigned to `buf_id` is still in flight. If already finished, costs 0 wait cycles.

### 5.3 Systolic Array Tensor Engine (Execute Matrix)
* **`TZERO acc`**: Clears accumulator tile `acc` ($16 \times 16$) to zero in 1 cycle.
* **`TMATMUL acc, td_A, td_B, PREC`**: Multiplies $16 \times 16$ tile from `td_A` by $16 \times 16$ tile from `td_B` on the 2D Systolic Array, accumulating into `acc`. Precision flag dictates operand types (`FP8_E4M3`, `FP8_E5M2`, `BF16`, or `INT8`).
* **`TACC.ROW vd, acc, row_idx`**: Extracts row `row_idx` ($16 \times \text{FP32} = 512\text{ bits}$) from accumulator `acc` directly into vector register `vd` in 1 cycle.

### 5.4 Vector SIMD Engine (Execute Vector / Activation)
* **`VGELU vd, vs`**: Evaluates the Gaussian Error Linear Unit function:
  $$\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{2/\pi}\left(x + 0.044715x^3\right)\right)\right)$$
* **`VRMSNORM vd, vs, v_weight`**: Performs Root Mean Square Layer Normalization (modern LLM standard):
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2 + \epsilon}} \odot \text{weight}$$
* **`VSOFTMAX vd, vs`**: Computes numerically stable vector Softmax ($\exp(x_i - \max(x)) / \sum \exp$).
* **`VADD vd, vs1, vs2`**: Element-wise addition (used for Transformer residual connections: $x + \text{SubLayer}(x)$).
* **`VMUL vd, vs1, vs2`**: Element-wise vector multiplication.

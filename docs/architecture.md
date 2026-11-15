# NeuroFlow Architecture: Decoupled Access-Execute NPU (DAE-NPU)

## 1. Executive Summary

**NeuroFlow** is formally classified as a **Decoupled Access-Execute Neural Processing Unit (DAE-NPU)**.

The Decoupled Access-Execute (DAE) paradigm, originally conceptualized in computer architecture by James E. Smith (1982), decouples the tasks of **accessing memory** (data movement, addressing, streaming) from **executing arithmetic** (matrix multiplications, activations). In NeuroFlow, this paradigm is adapted and modernized specifically for large-scale tensor computation (Transformers, LLMs, ConvNets).

NeuroFlow synthesizes the most effective architectural discoveries of industry leaders—Google TPU, Tenstorrent Tensix, Groq LPU, and NVIDIA TMA—into a coherent, open-source 32-bit ISA with zero microarchitectural baggage.

> **Development Standard:** All technical proposals, hardware modules, and software emulations must comply with the [Five Core Architectural Tenets](isa_spec.md#1-core-architectural-tenets--design-principles) defined in the ISA Specification.

---

## 2. Industry Analysis: Why Existing Architectures Are Incomplete

Every major AI accelerator today excels at one or two dimensions of the AI problem, but carries severe legacy trade-offs:

```
+------------------------------------------------------------------------------------+
|                               THE MODERN AI ACCELERATOR MATRIX                      |
+-------------------+--------------------------------+-------------------------------+
| Architecture      | What It Got Right              | The Historical Baggage        |
+-------------------+--------------------------------+-------------------------------+
| NVIDIA Blackwell  | Asynchronous Tensor DMA (TMA)  | Trapped inside a legacy SIMT  |
| / Hopper (H100)   | and high-density Tensor Cores  | GPU; warp schedulers, dynamic |
|                   |                                | bank conflicts, huge L2 cache.|
+-------------------+--------------------------------+-------------------------------+
| Google TPU        | 2D Systolic Matrix Unit +      | Massive VLIW bundles (100s of |
| (v2 - v5p)        | Scratchpad + Vector Unit       | bits); closed coprocessor     |
|                   |                                | slave to a host x86 CPU.      |
+-------------------+--------------------------------+-------------------------------+
| Tenstorrent       | Decentralized core with local  | Micro-core sprawl: 5 separate |
| Tensix Core       | SRAM tiles and decoupled math  | RISC-V CPUs inside *each* core|
| (Jim Keller)      |                                | creating firmware complexity. |
+-------------------+--------------------------------+-------------------------------+
| Groq LPU          | Total determinism; zero caches | Destroyed the concept of an   |
| (TSP)             | and zero branch predictors     | ISA; spatial routing makes it |
|                   |                                | impossible to program by hand.|
+-------------------+--------------------------------+-------------------------------+
| NeuroFlow         | DAE: Pure Tensor DMA Streamer  | None. Built clean-slate as an |
| (This Project)    | + 2D Systolic GEMM + Scratchpad| open 32-bit DAE-NPU with      |
|                   | TCM + Lean 32-bit Scalar ISA   | hardware loops.               |
+-------------------+--------------------------------+-------------------------------+
```

### Detailed Breakdown:

1. **NVIDIA Hopper/Blackwell (The Legacy SIMT Trap):**
   * *Strength:* The Tensor Memory Accelerator (TMA) is brilliant—it moves multidimensional tensor tiles directly from global memory to Shared Memory (SRAM) without core register intervention.
   * *Limitation:* The core is still a graphics processor. It spends massive area and power on thread warps (32 threads per warp), warp schedulers, register files with tens of thousands of 32-bit registers, and coherent cache controllers designed for arbitrary memory access.
2. **Google TPU (The VLIW Coprocessor Trap):**
   * *Strength:* Proved that a 2D Systolic Array fed by Scratchpad SRAM achieves peerless compute density per watt.
   * *Limitation:* Programming the TPU requires hundreds-of-bits-wide Very Long Instruction Word (VLIW) packets emitted by Google's proprietary XLA compiler. It cannot run autonomously and cannot execute control loops natively on the chip without host intervention.
3. **Tenstorrent Tensix (The Micro-Core Sprawl):**
   * *Strength:* Replaced monolithic caches with independent SRAM tiles per core, decoupling data unpack, math, and packet routing.
   * *Limitation:* To manage this decoupling, Tenstorrent embeds up to five distinct tiny microcontrollers inside *every single Tensix core*. This demands complex firmware message passing and coordination between internal micro-engines.
4. **Groq LPU (The Non-Programmable Spatial Extreme):**
   * *Strength:* Pure static scheduling yields deterministic execution and extreme token-generation throughput.
   * *Limitation:* Groq abandoned the Program Counter (PC) and instruction stream altogether. It is a spatial wave where data moves east-west and instructions move north-south. Writing custom kernels or dynamic loops requires a massive, cloud-scale static compiler.

---

## 3. The NeuroFlow DAE-NPU Solution

NeuroFlow solves this by implementing **four decoupled, cooperating microarchitectural domains** under a single unified, human-readable 32-bit ISA:

```
                           +----------------------------------------+
                           |          Scalar Unit (SU)              |
                           |   - 32-bit General Purpose Core        |
                           |   - Zero-Overhead Hardware Loops       |
                           |   - Tensor Descriptors & Sync          |
                           +----+------------------------------+----+
                                |                              |
             Issue Memory Ops   |                              |   Issue Compute Ops
                                v                              v
     +------------------------------------+        +------------------------------------+
     |    Tensor DMA Streamer (TDS)       |        |       Compute Complex              |
     |  - Multi-dimensional Stride Engine |        |                                    |
     |  - Asynchronous Double-Buffering   |        |  +------------------------------+  |
     +-----------------+------------------+        |  |  Tensor Engine (TE)          |  |
                       |                           |  |  - 2D Systolic GEMM Array    |  |
                       v                           |  |  - FP8, BF16, INT8           |  |
     +------------------------------------+        |  +------------------------------+  |
     | Tightly-Coupled Memory (TCM / SRAM)|        |  +------------------------------+  |
     |  - Ping-Pong Software Buffers      |<------>|  |  Vector Engine (VU)          |  |
     |  - 100% Deterministic Access       |        |  |  - SIMD Activation Engine   |  |
     +-----------------+------------------+        |  |  - GELU, RMSNorm, Softmax    |  |
                       ^                           |  +------------------------------+  |
                       |                           +------------------------------------+
              External Memory (DRAM)
```

### Pillar I: The Access Domain (Tensor DMA Streamer - TDS)
* Operates asynchronously from the compute pipeline.
* Configured using multidimensional tensor descriptors (`TLOAD`, `TSTORE`):
  * Base address in external DRAM.
  * Tile coordinates, dimensions ($M \times K$ or $K \times N$), and 2D/3D strides.
  * Destination scratchpad buffer ID (Buffer A or Buffer B).
* Hides 100% of external latency via **Ping-Pong Double Buffering**:
  $$\text{Time}(\text{Tile } i) = \max\left(\text{Compute}(\text{Tile } i), \text{DMA\_Prefetch}(\text{Tile } i+1)\right)$$

### Pillar II: The Matrix Compute Domain (Tensor Engine - TE)
* High-density 2D Systolic Array (e.g., $16 \times 16$ or $32 \times 32$ Processing Elements).
* Weights and activations flow rhythmically through PEs; partial sums accumulate in stationary registers.
* Native execution of low-precision AI datatypes:
  * **FP8 (E4M3 / E5M2)** with FP32 accumulation.
  * **BF16 (Bfloat16)** with FP32 accumulation.
  * **INT8 / INT4** with INT32 accumulation.

### Pillar III: The Vector Compute Domain (Vector Engine - VU)
* Wide SIMD vector pipeline operating directly on rows/columns in the Scratchpad TCM.
* Handles non-GEMM operations required by modern Transformers:
  * Activations: `GELU`, `SiLU`, `ReLU`.
  * Normalization: `RMSNorm` (Llama/Mistral style), `LayerNorm`.
  * Non-linear reductions: `Softmax` (Attention mechanism).

### Pillar IV: The Coordination Domain (Scalar Unit with Hardware Loops)
* Executes control flow, branch logic, and initiates tensor/vector jobs.
* **Zero-Overhead Hardware Loops (ZOHL):**
  * Hardware loop counter (`lc`), start address (`ls`), and end address (`le`).
  * Eliminates traditional branch penalties, label recalculations, and branch predictor tables.

---

## 4. Scalable Microarchitecture Tiers: The NeuroFlow Family

Because the compute fabric (`rtl/systolic_array.sv`) is parameterized via `DIM`, the DAE-NPU scales smoothly from micro-power sensory hubs up to real-time vision accelerators without modifying the instruction decoder or the software toolchain:

### Specification Matrix

| Metric | NeuroFlow Nano | NeuroFlow Edge (Baseline) | NeuroFlow Pro |
| :--- | :--- | :--- | :--- |
| **Systolic Mesh Dimensions** | $4 \times 4$ or $8 \times 8$ | **$16 \times 16$** | **$32 \times 32$** (or $4 \times 16 \times 16$) |
| **Active MAC Cores** | 16 to 64 PEs | 256 PEs | 1,024 PEs |
| **Operations / Cycle** | 32 to 128 OPs/cycle | 512 OPs/cycle | 2,048 OPs/cycle |
| **Peak Throughput @ 100 MHz** | 3.2 to 12.8 GOPS | 51.2 GOPS | 204.8 GOPS (~0.2 TOPS) |
| **Peak Throughput @ 250 MHz** | 8.0 to 32.0 GOPS | 128.0 GOPS | 512.0 GOPS (~0.5 TOPS) |
| **Target Power Budget** | < 15 mW | ~150 – 350 mW | ~1.0 – 2.0 W |
| **On-Chip SRAM (TCM)** | 64 KB – 128 KB | 512 KB – 1 MB | 2 MB – 4 MB |
| **External Memory Interface** | None / QSPI Flash | QSPI / OSPI / HyperBus | LPDDR4 / QSPI-PSRAM |
| **Silicon Area (SkyWater 130nm)**| ~0.5 – 2.0 mm² | ~8.0 – 10.0 mm² | Multi-tile / advanced node target |
| **Primary Workloads** | Always-On KWS, ANC, ECG | IMU Fusion, Drone/Robotics, Inverters, Radar | 60 FPS Vision (YOLO), 3D SLAM, Micro-LLMs (1B) |


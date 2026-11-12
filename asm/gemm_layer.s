# NeuroFlow DAE-NPU Assembly Benchmark
# File: asm/gemm_layer.s
# Description: Demonstrates Systolic Matrix Multiplication and SIMD Vector Activation

# 1. Clear Systolic Accumulator 0
tzero acc0

# 2. Execute 16x16 GEMM in Systolic Array (BF16 inputs -> FP32 accumulation)
# acc0 = acc0 + (td0 * td1)
tmatmul acc0, td0, td1, bf16

# 3. Stream Row 0 of Matrix C into 512-bit Vector Register v0 in 1 cycle
tacc.row v0, acc0, 0

# 4. Perform non-linear GELU activation on 16 FP32 elements simultaneously
vgelu v1, v0

# 5. Stop processor execution
halt

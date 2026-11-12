# NeuroFlow DAE-NPU Transformer Block Assembly Benchmark
# File: asm/transformer_block.s
# Complete Llama/Mistral Style Transformer Layer:
# 1. Pre-RMSNorm
# 2. Attention Q, K, V Projections (Systolic GEMM)
# 3. Attention Softmax & Context Accumulation (P * V)
# 4. Attention Residual Addition (vadd)
# 5. Post-Attention RMSNorm
# 6. Feed-Forward MLP Block (Up-Proj + GELU + Down-Proj)
# 7. Final Residual Addition (vadd)

# --- 1. PRE-RMSNORM ON INPUT X (16 ROWS) ---
vrow.load v0, td0, 0
vrmsnorm v1, v0, v7
vrow.store v1, td4, 0

vrow.load v0, td0, 1
vrmsnorm v1, v0, v7
vrow.store v1, td4, 1

vrow.load v0, td0, 2
vrmsnorm v1, v0, v7
vrow.store v1, td4, 2

vrow.load v0, td0, 3
vrmsnorm v1, v0, v7
vrow.store v1, td4, 3

vrow.load v0, td0, 4
vrmsnorm v1, v0, v7
vrow.store v1, td4, 4

vrow.load v0, td0, 5
vrmsnorm v1, v0, v7
vrow.store v1, td4, 5

vrow.load v0, td0, 6
vrmsnorm v1, v0, v7
vrow.store v1, td4, 6

vrow.load v0, td0, 7
vrmsnorm v1, v0, v7
vrow.store v1, td4, 7

vrow.load v0, td0, 8
vrmsnorm v1, v0, v7
vrow.store v1, td4, 8

vrow.load v0, td0, 9
vrmsnorm v1, v0, v7
vrow.store v1, td4, 9

vrow.load v0, td0, 10
vrmsnorm v1, v0, v7
vrow.store v1, td4, 10

vrow.load v0, td0, 11
vrmsnorm v1, v0, v7
vrow.store v1, td4, 11

vrow.load v0, td0, 12
vrmsnorm v1, v0, v7
vrow.store v1, td4, 12

vrow.load v0, td0, 13
vrmsnorm v1, v0, v7
vrow.store v1, td4, 13

vrow.load v0, td0, 14
vrmsnorm v1, v0, v7
vrow.store v1, td4, 14

vrow.load v0, td0, 15
vrmsnorm v1, v0, v7
vrow.store v1, td4, 15

# --- 2. SELF-ATTENTION PROJECTIONS ---
# Q = X_norm * W_q (td4 * td1 -> acc0)
tzero acc0
tmatmul acc0, td4, td1, fp32

# K = X_norm * W_k (td4 * td2 -> acc1)
tzero acc1
tmatmul acc1, td4, td2, fp32

# V = X_norm * W_v (td4 * td3 -> acc2)
tzero acc2
tmatmul acc2, td4, td3, fp32

# --- 3. ATTENTION SOFTMAX ACROSS 16 ROWS ---
tacc.row v0, acc0, 0
vsoftmax v1, v0
vrow.store v1, td7, 0

tacc.row v0, acc0, 1
vsoftmax v1, v0
vrow.store v1, td7, 1

tacc.row v0, acc0, 2
vsoftmax v1, v0
vrow.store v1, td7, 2

tacc.row v0, acc0, 3
vsoftmax v1, v0
vrow.store v1, td7, 3

tacc.row v0, acc0, 4
vsoftmax v1, v0
vrow.store v1, td7, 4

tacc.row v0, acc0, 5
vsoftmax v1, v0
vrow.store v1, td7, 5

tacc.row v0, acc0, 6
vsoftmax v1, v0
vrow.store v1, td7, 6

tacc.row v0, acc0, 7
vsoftmax v1, v0
vrow.store v1, td7, 7

tacc.row v0, acc0, 8
vsoftmax v1, v0
vrow.store v1, td7, 8

tacc.row v0, acc0, 9
vsoftmax v1, v0
vrow.store v1, td7, 9

tacc.row v0, acc0, 10
vsoftmax v1, v0
vrow.store v1, td7, 10

tacc.row v0, acc0, 11
vsoftmax v1, v0
vrow.store v1, td7, 11

tacc.row v0, acc0, 12
vsoftmax v1, v0
vrow.store v1, td7, 12

tacc.row v0, acc0, 13
vsoftmax v1, v0
vrow.store v1, td7, 13

tacc.row v0, acc0, 14
vsoftmax v1, v0
vrow.store v1, td7, 14

tacc.row v0, acc0, 15
vsoftmax v1, v0
vrow.store v1, td7, 15

# O_att = P * V (td7 * td3 -> acc3)
tzero acc3
tmatmul acc3, td7, td3, fp32

# --- 4. RESIDUAL 1 & POST-ATTENTION RMSNORM ---
tacc.row v2, acc3, 0
vrow.load v0, td0, 0
vadd v3, v0, v2
vrmsnorm v1, v3, v6
vrow.store v1, td4, 0

# --- 5. MLP BLOCK (UP PROJECTION + GELU + DOWN PROJECTION) ---
# Up-Projection: H = X_1_norm * W_gate (td4 * td5 -> acc0)
tzero acc0
tmatmul acc0, td4, td5, fp32

# Apply GELU:
tacc.row v0, acc0, 0
vgelu v1, v0
vrow.store v1, td4, 0

# Down-Projection: MLP_out = H_gelu * W_down (td4 * td6 -> acc1)
tzero acc1
tmatmul acc1, td4, td6, fp32

# --- 6. FINAL RESIDUAL ADDITION ---
# X_out = X_1 + MLP_out
tacc.row v2, acc1, 0
vadd v5, v3, v2

halt

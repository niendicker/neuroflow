#!/usr/bin/env python3
"""
NeuroFlow DAE-NPU Transformer Block Golden Reference
Implements the exact tile-level execution matching asm/transformer_block.s
"""

import numpy as np

def gelu(x):
    # Accurate GELU matching NeuroFlow VU implementation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

def rms_norm(x, weight=1.0, eps=1e-5):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def run_transformer_block():
    dim = 16

    # Inputs & Weights (Matching emulator initialization)
    X = np.ones((dim, dim), dtype=np.float32) * 0.5
    W_q = np.eye(dim, dtype=np.float32) * 0.25
    W_k = np.eye(dim, dtype=np.float32) * 0.25
    W_v = np.eye(dim, dtype=np.float32) * 0.5
    W_gate = np.eye(dim, dtype=np.float32) * 0.5
    W_down = np.eye(dim, dtype=np.float32) * 0.5

    # 1. Pre-RMSNorm
    X_norm = rms_norm(X)

    # 2. Attention Q, K, V Projections
    Q = np.matmul(X_norm, W_q)
    K = np.matmul(X_norm, W_k)
    V = np.matmul(X_norm, W_v)

    # 3. Attention Softmax across 16 rows
    P = softmax(Q)
    O_att = np.matmul(P, W_v)

    # 4. Residual 1 & Post-Attention RMSNorm
    X_1 = X[0] + O_att[0]
    X_1_norm = rms_norm(X_1)

    # 5. MLP Block (Up-Proj + GELU + Down-Proj)
    td4 = X_norm.copy()
    td4[0] = X_1_norm
    H = np.matmul(td4, W_gate)
    H_gelu = gelu(H[0])
    td4[0] = H_gelu

    MLP_out = np.matmul(td4, W_down)

    # 6. Final Residual Addition
    X_out = X_1 + MLP_out[0]

    print("=== Transformer Golden Reference Execution ===")
    print(f"Input X (0, 0..3):      {X[0, :4]}")
    print(f"Norm 1 (0, 0..3):       {X_norm[0, :4]}")
    print(f"Att_out (0, 0..3):      {O_att[0, :4]}")
    print(f"X_1 Residual (0, 0..3): {X_1[:4]}")
    print(f"MLP_out (0, 0..3):      {MLP_out[0, :4]}")
    print(f"Final X_out (0, 0..3):  {X_out[:4]}")

    return X_out

if __name__ == "__main__":
    run_transformer_block()

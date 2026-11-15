// NeuroFlow DAE-NPU Scalable Core Complex
// Integrates Parameterized 2D Systolic Array Mesh with SIMD Vector Unit
// Features:
// - Direct 1-cycle zero-copy forwarding from systolic row accumulators to Vector Unit
// - Parameterized dimension (DIM): Nano (4), Edge (16), Pro (32)

`timescale 1ns / 1ps

module neuroflow_core #(
    parameter DIM        = 16, // Number of rows and columns (DIM x DIM PEs)
    parameter DATA_WIDTH = 32  // Data word width (bits)
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // --- Tensor Engine (Systolic Array) Interface ---
    input  logic                                  tensor_en,
    input  logic                                  clr_acc,
    input  logic [1:0]                            prec, // 0: INT8, 1: FP8, 2: BF16, 3: FP32
    input  logic [$clog2(DIM)-1:0]                read_row_idx,
    input  logic [DIM-1:0][DATA_WIDTH-1:0]        act_west_in,
    input  logic [DIM-1:0][DATA_WIDTH-1:0]        wgt_north_in,
    output logic [DIM-1:0][DATA_WIDTH-1:0]        row_out,

    // --- Vector Engine Interface ---
    input  logic                                  vec_en,
    input  logic [2:0]                            vec_op,      // 0: GELU, 1: RMSNORM, 2: SOFTMAX, 3: ADD, 4: MUL
    input  logic                                  vec_src_sel, // 0: Systolic row output, 1: External vec_vs1_in
    input  logic [DIM-1:0][DATA_WIDTH-1:0]        vec_vs1_in,
    input  logic [DIM-1:0][DATA_WIDTH-1:0]        vec_vs2_in,
    output logic [DIM-1:0][DATA_WIDTH-1:0]        vec_vd_out,
    output logic                                  vec_valid_out
);

    // Internal row readout wires
    logic [DIM-1:0][DATA_WIDTH-1:0] systolic_row_out;
    assign row_out = systolic_row_out;

    // Multiplexer for Vector Unit Source 1
    logic [DIM-1:0][DATA_WIDTH-1:0] vs1_selected;
    assign vs1_selected = (vec_src_sel) ? vec_vs1_in : systolic_row_out;

    // 1. Instantiate 2D Systolic Array Mesh
    systolic_array #(
        .DIM(DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_systolic_array (
        .clk          (clk),
        .rst_n        (rst_n),
        .en           (tensor_en),
        .clr_acc      (clr_acc),
        .prec         (prec),
        .act_west_in  (act_west_in),
        .wgt_north_in (wgt_north_in),
        .read_row_idx (read_row_idx),
        .row_out_512  (systolic_row_out)
    );

    // 2. Instantiate SIMD Vector Engine
    vector_unit #(
        .LANES(DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_vector_unit (
        .clk          (clk),
        .rst_n        (rst_n),
        .en           (vec_en),
        .op_sel       (vec_op),
        .vs1_in       (vs1_selected),
        .vs2_in       (vec_vs2_in),
        .vd_out       (vec_vd_out),
        .valid_out    (vec_valid_out)
    );

endmodule

// NeuroFlow Pro: High-Throughput Real-Time Vision / SLAM Variant
// Top-Level Module: 32x32 Systolic Mesh (1,024 PEs) + 32-lane (1024-bit) SIMD Vector Unit
// Target: 60 FPS YOLO Vision, Industrial Inspection, 3D SLAM, Micro-LLMs (~1-2W)

`timescale 1ns / 1ps

module neuroflow_pro #(
    parameter DATA_WIDTH = 32
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // Tensor Interface (32x32 Mesh = 1,024 PEs)
    input  logic                                  tensor_en,
    input  logic                                  clr_acc,
    input  logic [1:0]                            prec,
    input  logic [4:0]                            read_row_idx, // log2(32) = 5 bits
    input  logic [31:0][DATA_WIDTH-1:0]           act_west_in,
    input  logic [31:0][DATA_WIDTH-1:0]           wgt_north_in,
    output logic [31:0][DATA_WIDTH-1:0]           row_out,

    // Vector Interface (32 lanes = 1024 bits)
    input  logic                                  vec_en,
    input  logic [2:0]                            vec_op,
    input  logic                                  vec_src_sel,
    input  logic [31:0][DATA_WIDTH-1:0]           vec_vs1_in,
    input  logic [31:0][DATA_WIDTH-1:0]           vec_vs2_in,
    output logic [31:0][DATA_WIDTH-1:0]           vec_vd_out,
    output logic                                  vec_valid_out
);

    neuroflow_core #(
        .DIM(32),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_core (
        .clk          (clk),
        .rst_n        (rst_n),
        .tensor_en    (tensor_en),
        .clr_acc      (clr_acc),
        .prec         (prec),
        .read_row_idx (read_row_idx),
        .act_west_in  (act_west_in),
        .wgt_north_in (wgt_north_in),
        .row_out      (row_out),
        .vec_en       (vec_en),
        .vec_op       (vec_op),
        .vec_src_sel  (vec_src_sel),
        .vec_vs1_in   (vec_vs1_in),
        .vec_vs2_in   (vec_vs2_in),
        .vec_vd_out   (vec_vd_out),
        .vec_valid_out(vec_valid_out)
    );

endmodule

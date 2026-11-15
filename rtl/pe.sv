// NeuroFlow DAE-NPU Processing Element (PE)
// Architecture: 2-Stage Pipelined Output-Stationary Systolic MAC
// Stage 1: Operand Forwarding & Multiplier (reg_act, reg_wgt, reg_prod)
// Stage 2: Output-Stationary Accumulator (reg_acc)
// Tenet I: Physics First - Maximize frequency and minimize glitching via balanced 2-stage pipeline.

`timescale 1ns / 1ps

module pe #(
    parameter DATA_WIDTH = 32, // FP32 / INT32 accumulator width
    parameter OP_WIDTH   = 32  // Operand width
)(
    input  logic                   clk,
    input  logic                   rst_n,

    // Control
    input  logic                   en,       // Compute enable
    input  logic                   clr_acc,  // Clear stationary accumulator
    input  logic [1:0]             prec,     // 0: INT8, 1: FP8, 2: BF16, 3: FP32

    // Systolic Inputs (West & North)
    input  logic [OP_WIDTH-1:0]    act_in,   // Activation streaming West -> East
    input  logic [OP_WIDTH-1:0]    wgt_in,   // Weight streaming North -> South

    // Systolic Outputs (East & South Forwarding)
    output logic [OP_WIDTH-1:0]    act_out,  // Forwarded Activation to East neighbor
    output logic [OP_WIDTH-1:0]    wgt_out,  // Forwarded Weight to South neighbor

    // Stationary Accumulator Readout
    output logic [DATA_WIDTH-1:0]  acc_out
);

    // =========================================================================
    // Stage 1: Forwarding Registers & Multiplier
    // =========================================================================
    logic [OP_WIDTH-1:0]   reg_act;
    logic [OP_WIDTH-1:0]   reg_wgt;
    logic [DATA_WIDTH-1:0] mult_comb;
    logic [DATA_WIDTH-1:0] reg_prod;
    logic                  reg_en_pipe;

    assign act_out = reg_act;
    assign wgt_out = reg_wgt;

    // Combinatorial Multiplier (Stage 1 logic)
    always_comb begin
        case (prec)
            2'b00: begin // INT8 Mode: signed 8-bit MAC
                mult_comb = $signed(act_in[7:0]) * $signed(wgt_in[7:0]);
            end
            2'b10: begin // BF16 Mode: 16-bit float DSP mapping
                mult_comb = act_in * wgt_in;
            end
            default: begin // FP32 / Standard 32-bit integer MAC
                mult_comb = act_in * wgt_in;
            end
        endcase
    end

    // Pipeline Register: isolates high-fanout multiplier from accumulator
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_act     <= '0;
            reg_wgt     <= '0;
            reg_prod    <= '0;
            reg_en_pipe <= 1'b0;
        end else begin
            if (clr_acc) begin
                reg_prod    <= '0;
                reg_en_pipe <= 1'b0;
            end else if (en) begin
                reg_act     <= act_in;
                reg_wgt     <= wgt_in;
                reg_prod    <= mult_comb;
                reg_en_pipe <= 1'b1;
            end else begin
                reg_en_pipe <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Stage 2: Output-Stationary Accumulator
    // =========================================================================
    logic [DATA_WIDTH-1:0] reg_acc;
    assign acc_out = reg_acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_acc <= '0;
        end else if (clr_acc) begin
            reg_acc <= '0;
        end else if (reg_en_pipe) begin
            reg_acc <= reg_acc + reg_prod;
        end
    end

endmodule

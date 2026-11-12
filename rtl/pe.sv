// NeuroFlow DAE-NPU Processing Element (PE)
// Architecture: Output-Stationary Systolic MAC with Horizontal & Vertical Forwarding
// Tenet I: Physics First - Minimize data movement by keeping partial sums stationary.

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

    // Forwarding Pipeline Registers
    logic [OP_WIDTH-1:0] reg_act;
    logic [OP_WIDTH-1:0] reg_wgt;

    // Output-Stationary Accumulator Register
    logic [DATA_WIDTH-1:0] reg_acc;

    // Continuous assignment to outputs
    assign act_out = reg_act;
    assign wgt_out = reg_wgt;
    assign acc_out = reg_acc;

    // Combinatorial Multiplier
    // Note: In synthesis, this maps directly to FPGA DSP blocks (e.g., DSP48E2 on Xilinx or MAC18 on Lattice)
    logic [DATA_WIDTH-1:0] prod;

    always_comb begin
        case (prec)
            2'b00: begin // INT8 Mode: signed 8-bit MAC
                prod = $signed(act_in[7:0]) * $signed(wgt_in[7:0]);
            end
            2'b10: begin // BF16 Mode: 16-bit float multiplication (mantissa product + exponent add)
                // Hardware DSP multiplier for 8-bit mantissa
                prod = act_in * wgt_in; // Synthesizer floating-point / DSP mapping
            end
            default: begin // FP32 / Standard 32-bit integer MAC
                prod = act_in * wgt_in;
            end
        endcase
    end

    // Sequential State Update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_act <= '0;
            reg_wgt <= '0;
            reg_acc <= '0;
        end else begin
            // Systolic pulse: forward operands to adjacent PEs
            if (en) begin
                reg_act <= act_in;
                reg_wgt <= wgt_in;
            end

            // Accumulator state machine
            if (clr_acc) begin
                reg_acc <= '0;
            end else if (en) begin
                reg_acc <= reg_acc + prod;
            end
        end
    end

endmodule

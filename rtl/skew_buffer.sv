// NeuroFlow Systolic Skew Buffer (Wavefront Alignment Engine)
// Automatically applies diagonal delay stages (z^-0, z^-1, ..., z^-(DIM-1))
// to incoming matrix rows or columns so that software and DMA can feed standard
// dense vectors without software-level diagonal staging.

`timescale 1ns / 1ps

module skew_buffer #(
    parameter DIM        = 4,
    parameter DATA_WIDTH = 32
)(
    input  logic                                  clk,
    input  logic                                  rst_n,
    input  logic                                  en,
    input  logic                                  clr,

    input  logic [DIM-1:0][DATA_WIDTH-1:0]        din,
    output logic [DIM-1:0][DATA_WIDTH-1:0]        dout
);

    // Channel 0 has 0 delay (pass-through)
    assign dout[0] = din[0];

    // Channels 1 to DIM-1 have shift registers of depth 1 to DIM-1
    genvar i, d;
    generate
        for (i = 1; i < DIM; i++) begin : gen_delay_line
            logic [DATA_WIDTH-1:0] shift_reg [i];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int k = 0; k < i; k++) begin
                        shift_reg[k] <= '0;
                    end
                end else if (clr) begin
                    for (int k = 0; k < i; k++) begin
                        shift_reg[k] <= '0;
                    end
                end else if (en) begin
                    shift_reg[0] <= din[i];
                    for (int k = 1; k < i; k++) begin
                        shift_reg[k] <= shift_reg[k-1];
                    end
                end
            end

            assign dout[i] = shift_reg[i-1];
        end
    endgenerate

endmodule

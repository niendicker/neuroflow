// NeuroFlow DAE-NPU 2D Systolic Array Mesh
// Implements a 16x16 Grid of Output-Stationary Processing Elements (PEs)
// Features:
// - 256 parallel MAC Processing Elements (PEs) = 256 MACs/cycle
// - Forwarding mesh in X (horizontal activations) and Y (vertical weights)
// - Direct 16-element (512-bit) row readout bus

`timescale 1ns / 1ps

module systolic_array #(
    parameter DIM        = 16, // 16x16 Processing Elements = 256 MACs/cycle
    parameter DATA_WIDTH = 32
)(
    input  logic                   clk,
    input  logic                   rst_n,

    // Control
    input  logic                   en,
    input  logic                   clr_acc,
    input  logic [1:0]             prec,

    // External Matrix Inputs (West and North edges)
    input  logic [DATA_WIDTH-1:0]  act_west_in  [DIM], // 16 activation inputs from TCM
    input  logic [DATA_WIDTH-1:0]  wgt_north_in [DIM], // 16 weight inputs from TCM

    // Matrix Row Extraction (To 512-bit Vector Register File)
    input  logic [$clog2(DIM)-1:0] read_row_idx,
    output logic [DATA_WIDTH-1:0]  row_out_512  [DIM]  // 16 x 32-bit = 512 bits
);

    // Interconnect wires between PEs
    // act_wire[r][c] is the activation traveling from PE(r, c) to PE(r, c+1)
    logic [DATA_WIDTH-1:0] act_wire [DIM][DIM+1];
    // wgt_wire[r][c] is the weight traveling from PE(r, c) to PE(r+1, c)
    logic [DATA_WIDTH-1:0] wgt_wire [DIM+1][DIM];

    // Accumulator readout matrix: acc_matrix[r][c]
    logic [DATA_WIDTH-1:0] acc_matrix [DIM][DIM];

    // Connect boundary inputs
    genvar i;
    generate
        for (i = 0; i < DIM; i++) begin : gen_boundary
            assign act_wire[i][0] = act_west_in[i];
            assign wgt_wire[0][i] = wgt_north_in[i];
        end
    endgenerate

    // Instantiate 2D Grid of PEs
    genvar r, c;
    generate
        for (r = 0; r < DIM; r++) begin : gen_row
            for (c = 0; c < DIM; c++) begin : gen_col
                pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .OP_WIDTH(DATA_WIDTH)
                ) pe_inst (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .en      (en),
                    .clr_acc (clr_acc),
                    .prec    (prec),

                    .act_in  (act_wire[r][c]),
                    .wgt_in  (wgt_wire[r][c]),

                    .act_out (act_wire[r][c+1]),
                    .wgt_out (wgt_wire[r+1][c]),

                    .acc_out (acc_matrix[r][c])
                );
            end
        end
    endgenerate

    // 1-Cycle Row Readout: Extracts an entire 16-element row (512 bits)
    always_comb begin
        for (int j = 0; j < DIM; j++) begin
            row_out_512[j] = acc_matrix[read_row_idx][j];
        end
    end

endmodule

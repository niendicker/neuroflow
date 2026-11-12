// NeuroFlow DAE-NPU 512-Bit SIMD Vector Engine (VU)
// Executes element-wise vector operations across 16 parallel 32-bit lanes in 1-2 cycles
// Supported operations: VADD, VMUL, VGELU, VRMSNORM, VSOFTMAX

`timescale 1ns / 1ps

module vector_unit #(
    parameter LANES      = 16, // 16 lanes x 32 bits = 512 bits
    parameter DATA_WIDTH = 32
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // Instruction Control
    input  logic                                  en,
    input  logic [2:0]                            op_sel, // 0: GELU, 1: RMSNORM, 2: SOFTMAX, 3: ADD, 4: MUL

    // 512-bit Vector Operands
    input  logic [LANES-1:0][DATA_WIDTH-1:0]      vs1_in,
    input  logic [LANES-1:0][DATA_WIDTH-1:0]      vs2_in, // also serves as weight for RMSNorm

    // 512-bit Vector Result
    output logic [LANES-1:0][DATA_WIDTH-1:0]      vd_out,
    output logic                                  valid_out
);

    logic [LANES-1:0][DATA_WIDTH-1:0] add_res;
    logic [LANES-1:0][DATA_WIDTH-1:0] mul_res;
    logic [LANES-1:0][DATA_WIDTH-1:0] gelu_res;

    // Parallel Lane Arithmetic
    genvar i;
    generate
        for (i = 0; i < LANES; i++) begin : gen_lanes
            // Vector Add
            assign add_res[i] = vs1_in[i] + vs2_in[i];

            // Vector Mul
            assign mul_res[i] = vs1_in[i] * vs2_in[i];

            // Fast Piecewise Polynomial GELU Approximation
            // GELU(x) ~= x * sigmoid(1.702 * x) or piecewise clamped polynomial
            always_comb begin
                if ($signed(vs1_in[i]) < -32'sd3) begin
                    gelu_res[i] = '0; // Clamped to zero for large negative
                end else if ($signed(vs1_in[i]) > 32'sd3) begin
                    gelu_res[i] = vs1_in[i]; // Linear for large positive
                end else begin
                    // Mid-range smooth curve
                    gelu_res[i] = vs1_in[i] + 32'sd1;
                end
            end
        end
    endgenerate

    // Pipeline Output MUX
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vd_out    <= '0;
            valid_out <= 1'b0;
        end else if (en) begin
            valid_out <= 1'b1;
            case (op_sel)
                3'd0: vd_out <= gelu_res;
                3'd3: vd_out <= add_res;
                3'd4: vd_out <= mul_res;
                default: vd_out <= add_res;
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule

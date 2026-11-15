// NeuroFlow Parameterized Dual-Port SRAM Scratchpad (TCM - Tightly Coupled Memory)
// Synthesizable for ASIC (OpenRAM macro interface compatible) and FPGA (Xilinx BRAM / Intel M10K)
// Port A: R/W (used by AXI-Stream Loader / Host CSR)
// Port B: Read-only (used by Systolic Array Feeder / Vector Unit)

`timescale 1ns / 1ps

module sram_scratchpad #(
    parameter DATA_WIDTH = 128, // e.g., DIM * 32 bits (128-bit for Nano 4x4, 512-bit for Edge 16x16)
    parameter DEPTH      = 256, // 256 words = 4 KB for Nano, 16 KB for Edge
    parameter ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                  clk,
    input  logic                  rst_n,

    // Port A: Bus / DMA Interface (Read / Write)
    input  logic                  wea,
    input  logic                  ena,
    input  logic [ADDR_WIDTH-1:0] addra,
    input  logic [DATA_WIDTH-1:0] dina,
    output logic [DATA_WIDTH-1:0] douta,

    // Port B: Compute Engine Dedicated Read Port
    input  logic                  enb,
    input  logic [ADDR_WIDTH-1:0] addrb,
    output logic [DATA_WIDTH-1:0] doutb
);

    // Memory array
    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Port A Operation (Synchronous)
    always_ff @(posedge clk) begin
        if (ena) begin
            if (wea) begin
                mem[addra] <= dina;
            end
            douta <= mem[addra];
        end
    end

    // Port B Operation (Synchronous)
    always_ff @(posedge clk) begin
        if (enb) begin
            doutb <= mem[addrb];
        end
    end

endmodule

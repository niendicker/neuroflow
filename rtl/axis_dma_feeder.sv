// NeuroFlow AXI4-Stream DMA Feeder & Drain Engine
// Manages high-bandwidth streaming between external AXI-Stream buses (Host DMA, PCIe, Ethernet)
// and on-chip SRAM Scratchpads (TCM).
// - s_axis_act: Stream in Activation matrix/vectors directly into Activation SRAM
// - s_axis_wgt: Stream in Weight matrix/vectors directly into Weight SRAM
// - m_axis_out: Stream out Result matrix/vectors from Output SRAM to Host

`timescale 1ns / 1ps

module axis_dma_feeder #(
    parameter DIM        = 4,
    parameter DATA_WIDTH = 32,
    parameter DEPTH      = 256,
    parameter ADDR_WIDTH = $clog2(DEPTH),
    parameter VEC_WIDTH  = DIM * DATA_WIDTH
)(
    input  logic                  clk,
    input  logic                  rst_n,

    // --- Activation Slave Stream (s_axis_act) ---
    input  logic [VEC_WIDTH-1:0]  s_axis_act_tdata,
    input  logic                  s_axis_act_tvalid,
    output logic                  s_axis_act_tready,
    input  logic                  s_axis_act_tlast,

    // SRAM Activation Port A Interface
    output logic                  act_sram_we,
    output logic                  act_sram_en,
    output logic [ADDR_WIDTH-1:0] act_sram_addr,
    output logic [VEC_WIDTH-1:0]  act_sram_wdata,
    input  logic [ADDR_WIDTH-1:0] act_base_addr,

    // --- Weight Slave Stream (s_axis_wgt) ---
    input  logic [VEC_WIDTH-1:0]  s_axis_wgt_tdata,
    input  logic                  s_axis_wgt_tvalid,
    output logic                  s_axis_wgt_tready,
    input  logic                  s_axis_wgt_tlast,

    // SRAM Weight Port A Interface
    output logic                  wgt_sram_we,
    output logic                  wgt_sram_en,
    output logic [ADDR_WIDTH-1:0] wgt_sram_addr,
    output logic [VEC_WIDTH-1:0]  wgt_sram_wdata,
    input  logic [ADDR_WIDTH-1:0] wgt_base_addr,

    // --- Output Master Stream (m_axis_out) ---
    output logic [VEC_WIDTH-1:0]  m_axis_out_tdata,
    output logic                  m_axis_out_tvalid,
    input  logic                  m_axis_out_tready,
    output logic                  m_axis_out_tlast,

    // SRAM Output Port B Interface (Read out to stream)
    output logic                  out_sram_enb,
    output logic [ADDR_WIDTH-1:0] out_sram_raddr,
    input  logic [VEC_WIDTH-1:0]  out_sram_rdata,
    input  logic [ADDR_WIDTH-1:0] out_base_addr,
    input  logic [ADDR_WIDTH-1:0] out_len, // number of rows/vectors to stream out
    input  logic                  start_stream_out,
    output logic                  stream_out_busy,
    output logic                  stream_out_done
);

    // -------------------------------------------------------------------------
    // Activation Stream Receiver
    // -------------------------------------------------------------------------
    logic [ADDR_WIDTH-1:0] act_wr_ptr;

    assign s_axis_act_tready = 1'b1; // Always ready to receive into SRAM buffer
    assign act_sram_en       = s_axis_act_tvalid;
    assign act_sram_we       = s_axis_act_tvalid;
    assign act_sram_addr     = act_base_addr + act_wr_ptr;
    assign act_sram_wdata    = s_axis_act_tdata;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_wr_ptr <= '0;
        end else if (s_axis_act_tvalid && s_axis_act_tready) begin
            if (s_axis_act_tlast) begin
                act_wr_ptr <= '0;
            end else begin
                act_wr_ptr <= act_wr_ptr + 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Weight Stream Receiver
    // -------------------------------------------------------------------------
    logic [ADDR_WIDTH-1:0] wgt_wr_ptr;

    assign s_axis_wgt_tready = 1'b1; // Always ready to receive into SRAM buffer
    assign wgt_sram_en       = s_axis_wgt_tvalid;
    assign wgt_sram_we       = s_axis_wgt_tvalid;
    assign wgt_sram_addr     = wgt_base_addr + wgt_wr_ptr;
    assign wgt_sram_wdata    = s_axis_wgt_tdata;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wgt_wr_ptr <= '0;
        end else if (s_axis_wgt_tvalid && s_axis_wgt_tready) begin
            if (s_axis_wgt_tlast) begin
                wgt_wr_ptr <= '0;
            end else begin
                wgt_wr_ptr <= wgt_wr_ptr + 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Output Stream Transmitter (FSM)
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        SO_IDLE  = 2'b00,
        SO_READ  = 2'b01,
        SO_BURST = 2'b10,
        SO_DONE  = 2'b11
    } stream_out_state_t;

    stream_out_state_t so_state;
    logic [ADDR_WIDTH-1:0] out_rd_ptr;
    logic [ADDR_WIDTH-1:0] out_total_len;

    assign stream_out_busy = (so_state != SO_IDLE);
    assign stream_out_done = (so_state == SO_DONE);

    assign out_sram_enb   = (so_state == SO_READ) || (so_state == SO_BURST && m_axis_out_tready);
    assign out_sram_raddr = out_base_addr + out_rd_ptr;

    assign m_axis_out_tdata = out_sram_rdata;
    assign m_axis_out_tvalid = (so_state == SO_BURST);
    assign m_axis_out_tlast  = (so_state == SO_BURST) && (out_rd_ptr == out_total_len);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            so_state      <= SO_IDLE;
            out_rd_ptr    <= '0;
            out_total_len <= '0;
        end else begin
            case (so_state)
                SO_IDLE: begin
                    if (start_stream_out && (out_len > 0)) begin
                        out_total_len <= out_len;
                        out_rd_ptr    <= '0;
                        so_state      <= SO_READ;
                    end
                end

                SO_READ: begin
                    // 1-cycle pipeline delay for SRAM synchronous read
                    out_rd_ptr <= out_rd_ptr + 1'b1;
                    so_state   <= SO_BURST;
                end

                SO_BURST: begin
                    if (m_axis_out_tready) begin
                        if (out_rd_ptr == out_total_len) begin
                            so_state <= SO_DONE;
                        end else begin
                            out_rd_ptr <= out_rd_ptr + 1'b1;
                        end
                    end
                end

                SO_DONE: begin
                    so_state <= SO_IDLE;
                end

                default: so_state <= SO_IDLE;
            endcase
        end
    end

endmodule

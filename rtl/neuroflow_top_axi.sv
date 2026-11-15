// NeuroFlow Top-Level System-on-Chip (SoC) Accelerator Wrapper
// Integrates:
// - Standard AXI4-Lite CSR (Memory-Mapped Host Control)
// - Tri-Bank Dual-Port SRAM Scratchpad (TCM: Activations, Weights, Outputs)
// - Dual Hardware Skew Buffers (Automatic Systolic Wavefront Diagonal Delay)
// - NeuroFlow Core Complex (2D Systolic Array Mesh + SIMD Vector Engine)
// - AXI4-Stream High-Throughput Feeder & Drain Engine

`timescale 1ns / 1ps

module neuroflow_top_axi #(
    parameter DIM            = 4,
    parameter DATA_WIDTH     = 32,
    parameter SRAM_DEPTH     = 256,
    parameter AXI_ADDR_WIDTH = 8,
    localparam VEC_WIDTH     = DIM * DATA_WIDTH,
    localparam SRAM_ADDR_W   = $clog2(SRAM_DEPTH)
)(
    input  logic                      clk,
    input  logic                      rst_n,

    // =========================================================================
    // AXI4-Lite Slave (Control & Status Registers)
    // =========================================================================
    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,
    input  logic [31:0]               s_axi_wdata,
    input  logic [3:0]                s_axi_wstrb,
    input  logic                      s_axi_wvalid,
    output logic                      s_axi_wready,
    output logic [1:0]                s_axi_bresp,
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,
    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,
    output logic [31:0]               s_axi_rdata,
    output logic [1:0]                s_axi_rresp,
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready,

    // =========================================================================
    // AXI4-Stream Slave: Activation Input Stream
    // =========================================================================
    input  logic [VEC_WIDTH-1:0]      s_axis_act_tdata,
    input  logic                      s_axis_act_tvalid,
    output logic                      s_axis_act_tready,
    input  logic                      s_axis_act_tlast,

    // =========================================================================
    // AXI4-Stream Slave: Weight Input Stream
    // =========================================================================
    input  logic [VEC_WIDTH-1:0]      s_axis_wgt_tdata,
    input  logic                      s_axis_wgt_tvalid,
    output logic                      s_axis_wgt_tready,
    input  logic                      s_axis_wgt_tlast,

    // =========================================================================
    // AXI4-Stream Master: Result Output Stream
    // =========================================================================
    output logic [VEC_WIDTH-1:0]      m_axis_out_tdata,
    output logic                      m_axis_out_tvalid,
    input  logic                      m_axis_out_tready,
    output logic                      m_axis_out_tlast,

    // Interrupt
    output logic                      irq_done
);

    // -------------------------------------------------------------------------
    // CSR Control & Status Interconnect
    // -------------------------------------------------------------------------
    logic                   reg_start;
    logic                   reg_abort;
    logic                   reg_clr_acc;
    logic                   reg_tensor_en;
    logic                   reg_vec_en;
    logic [1:0]             reg_prec;
    logic                   reg_vec_src_sel;
    logic [2:0]             reg_vec_op;
    logic [15:0]            reg_matrix_k;
    logic [15:0]            reg_act_base_addr;
    logic [15:0]            reg_wgt_base_addr;
    logic [15:0]            reg_out_base_addr;
    logic [$clog2(DIM)-1:0] reg_read_row_idx;

    logic                   fsm_busy;
    logic                   fsm_done;
    logic [31:0]            cycle_counter;

    assign irq_done = fsm_done;

    axi_lite_csr #(
        .ADDR_WIDTH(AXI_ADDR_WIDTH),
        .DATA_WIDTH(32),
        .DIM(DIM)
    ) u_csr (
        .clk               (clk),
        .rst_n             (rst_n),
        .s_axi_awaddr      (s_axi_awaddr),
        .s_axi_awvalid     (s_axi_awvalid),
        .s_axi_awready     (s_axi_awready),
        .s_axi_wdata       (s_axi_wdata),
        .s_axi_wstrb       (s_axi_wstrb),
        .s_axi_wvalid      (s_axi_wvalid),
        .s_axi_wready      (s_axi_wready),
        .s_axi_bresp       (s_axi_bresp),
        .s_axi_bvalid      (s_axi_bvalid),
        .s_axi_bready      (s_axi_bready),
        .s_axi_araddr      (s_axi_araddr),
        .s_axi_arvalid     (s_axi_arvalid),
        .s_axi_arready     (s_axi_arready),
        .s_axi_rdata       (s_axi_rdata),
        .s_axi_rresp       (s_axi_rresp),
        .s_axi_rvalid      (s_axi_rvalid),
        .s_axi_rready      (s_axi_rready),
        .reg_start         (reg_start),
        .reg_abort         (reg_abort),
        .reg_clr_acc       (reg_clr_acc),
        .reg_tensor_en     (reg_tensor_en),
        .reg_vec_en        (reg_vec_en),
        .reg_prec          (reg_prec),
        .reg_vec_src_sel   (reg_vec_src_sel),
        .reg_vec_op        (reg_vec_op),
        .reg_matrix_k      (reg_matrix_k),
        .reg_act_base_addr (reg_act_base_addr),
        .reg_wgt_base_addr (reg_wgt_base_addr),
        .reg_out_base_addr (reg_out_base_addr),
        .reg_read_row_idx  (reg_read_row_idx),
        .status_busy       (fsm_busy),
        .status_done       (fsm_done),
        .perf_cycle_count  (cycle_counter)
    );

    // -------------------------------------------------------------------------
    // SRAM Scratchpad Memory Banks
    // -------------------------------------------------------------------------
    // Bank 1: Activation SRAM
    logic                   act_sram_wea;
    logic                   act_sram_ena;
    logic [SRAM_ADDR_W-1:0] act_sram_addra;
    logic [VEC_WIDTH-1:0]   act_sram_dina;
    logic [VEC_WIDTH-1:0]   act_sram_douta;

    logic                   act_sram_enb;
    logic [SRAM_ADDR_W-1:0] act_sram_addrb;
    logic [VEC_WIDTH-1:0]   act_sram_doutb;

    sram_scratchpad #(
        .DATA_WIDTH(VEC_WIDTH),
        .DEPTH(SRAM_DEPTH)
    ) u_sram_act (
        .clk   (clk),
        .rst_n (rst_n),
        .wea   (act_sram_wea),
        .ena   (act_sram_ena),
        .addra (act_sram_addra),
        .dina  (act_sram_dina),
        .douta (act_sram_douta),
        .enb   (act_sram_enb),
        .addrb (act_sram_addrb),
        .doutb (act_sram_doutb)
    );

    // Bank 2: Weight SRAM
    logic                   wgt_sram_wea;
    logic                   wgt_sram_ena;
    logic [SRAM_ADDR_W-1:0] wgt_sram_addra;
    logic [VEC_WIDTH-1:0]   wgt_sram_dina;
    logic [VEC_WIDTH-1:0]   wgt_sram_douta;

    logic                   wgt_sram_enb;
    logic [SRAM_ADDR_W-1:0] wgt_sram_addrb;
    logic [VEC_WIDTH-1:0]   wgt_sram_doutb;

    sram_scratchpad #(
        .DATA_WIDTH(VEC_WIDTH),
        .DEPTH(SRAM_DEPTH)
    ) u_sram_wgt (
        .clk   (clk),
        .rst_n (rst_n),
        .wea   (wgt_sram_wea),
        .ena   (wgt_sram_ena),
        .addra (wgt_sram_addra),
        .dina  (wgt_sram_dina),
        .douta (wgt_sram_douta),
        .enb   (wgt_sram_enb),
        .addrb (wgt_sram_addrb),
        .doutb (wgt_sram_doutb)
    );

    // Bank 3: Output SRAM
    logic                   out_sram_wea;
    logic                   out_sram_ena;
    logic [SRAM_ADDR_W-1:0] out_sram_addra;
    logic [VEC_WIDTH-1:0]   out_sram_dina;
    logic [VEC_WIDTH-1:0]   out_sram_douta;

    logic                   out_sram_enb;
    logic [SRAM_ADDR_W-1:0] out_sram_addrb;
    logic [VEC_WIDTH-1:0]   out_sram_doutb;

    sram_scratchpad #(
        .DATA_WIDTH(VEC_WIDTH),
        .DEPTH(SRAM_DEPTH)
    ) u_sram_out (
        .clk   (clk),
        .rst_n (rst_n),
        .wea   (out_sram_wea),
        .ena   (out_sram_ena),
        .addra (out_sram_addra),
        .dina  (out_sram_dina),
        .douta (out_sram_douta),
        .enb   (out_sram_enb),
        .addrb (out_sram_addrb),
        .doutb (out_sram_doutb)
    );

    // -------------------------------------------------------------------------
    // AXI-Stream DMA Feeder & Drain Engine
    // -------------------------------------------------------------------------
    logic start_stream_out;
    logic stream_out_busy;
    logic stream_out_done;

    axis_dma_feeder #(
        .DIM(DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(SRAM_DEPTH)
    ) u_axis_dma (
        .clk               (clk),
        .rst_n             (rst_n),
        .s_axis_act_tdata  (s_axis_act_tdata),
        .s_axis_act_tvalid (s_axis_act_tvalid),
        .s_axis_act_tready (s_axis_act_tready),
        .s_axis_act_tlast  (s_axis_act_tlast),
        .act_sram_we       (act_sram_wea),
        .act_sram_en       (act_sram_ena),
        .act_sram_addr     (act_sram_addra),
        .act_sram_wdata    (act_sram_dina),
        .act_base_addr     (reg_act_base_addr[SRAM_ADDR_W-1:0]),
        .s_axis_wgt_tdata  (s_axis_wgt_tdata),
        .s_axis_wgt_tvalid (s_axis_wgt_tvalid),
        .s_axis_wgt_tready (s_axis_wgt_tready),
        .s_axis_wgt_tlast  (s_axis_wgt_tlast),
        .wgt_sram_we       (wgt_sram_wea),
        .wgt_sram_en       (wgt_sram_ena),
        .wgt_sram_addr     (wgt_sram_addra),
        .wgt_sram_wdata    (wgt_sram_dina),
        .wgt_base_addr     (reg_wgt_base_addr[SRAM_ADDR_W-1:0]),
        .m_axis_out_tdata  (m_axis_out_tdata),
        .m_axis_out_tvalid (m_axis_out_tvalid),
        .m_axis_out_tready (m_axis_out_tready),
        .m_axis_out_tlast  (m_axis_out_tlast),
        .out_sram_enb      (out_sram_enb),
        .out_sram_raddr    (out_sram_addrb),
        .out_sram_rdata    (out_sram_doutb),
        .out_base_addr     (reg_out_base_addr[SRAM_ADDR_W-1:0]),
        .out_len           (SRAM_ADDR_W'(DIM)),
        .start_stream_out  (start_stream_out),
        .stream_out_busy   (stream_out_busy),
        .stream_out_done   (stream_out_done)
    );

    // -------------------------------------------------------------------------
    // Hardware Skew Buffers (West Activations & North Weights)
    // -------------------------------------------------------------------------
    logic skew_en;
    logic skew_clr;

    logic [DIM-1:0][DATA_WIDTH-1:0] act_unskewed;
    logic [DIM-1:0][DATA_WIDTH-1:0] act_skewed;
    logic [DIM-1:0][DATA_WIDTH-1:0] wgt_unskewed;
    logic [DIM-1:0][DATA_WIDTH-1:0] wgt_skewed;

    // Gate SRAM outputs so only valid feeding cycles inject non-zero data into array
    logic feed_active;
    assign feed_active = (state == ST_FEED);

    assign act_unskewed = feed_active ? act_sram_doutb : '0;
    assign wgt_unskewed = feed_active ? wgt_sram_doutb : '0;

    skew_buffer #(
        .DIM(DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_skew_act (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (skew_en),
        .clr   (skew_clr),
        .din   (act_unskewed),
        .dout  (act_skewed)
    );

    skew_buffer #(
        .DIM(DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_skew_wgt (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (skew_en),
        .clr   (skew_clr),
        .din   (wgt_unskewed),
        .dout  (wgt_skewed)
    );

    // -------------------------------------------------------------------------
    // NeuroFlow Core Complex (Systolic Array + SIMD Vector Engine)
    // -------------------------------------------------------------------------
    logic                           core_tensor_en;
    logic                           core_clr_acc;
    logic [$clog2(DIM)-1:0]         core_read_row_idx;
    logic [DIM-1:0][DATA_WIDTH-1:0] core_row_out;

    logic                           core_vec_en;
    logic [DIM-1:0][DATA_WIDTH-1:0] core_vec_vs1_in;
    logic [DIM-1:0][DATA_WIDTH-1:0] core_vec_vs2_in;
    logic [DIM-1:0][DATA_WIDTH-1:0] core_vec_vd_out;
    logic                           core_vec_valid_out;

    // Row index for systolic accumulator extraction is driven directly by drain counter
    assign core_read_row_idx = drain_row_counter[$clog2(DIM)-1:0];

    neuroflow_core #(
        .DIM(DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_core (
        .clk           (clk),
        .rst_n         (rst_n),
        .tensor_en     (core_tensor_en),
        .clr_acc       (core_clr_acc),
        .prec          (reg_prec),
        .read_row_idx  (core_read_row_idx),
        .act_west_in   (act_skewed),
        .wgt_north_in  (wgt_skewed),
        .row_out       (core_row_out),
        .vec_en        (core_vec_en),
        .vec_op        (reg_vec_op),
        .vec_src_sel   (reg_vec_src_sel),
        .vec_vs1_in    (core_vec_vs1_in),
        .vec_vs2_in    (core_vec_vs2_in),
        .vec_vd_out    (core_vec_vd_out),
        .vec_valid_out (core_vec_valid_out)
    );

    // -------------------------------------------------------------------------
    // Autonomous Execution Orchestrator FSM
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        ST_IDLE      = 3'b000,
        ST_CLEAR     = 3'b001,
        ST_FEED      = 3'b010,
        ST_WAIT_PIPE = 3'b011,
        ST_DRAIN     = 3'b100,
        ST_STREAM    = 3'b101,
        ST_DONE      = 3'b110
    } state_t;

    state_t state;

    logic [15:0] feed_counter;
    logic [15:0] pipe_latency_counter;
    logic [$clog2(DIM):0] drain_row_counter; // one extra bit to check == DIM

    // Calculation of total pipeline latency:
    // (DIM - 1) cycles skew + 2 cycles PE MAC pipeline + (DIM) systolic forwarding
    localparam int TOTAL_PIPE_LATENCY = (DIM - 1) + 2 + DIM;

    assign fsm_busy = (state != ST_IDLE);
    assign fsm_done = (state == ST_DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                <= ST_IDLE;
            cycle_counter        <= '0;
            feed_counter         <= '0;
            pipe_latency_counter <= '0;
            drain_row_counter    <= '0;

            act_sram_enb         <= 1'b0;
            act_sram_addrb       <= '0;
            wgt_sram_enb         <= 1'b0;
            wgt_sram_addrb       <= '0;

            out_sram_ena         <= 1'b0;
            out_sram_wea         <= 1'b0;
            out_sram_addra       <= '0;
            out_sram_dina        <= '0;

            skew_en              <= 1'b0;
            skew_clr             <= 1'b0;
            core_tensor_en       <= 1'b0;
            core_clr_acc         <= 1'b0;
            core_vec_en          <= 1'b0;
            core_vec_vs1_in      <= '0;
            core_vec_vs2_in      <= '0;
            start_stream_out     <= 1'b0;
        end else begin
            // Free-running cycle counter when busy
            if (fsm_busy) begin
                cycle_counter <= cycle_counter + 1'b1;
            end else if (reg_start) begin
                cycle_counter <= '0;
            end

            // Default strobes
            core_clr_acc     <= 1'b0;
            skew_clr         <= 1'b0;
            out_sram_wea     <= 1'b0;
            out_sram_ena     <= 1'b0;
            start_stream_out <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (reg_start) begin
                        state        <= ST_CLEAR;
                        feed_counter <= '0;
                    end
                end

                ST_CLEAR: begin
                    // 1 cycle to clear accumulators and skew registers
                    core_clr_acc <= 1'b1;
                    skew_clr     <= 1'b1;
                    state        <= ST_FEED;

                    // Prime first SRAM read
                    act_sram_enb   <= 1'b1;
                    act_sram_addrb <= reg_act_base_addr[SRAM_ADDR_W-1:0];
                    wgt_sram_enb   <= 1'b1;
                    wgt_sram_addrb <= reg_wgt_base_addr[SRAM_ADDR_W-1:0];
                    skew_en        <= 1'b1;
                    core_tensor_en <= 1'b1;
                end

                ST_FEED: begin
                    skew_en        <= 1'b1;
                    core_tensor_en <= 1'b1;
                    feed_counter   <= feed_counter + 1'b1;

                    // Address generation: pre-fetch next address until reg_matrix_k - 1
                    if (feed_counter + 1'b1 < reg_matrix_k) begin
                        act_sram_enb   <= 1'b1;
                        act_sram_addrb <= reg_act_base_addr[SRAM_ADDR_W-1:0] + SRAM_ADDR_W'(feed_counter + 1'b1);
                        wgt_sram_enb   <= 1'b1;
                        wgt_sram_addrb <= reg_wgt_base_addr[SRAM_ADDR_W-1:0] + SRAM_ADDR_W'(feed_counter + 1'b1);
                    end else begin
                        act_sram_enb   <= 1'b0;
                        wgt_sram_enb   <= 1'b0;
                    end

                    // Transition to ST_WAIT_PIPE once all K tokens have been presented to the skew buffer
                    if (feed_counter == reg_matrix_k) begin
                        pipe_latency_counter <= '0;
                        state                <= ST_WAIT_PIPE;
                    end
                end

                ST_WAIT_PIPE: begin
                    // Keep systolic clocks active so wavefront reaches all PEs
                    skew_en        <= 1'b1;
                    core_tensor_en <= 1'b1;
                    pipe_latency_counter <= pipe_latency_counter + 1'b1;

                    if (pipe_latency_counter >= 16'(TOTAL_PIPE_LATENCY)) begin
                        skew_en           <= 1'b0;
                        core_tensor_en    <= 1'b0;
                        drain_row_counter <= '0;
                        state             <= ST_DRAIN;
                    end
                end

                ST_DRAIN: begin
                    // Read out each row of the systolic array and save to Output SRAM
                    if (drain_row_counter < DIM) begin
                        // Write current row to Output SRAM
                        out_sram_ena   <= 1'b1;
                        out_sram_wea   <= 1'b1;
                        out_sram_addra <= reg_out_base_addr[SRAM_ADDR_W-1:0] + SRAM_ADDR_W'(drain_row_counter);
                        out_sram_dina  <= (reg_vec_en) ? core_vec_vd_out : core_row_out;

                        drain_row_counter <= drain_row_counter + 1'b1;
                    end else begin
                        out_sram_ena     <= 1'b0;
                        out_sram_wea     <= 1'b0;
                        start_stream_out <= 1'b1; // Trigger AXI-Stream drain
                        state            <= ST_STREAM;
                    end
                end

                ST_STREAM: begin
                    // Wait for AXI-Stream DMA to finish transmitting result matrix to host
                    if (stream_out_done || !stream_out_busy) begin
                        state <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    // Remains in DONE until next START or reset
                    if (reg_start) begin
                        state        <= ST_CLEAR;
                        feed_counter <= '0;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

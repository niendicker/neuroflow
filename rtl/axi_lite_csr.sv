// NeuroFlow AXI4-Lite Control & Status Registers (CSR)
// Standard ARM AMBA AXI4-Lite Slave Interface (32-bit)
// Provides host CPU memory-mapped registers for orchestration, profiling, and configuration.

`timescale 1ns / 1ps

module axi_lite_csr #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 32,
    parameter DIM        = 4
)(
    input  logic                   clk,
    input  logic                   rst_n,

    // AXI4-Lite Write Address Channel
    input  logic [ADDR_WIDTH-1:0]  s_axi_awaddr,
    input  logic                   s_axi_awvalid,
    output logic                   s_axi_awready,

    // AXI4-Lite Write Data Channel
    input  logic [DATA_WIDTH-1:0]  s_axi_wdata,
    input  logic [3:0]             s_axi_wstrb,
    input  logic                   s_axi_wvalid,
    output logic                   s_axi_wready,

    // AXI4-Lite Write Response Channel
    output logic [1:0]             s_axi_bresp,
    output logic                   s_axi_bvalid,
    input  logic                   s_axi_bready,

    // AXI4-Lite Read Address Channel
    input  logic [ADDR_WIDTH-1:0]  s_axi_araddr,
    input  logic                   s_axi_arvalid,
    output logic                   s_axi_arready,

    // AXI4-Lite Read Data Channel
    output logic [DATA_WIDTH-1:0]  s_axi_rdata,
    output logic [1:0]             s_axi_rresp,
    output logic                   s_axi_rvalid,
    input  logic                   s_axi_rready,

    // Internal Register Outputs to Compute / Memory Engine
    output logic                   reg_start,
    output logic                   reg_abort,
    output logic                   reg_clr_acc,
    output logic                   reg_tensor_en,
    output logic                   reg_vec_en,
    output logic [1:0]             reg_prec,
    output logic                   reg_vec_src_sel,
    output logic [2:0]             reg_vec_op,
    output logic [15:0]            reg_matrix_k,
    output logic [15:0]            reg_act_base_addr,
    output logic [15:0]            reg_wgt_base_addr,
    output logic [15:0]            reg_out_base_addr,
    output logic [$clog2(DIM)-1:0] reg_read_row_idx,

    // Status Inputs from Execution FSM
    input  logic                   status_busy,
    input  logic                   status_done,
    input  logic [31:0]            perf_cycle_count
);

    // Register Offsets
    localparam ADDR_CONTROL      = 8'h00;
    localparam ADDR_STATUS       = 8'h04;
    localparam ADDR_CONFIG       = 8'h08;
    localparam ADDR_INFO         = 8'h0C;
    localparam ADDR_MATRIX_K     = 8'h10;
    localparam ADDR_ACT_BASE     = 8'h14;
    localparam ADDR_WGT_BASE     = 8'h18;
    localparam ADDR_OUT_BASE     = 8'h1C;
    localparam ADDR_READ_ROW     = 8'h20;
    localparam ADDR_PERF_CYCLES  = 8'h24;

    // Registers Storage
    logic [31:0] r_control;
    logic [31:0] r_config;
    logic [31:0] r_matrix_k;
    logic [31:0] r_act_base;
    logic [31:0] r_wgt_base;
    logic [31:0] r_out_base;
    logic [31:0] r_read_row;

    // Self-clearing start pulse
    logic start_pulse;
    assign reg_start = start_pulse;

    // Register Mapping
    assign reg_abort         = r_control[1];
    assign reg_clr_acc       = r_control[2];
    assign reg_tensor_en     = r_control[3];
    assign reg_vec_en        = r_control[4];

    assign reg_prec          = r_config[1:0];
    assign reg_vec_src_sel   = r_config[2];
    assign reg_vec_op        = r_config[5:3];

    assign reg_matrix_k      = r_matrix_k[15:0];
    assign reg_act_base_addr = r_act_base[15:0];
    assign reg_wgt_base_addr = r_wgt_base[15:0];
    assign reg_out_base_addr = r_out_base[15:0];
    assign reg_read_row_idx  = r_read_row[$clog2(DIM)-1:0];

    // Status and Info Read-Only Signals
    logic [31:0] r_status;
    assign r_status = {30'd0, status_done, status_busy};

    logic [31:0] r_info;
    assign r_info = {8'h00, 8'(DATA_WIDTH), 8'(DIM), 8'h10}; // [HW_VER=1.0, DIM, DATA_WIDTH, 0]

    // -------------------------------------------------------------------------
    // AXI-Lite Write Channels
    // -------------------------------------------------------------------------
    logic [ADDR_WIDTH-1:0] axi_awaddr;
    logic                  aw_en;

    assign s_axi_bresp = 2'b00; // OKAY

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            aw_en         <= 1'b1;
            axi_awaddr    <= '0;
            r_control     <= 32'h00000008; // tensor_en default 1
            r_config      <= '0;
            r_matrix_k    <= 32'd4;
            r_act_base    <= '0;
            r_wgt_base    <= '0;
            r_out_base    <= '0;
            r_read_row    <= '0;
            start_pulse   <= 1'b0;
        end else begin
            start_pulse <= 1'b0; // 1-cycle auto-clear

            // Write Address Ready Handshake
            if (~s_axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                s_axi_awready <= 1'b1;
                s_axi_wready  <= 1'b1;
                axi_awaddr    <= s_axi_awaddr;
                aw_en         <= 1'b0;
            end else begin
                s_axi_awready <= 1'b0;
                s_axi_wready  <= 1'b0;
            end

            // Write Data Execution
            if (s_axi_awready && s_axi_wready) begin
                case (axi_awaddr)
                    ADDR_CONTROL: begin
                        r_control <= s_axi_wdata;
                        if (s_axi_wdata[0]) begin
                            start_pulse <= 1'b1; // Trigger start pulse
                        end
                    end
                    ADDR_CONFIG:      r_config      <= s_axi_wdata;
                    ADDR_MATRIX_K:   r_matrix_k    <= s_axi_wdata;
                    ADDR_ACT_BASE:   r_act_base    <= s_axi_wdata;
                    ADDR_WGT_BASE:   r_wgt_base    <= s_axi_wdata;
                    ADDR_OUT_BASE:   r_out_base    <= s_axi_wdata;
                    ADDR_READ_ROW:   r_read_row    <= s_axi_wdata;
                    default: ;
                endcase
                s_axi_bvalid <= 1'b1;
            end

            // Write Response Handshake
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
                aw_en        <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // AXI-Lite Read Channels
    // -------------------------------------------------------------------------
    logic [ADDR_WIDTH-1:0] axi_araddr;
    assign s_axi_rresp = 2'b00; // OKAY

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= '0;
            axi_araddr    <= '0;
        end else begin
            // Read Address Handshake
            if (~s_axi_arready && s_axi_arvalid) begin
                s_axi_arready <= 1'b1;
                axi_araddr    <= s_axi_araddr;
            end else begin
                s_axi_arready <= 1'b0;
            end

            // Read Data Generation
            if (s_axi_arready && s_axi_arvalid && ~s_axi_rvalid) begin
                s_axi_rvalid <= 1'b1;
                case (axi_araddr)
                    ADDR_CONTROL:     s_axi_rdata <= r_control;
                    ADDR_STATUS:      s_axi_rdata <= r_status;
                    ADDR_CONFIG:      s_axi_rdata <= r_config;
                    ADDR_INFO:        s_axi_rdata <= r_info;
                    ADDR_MATRIX_K:    s_axi_rdata <= r_matrix_k;
                    ADDR_ACT_BASE:    s_axi_rdata <= r_act_base;
                    ADDR_WGT_BASE:    s_axi_rdata <= r_wgt_base;
                    ADDR_OUT_BASE:    s_axi_rdata <= r_out_base;
                    ADDR_READ_ROW:    s_axi_rdata <= r_read_row;
                    ADDR_PERF_CYCLES: s_axi_rdata <= perf_cycle_count;
                    default:          s_axi_rdata <= 32'hDEADBEEF;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule

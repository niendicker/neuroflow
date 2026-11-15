// Verilator C++ Testbench for NeuroFlow Top AXI SoC Integration
// Validates:
// 1. AXI4-Lite Control & Status Registers (CSR) read/write handshake
// 2. AXI4-Stream Activation and Weight DMA Loading into Dual-Port SRAM (TCM)
// 3. Hardware Skew Buffer Wavefront generation
// 4. Autonomous Systolic Array GEMM Execution & Pipeline Drain
// 5. AXI4-Stream Result DMA Drain and bit-exact software validation

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <cassert>
#include "Vneuroflow_top_axi.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

const int DIM = 4;

// CSR Register Offsets
const uint32_t CSR_CONTROL      = 0x00;
const uint32_t CSR_STATUS       = 0x04;
const uint32_t CSR_CONFIG       = 0x08;
const uint32_t CSR_INFO         = 0x0C;
const uint32_t CSR_MATRIX_K     = 0x10;
const uint32_t CSR_ACT_BASE     = 0x14;
const uint32_t CSR_WGT_BASE     = 0x18;
const uint32_t CSR_OUT_BASE     = 0x1C;
const uint32_t CSR_READ_ROW     = 0x20;
const uint32_t CSR_PERF_CYCLES  = 0x24;

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    auto top = std::make_unique<Vneuroflow_top_axi>();
    auto tfp = std::make_unique<VerilatedVcdC>();

    top->trace(tfp.get(), 99);
    tfp->open("top_axi_waveform.vcd");

    std::cout << "============================================================" << std::endl;
    std::cout << "   NeuroFlow Top AXI SoC Bus & Memory Verification (Verilator)" << std::endl;
    std::cout << "   - AXI4-Lite CSR Master/Slave                               " << std::endl;
    std::cout << "   - Dual-Port Scratchpad SRAM (TCM)                          " << std::endl;
    std::cout << "   - Hardware Diagonal Skew Buffers                           " << std::endl;
    std::cout << "   - Autonomous 4x4 Systolic Core Execution                   " << std::endl;
    std::cout << "   - AXI4-Stream Result Readout                               " << std::endl;
    std::cout << "============================================================" << std::endl;

    vluint64_t sim_time = 0;

    auto tick = [&]() {
        top->clk = 0;
        top->eval();
        tfp->dump(sim_time++);
        top->clk = 1;
        top->eval();
        tfp->dump(sim_time++);
    };

    // Helper: Reset DUT
    auto reset_dut = [&]() {
        top->rst_n = 0;
        top->s_axi_awvalid = 0;
        top->s_axi_wvalid = 0;
        top->s_axi_bready = 0;
        top->s_axi_arvalid = 0;
        top->s_axi_rready = 0;
        top->s_axis_act_tvalid = 0;
        top->s_axis_act_tlast = 0;
        top->s_axis_wgt_tvalid = 0;
        top->s_axis_wgt_tlast = 0;
        top->m_axis_out_tready = 0;

        for (int i = 0; i < 10; i++) tick();
        top->rst_n = 1;
        for (int i = 0; i < 5; i++) tick();
    };

    // Helper: AXI4-Lite Write
    auto axi_write = [&](uint32_t addr, uint32_t data) {
        top->s_axi_awaddr = addr;
        top->s_axi_awvalid = 1;
        top->s_axi_wdata = data;
        top->s_axi_wstrb = 0xF;
        top->s_axi_wvalid = 1;
        top->s_axi_bready = 1;

        tick();
        while (!top->s_axi_bvalid) {
            tick();
        }
        top->s_axi_awvalid = 0;
        top->s_axi_wvalid = 0;
        tick();
        top->s_axi_bready = 0;
    };

    // Helper: AXI4-Lite Read
    auto axi_read = [&](uint32_t addr) -> uint32_t {
        top->s_axi_araddr = addr;
        top->s_axi_arvalid = 1;
        top->s_axi_rready = 1;

        tick();
        while (!top->s_axi_rvalid) {
            tick();
        }
        uint32_t data = top->s_axi_rdata;
        top->s_axi_arvalid = 0;
        tick();
        top->s_axi_rready = 0;
        return data;
    };

    reset_dut();

    // 1. Verify AXI-Lite Read from INFO register
    std::cout << "\n[1] Reading CSR INFO register (0x0C)..." << std::endl;
    uint32_t info_val = axi_read(CSR_INFO);
    uint32_t hw_ver   = info_val & 0xFF;
    uint32_t dim_read = (info_val >> 8) & 0xFF;
    uint32_t dw_read  = (info_val >> 16) & 0xFF;
    std::cout << "    Hardware Version : 0x" << std::hex << hw_ver << std::dec << std::endl;
    std::cout << "    Systolic Mesh DIM: " << dim_read << "x" << dim_read << std::endl;
    std::cout << "    Data Width       : " << dw_read << " bits" << std::endl;
    assert(dim_read == 4);
    assert(dw_read == 32);

    // 2. Define Matrices for Verification
    // Matrix A (4x4):
    // [  1,  2,  3,  4 ]
    // [  5,  6,  7,  8 ]
    // [  9, 10, 11, 12 ]
    // [ 13, 14, 15, 16 ]
    int32_t A[DIM][DIM] = {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    // Matrix B (4x4):
    // [ 2, 0, 1, 0 ]
    // [ 0, 2, 0, 1 ]
    // [ 1, 0, 2, 0 ]
    // [ 0, 1, 0, 2 ]
    int32_t B[DIM][DIM] = {
        {2, 0, 1, 0},
        {0, 2, 0, 1},
        {1, 0, 2, 0},
        {0, 1, 0, 2}
    };

    // Golden Matrix C = A x B
    int32_t C_golden[DIM][DIM] = {0};
    for (int r = 0; r < DIM; r++) {
        for (int c = 0; c < DIM; c++) {
            for (int k = 0; k < DIM; k++) {
                C_golden[r][c] += A[r][k] * B[k][c];
            }
        }
    }

    std::cout << "\n[2] Golden Reference GEMM Result Matrix C (A x B):" << std::endl;
    for (int r = 0; r < DIM; r++) {
        std::cout << "    Row " << r << ": [";
        for (int c = 0; c < DIM; c++) {
            std::cout << std::setw(5) << C_golden[r][c] << (c < DIM-1 ? ", " : " ]\n");
        }
    }

    // 3. Stream Matrices via AXI-Stream into SRAM Scratchpads
    std::cout << "\n[3] Streaming Activation Matrix A into SRAM via s_axis_act..." << std::endl;
    // Step k: West needs column k of A: [A[0][k], A[1][k], A[2][k], A[3][k]]
    for (int k = 0; k < DIM; k++) {
        top->s_axis_act_tvalid = 1;
        top->s_axis_act_tdata[0] = A[0][k];
        top->s_axis_act_tdata[1] = A[1][k];
        top->s_axis_act_tdata[2] = A[2][k];
        top->s_axis_act_tdata[3] = A[3][k];
        top->s_axis_act_tlast = (k == DIM - 1) ? 1 : 0;

        tick();
        while (!top->s_axis_act_tready) {
            tick();
        }
    }
    top->s_axis_act_tvalid = 0;
    top->s_axis_act_tlast = 0;
    tick();

    std::cout << "[4] Streaming Weight Matrix B into SRAM via s_axis_wgt..." << std::endl;
    // Step k: North needs row k of B: [B[k][0], B[k][1], B[k][2], B[k][3]]
    for (int k = 0; k < DIM; k++) {
        top->s_axis_wgt_tvalid = 1;
        top->s_axis_wgt_tdata[0] = B[k][0];
        top->s_axis_wgt_tdata[1] = B[k][1];
        top->s_axis_wgt_tdata[2] = B[k][2];
        top->s_axis_wgt_tdata[3] = B[k][3];
        top->s_axis_wgt_tlast = (k == DIM - 1) ? 1 : 0;

        tick();
        while (!top->s_axis_wgt_tready) {
            tick();
        }
    }
    top->s_axis_wgt_tvalid = 0;
    top->s_axis_wgt_tlast = 0;
    tick();

    // 4. Configure Accelerator via AXI-Lite and Trigger Execution
    std::cout << "\n[5] Configuring Execution via AXI-Lite CSR..." << std::endl;
    axi_write(CSR_MATRIX_K, DIM);
    axi_write(CSR_ACT_BASE, 0);
    axi_write(CSR_WGT_BASE, 0);
    axi_write(CSR_OUT_BASE, 0);
    axi_write(CSR_CONFIG, 0x00); // INT32, direct systolic GEMM

    std::cout << "    Triggering Autonomous Execution (START=1)..." << std::endl;
    axi_write(CSR_CONTROL, 0x09); // START=1, TENSOR_EN=1

    // Enable receiver for output stream
    top->m_axis_out_tready = 1;

    // 5. Monitor Execution FSM
    std::cout << "\n[6] Waiting for Hardware Completion..." << std::endl;
    int timeout = 200;
    int received_rows = 0;
    int32_t C_hw[DIM][DIM] = {0};

    while (timeout-- > 0) {
        tick();

        // Capture AXI-Stream Output Rows
        if (top->m_axis_out_tvalid && top->m_axis_out_tready) {
            if (received_rows < DIM) {
                C_hw[received_rows][0] = top->m_axis_out_tdata[0];
                C_hw[received_rows][1] = top->m_axis_out_tdata[1];
                C_hw[received_rows][2] = top->m_axis_out_tdata[2];
                C_hw[received_rows][3] = top->m_axis_out_tdata[3];
                std::cout << "    Received Output Row " << received_rows 
                          << " via AXI-Stream: [" 
                          << std::setw(5) << C_hw[received_rows][0] << ", "
                          << std::setw(5) << C_hw[received_rows][1] << ", "
                          << std::setw(5) << C_hw[received_rows][2] << ", "
                          << std::setw(5) << C_hw[received_rows][3] << " ]"
                          << (top->m_axis_out_tlast ? " (TLAST)" : "") << std::endl;
                received_rows++;
            }
        }

        if (top->irq_done && received_rows >= DIM) {
            break;
        }
    }

    assert(timeout > 0);

    // Read Performance Counter
    uint32_t perf_cycles = axi_read(CSR_PERF_CYCLES);
    std::cout << "    Execution Finished in " << perf_cycles << " clock cycles!" << std::endl;

    // 6. Bit-Exact Result Verification
    std::cout << "\n[7] Verifying Silicon Output vs. Golden Reference:" << std::endl;
    bool mismatch = false;
    for (int r = 0; r < DIM; r++) {
        for (int c = 0; c < DIM; c++) {
            if (C_hw[r][c] != C_golden[r][c]) {
                std::cout << "    MISMATCH at (" << r << ", " << c << "): Expected "
                          << C_golden[r][c] << " | Got " << C_hw[r][c] << std::endl;
                mismatch = true;
            }
        }
    }

    if (!mismatch) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "   [PASSED] ALL 16 MATRIX ELEMENTS MATCH EXACTLY!            " << std::endl;
        std::cout << "   AXI4-Lite + SRAM TCM + Skew Buffer + Systolic + AXI-Stream " << std::endl;
        std::cout << "   Autonomous Memory Subsystem 100% OPERATIONAL!             " << std::endl;
        std::cout << "============================================================" << std::endl;
    } else {
        std::cout << "\n[FAILED] Output mismatch detected!" << std::endl;
        tfp->close();
        return 1;
    }

    tfp->close();
    return 0;
}

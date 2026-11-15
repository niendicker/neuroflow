// Verilator C++ Testbench for NeuroFlow Nano (4x4 Mesh = 16 PEs + 128-bit Vector Unit)
// Tests:
// 1. 4x4 Matrix Multiply with Systolic Wavefront Streaming
// 2. Direct 1-Cycle Forwarding of Accumulator Rows into SIMD Vector Unit (GELU / ADD)
// 3. VCD Waveform generation (nano_waveform.vcd)

#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include "Vneuroflow_nano.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

const int DIM = 4;

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    auto top = std::make_unique<Vneuroflow_nano>();
    auto tfp = std::make_unique<VerilatedVcdC>();

    top->trace(tfp.get(), 99);
    tfp->open("nano_waveform.vcd");

    std::cout << "============================================================" << std::endl;
    std::cout << "      NeuroFlow Nano Hardware Verification (Verilator)      " << std::endl;
    std::cout << "      Dimensions: 4x4 Mesh (16 PEs) + 128-bit Vector Engine " << std::endl;
    std::cout << "============================================================" << std::endl;

    // Test Matrices:
    // Matrix A: 4x4 with element value = 5
    // Matrix B: 4x4 with element value = 7
    // Expected GEMM Result: C = A x B = 4 * (5 * 7) = 140 per element
    uint32_t val_a = 5;
    uint32_t val_b = 7;
    uint32_t expected_gemm = DIM * (val_a * val_b); // 140

    vluint64_t sim_time = 0;

    auto tick = [&]() {
        top->clk = 0;
        top->eval();
        tfp->dump(sim_time++);
        top->clk = 1;
        top->eval();
        tfp->dump(sim_time++);
    };

    // 1. Reset
    top->rst_n = 0;
    top->tensor_en = 0;
    top->clr_acc = 1;
    top->prec = 0; // INT8 mode
    top->read_row_idx = 0;
    top->vec_en = 0;
    top->vec_op = 0;
    top->vec_src_sel = 0;

    for (int i = 0; i < 4; i++) {
        top->act_west_in[i] = 0;
        top->wgt_north_in[i] = 0;
    }

    tick();
    tick();
    top->rst_n = 1;
    top->clr_acc = 0;
    tick();

    std::cout << "[SIM] Starting Systolic Matrix Multiplication..." << std::endl;
    top->tensor_en = 1;

    // Systolic feed with temporal diagonal skew
    // Total latency for 4x4 systolic matrix multiply is 3 * DIM cycles = 12 cycles
    int total_cycles = 3 * DIM + 2;

    for (int cycle = 0; cycle < total_cycles; cycle++) {
        for (int r = 0; r < DIM; r++) {
            int k = cycle - r;
            if (k >= 0 && k < DIM) {
                top->act_west_in[r] = val_a;
            } else {
                top->act_west_in[r] = 0;
            }
        }

        for (int c = 0; c < DIM; c++) {
            int k = cycle - c;
            if (k >= 0 && k < DIM) {
                top->wgt_north_in[c] = val_b;
            } else {
                top->wgt_north_in[c] = 0;
            }
        }

        tick();
    }

    top->tensor_en = 0;
    tick();
    tick();

    // 2. Verify Systolic Accumulator Row Readouts
    std::cout << "[SIM] Verifying Systolic Array Matrix Outputs (Expected: " << expected_gemm << ")..." << std::endl;
    bool gemm_passed = true;

    for (int r = 0; r < DIM; r++) {
        top->read_row_idx = r;
        tick();

        std::cout << "  Row " << r << ": [ ";
        for (int c = 0; c < DIM; c++) {
            uint32_t actual = top->row_out[c];
            std::cout << actual << " ";
            if (actual != expected_gemm) {
                gemm_passed = false;
            }
        }
        std::cout << "]" << std::endl;
    }

    assert(gemm_passed && "Systolic GEMM verification failed!");
    std::cout << ">>> Phase 1 (GEMM 4x4 Matrix Multiply): PASSED ✅" << std::endl;

    // 3. Verify Direct 1-Cycle Pipelining into SIMD Vector Unit (ADD operation)
    std::cout << "[SIM] Testing direct row forwarding to Vector Engine (VADD with bias 10)..." << std::endl;
    top->read_row_idx = 0; // Select Row 0 (all 140)
    top->vec_src_sel = 0;  // 0: Feed directly from systolic row readout
    top->vec_op = 3;       // 3: VADD
    top->vec_en = 1;

    // Set vs2_in (bias vector) = 10
    top->vec_vs2_in[0] = 10;
    top->vec_vs2_in[1] = 10;
    top->vec_vs2_in[2] = 10;
    top->vec_vs2_in[3] = 10;

    tick();
    tick();

    std::cout << "  Vector Unit Output (vd_out): [ ";
    bool vec_passed = true;
    for (int c = 0; c < DIM; c++) {
        uint32_t actual = top->vec_vd_out[c];
        std::cout << actual << " ";
        // Expected: 140 + 10 = 150
        if (actual != 150) {
            vec_passed = false;
        }
    }
    std::cout << "]" << std::endl;

    assert(vec_passed && "Vector Unit verification failed!");
    std::cout << ">>> Phase 2 (Tensor -> Vector Unit Pipeline): PASSED ✅" << std::endl;

    tfp->close();
    std::cout << "============================================================" << std::endl;
    std::cout << "      All NeuroFlow Nano Tests PASSED Successfully!         " << std::endl;
    std::cout << "      Waveforms written to rtl/nano_waveform.vcd            " << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}

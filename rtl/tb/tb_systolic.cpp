// Verilator C++ Testbench for NeuroFlow 2D Systolic Array (16x16 PEs)
// Tests full 2D matrix multiplication with systolic wavefront streaming and VCD waveform dumping

#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include "Vsystolic_array.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

const int DIM = 16;

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    auto top = std::make_unique<Vsystolic_array>();
    auto tfp = std::make_unique<VerilatedVcdC>();

    top->trace(tfp.get(), 99);
    tfp->open("systolic_waveform.vcd");

    std::cout << "============================================================" << std::endl;
    std::cout << "     NeuroFlow 2D Systolic Array Hardware Simulation        " << std::endl;
    std::cout << "     Dimensions: 16x16 Mesh (256 Processing Elements)       " << std::endl;
    std::cout << "============================================================" << std::endl;

    // Test Matrices:
    // Matrix A: 16x16 filled with 2
    // Matrix B: 16x16 filled with 3
    // Expected Result: C = A x B = 16 * (2 * 3) = 96 per element
    std::vector<std::vector<uint32_t>> mat_a(DIM, std::vector<uint32_t>(DIM, 2));
    std::vector<std::vector<uint32_t>> mat_b(DIM, std::vector<uint32_t>(DIM, 3));
    uint32_t expected_val = 16 * (2 * 3); // 96

    vluint64_t sim_time = 0;

    auto tick = [&]() {
        top->clk = 0;
        top->eval();
        tfp->dump(sim_time++);
        top->clk = 1;
        top->eval();
        tfp->dump(sim_time++);
    };

    // 1. Reset and Clear Accumulators
    top->rst_n = 0;
    top->en = 0;
    top->clr_acc = 1;
    top->prec = 0; // INT8 mode
    tick();
    tick();

    top->rst_n = 1;
    top->clr_acc = 0;
    top->en = 1;

    // Total cycles needed for 16x16 systolic array to skew, compute, and drain:
    // DIM + DIM + DIM = 48 cycles
    int total_cycles = 3 * DIM + 2;

    std::cout << "\nStreaming 2D wavefronts across the 16x16 array (" << total_cycles << " cycles)..." << std::endl;

    for (int t = 0; t < total_cycles; ++t) {
        // Skewed input feeding:
        // Row r receives A[r][t - r] if 0 <= t - r < DIM, else 0
        for (int r = 0; r < DIM; ++r) {
            int k = t - r;
            if (k >= 0 && k < DIM) {
                top->act_west_in[r] = mat_a[r][k];
            } else {
                top->act_west_in[r] = 0;
            }
        }

        // Column c receives B[t - c][c] if 0 <= t - c < DIM, else 0
        for (int c = 0; c < DIM; ++c) {
            int k = t - c;
            if (k >= 0 && k < DIM) {
                top->wgt_north_in[c] = mat_b[k][c];
            } else {
                top->wgt_north_in[c] = 0;
            }
        }

        tick();
    }

    // 2. Read Out All 16 Rows via the 512-bit Extraction Bus
    std::cout << "\n--- Verifying Output Matrix C (Accumulators) ---" << std::endl;
    bool all_passed = true;

    for (int r = 0; r < DIM; ++r) {
        top->read_row_idx = r;
        top->eval();

        if (r == 0 || r == 15) {
            std::cout << "Row " << r << " (First 4 elements): [";
            for (int c = 0; c < 4; ++c) {
                std::cout << top->row_out_512[c] << (c < 3 ? ", " : "");
            }
            std::cout << " ...]" << std::endl;
        }

        for (int c = 0; c < DIM; ++c) {
            if (top->row_out_512[c] != expected_val) {
                std::cerr << "Mismatch at C[" << r << "][" << c << "]: expected "
                          << expected_val << ", got " << top->row_out_512[c] << std::endl;
                all_passed = false;
            }
        }
    }

    tfp->close();
    std::cout << "\nWaveform dumped to: systolic_waveform.vcd" << std::endl;

    if (all_passed) {
        std::cout << "\n[PASSED] 16x16 Systolic Array Hardware Mesh (256 PEs) 100% verified!" << std::endl;
        std::cout << "Total Hardware Compute: 4,096 MACs in 50 cycles (81.9 MACs/cycle sustained)" << std::endl;
        return 0;
    } else {
        std::cerr << "\n[FAILED] Output matrix mismatch!" << std::endl;
        return 1;
    }
}

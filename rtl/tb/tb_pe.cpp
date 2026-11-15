// Verilator C++ Testbench for NeuroFlow Processing Element (PE)
// Tests 2-Stage Pipelined MAC: Multiplier Stage -> Accumulator Stage

#include <iostream>
#include <memory>
#include <cassert>
#include "Vpe.h"
#include "verilated.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto top = std::make_unique<Vpe>();

    std::cout << "============================================================" << std::endl;
    std::cout << "     NeuroFlow RTL Hardware Simulation (Verilator)          " << std::endl;
    std::cout << "     Testing Module: pe.sv (2-Stage Pipelined MAC)          " << std::endl;
    std::cout << "============================================================" << std::endl;

    // Initialize reset
    top->clk = 0;
    top->rst_n = 0;
    top->en = 0;
    top->clr_acc = 1;
    top->eval();

    // Clock pulse: Apply reset
    top->clk = 1; top->eval();
    top->clk = 0; top->eval();

    // Release reset
    top->rst_n = 1;
    top->clr_acc = 0;
    top->en = 1;
    top->prec = 0; // INT8 mode

    // Feed activation = 4, weight = 5 -> Product = 20 per cycle
    top->act_in = 4;
    top->wgt_in = 5;

    // Run for 5 clock cycles
    for (int cycle = 1; cycle <= 5; ++cycle) {
        top->clk = 1; top->eval();
        top->clk = 0; top->eval();
        std::cout << "Cycle " << cycle << " | Act_out: " << top->act_out
                  << " | Wgt_out: " << top->wgt_out
                  << " | Acc_out: " << top->acc_out << std::endl;
    }

    // 1-Cycle Pipeline Drain: Disable inputs and pulse clock to accumulate the 5th product
    top->en = 0;
    top->clk = 1; top->eval();
    top->clk = 0; top->eval();
    std::cout << "Cycle 6 (Pipeline Drain) | Acc_out: " << top->acc_out << std::endl;

    // After 5 inputs of 4 * 5 = 20, Acc_out should equal 100
    uint32_t expected = 5 * (4 * 5);
    std::cout << "\nExpected Acc_out: " << expected << " | Got: " << top->acc_out << std::endl;

    if (top->acc_out == expected) {
        std::cout << "\n[PASSED] 2-Stage Pipelined PE RTL hardware verified successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "\n[FAILED] Mismatch in PE accumulator readout!" << std::endl;
        return 1;
    }
}

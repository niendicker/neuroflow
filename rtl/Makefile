# NeuroFlow RTL Simulation Makefile (Verilator)

VERILATOR = verilator
VERILATOR_FLAGS = -Wall -Wno-fatal --cc --exe --build -O3

all: sim_pe

sim_pe: pe.sv tb/tb_pe.cpp
	@echo "=== Compiling Processing Element (PE) with Verilator ==="
	$(VERILATOR) $(VERILATOR_FLAGS) pe.sv tb/tb_pe.cpp -o sim_pe
	@echo "=== Running Hardware Simulation ==="
	./obj_dir/sim_pe

clean:
	rm -rf obj_dir

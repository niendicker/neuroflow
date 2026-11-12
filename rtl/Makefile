# NeuroFlow RTL Simulation Makefile (Verilator)

VERILATOR = verilator
VERILATOR_FLAGS = -Wall -Wno-fatal --cc --exe --build -O3 --trace

all: sim_pe sim_systolic

sim_pe: pe.sv tb/tb_pe.cpp
	@echo "=== Compiling Processing Element (PE) with Verilator ==="
	$(VERILATOR) $(VERILATOR_FLAGS) pe.sv tb/tb_pe.cpp -o sim_pe
	@echo "=== Running PE Hardware Simulation ==="
	./obj_dir/sim_pe

sim_systolic: systolic_array.sv pe.sv tb/tb_systolic.cpp
	@echo "=== Compiling 16x16 Systolic Array Mesh (256 PEs) with Verilator ==="
	$(VERILATOR) $(VERILATOR_FLAGS) systolic_array.sv pe.sv tb/tb_systolic.cpp -o sim_systolic
	@echo "=== Running Systolic Array Hardware Simulation ==="
	./obj_dir/sim_systolic

clean:
	rm -rf obj_dir *.vcd

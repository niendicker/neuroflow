# NeuroFlow RTL Simulation & Synthesis Makefile (Verilator, Yosys, OpenLane)

VERILATOR = verilator
VERILATOR_FLAGS = -Wall -Wno-fatal --cc --exe --build -O3 --trace
YOSYS = yosys
DOCKER = docker

all: sim_pe sim_systolic sim_nano

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

sim_nano: neuroflow_nano.sv neuroflow_core.sv systolic_array.sv pe.sv vector_unit.sv tb/tb_nano.cpp
	@echo "=== Compiling NeuroFlow Nano (16 PEs + Vector Engine) with Verilator ==="
	$(VERILATOR) $(VERILATOR_FLAGS) neuroflow_nano.sv neuroflow_core.sv systolic_array.sv pe.sv vector_unit.sv tb/tb_nano.cpp -o sim_nano
	@echo "=== Running NeuroFlow Nano Hardware Simulation ==="
	./obj_dir/sim_nano

sim_top_axi: neuroflow_top_axi.sv axi_lite_csr.sv sram_scratchpad.sv axis_dma_feeder.sv skew_buffer.sv neuroflow_core.sv systolic_array.sv pe.sv vector_unit.sv tb/tb_top_axi.cpp
	@echo "=== Compiling NeuroFlow Top AXI SoC Subsystem with Verilator ==="
	$(VERILATOR) $(VERILATOR_FLAGS) neuroflow_top_axi.sv axi_lite_csr.sv sram_scratchpad.sv axis_dma_feeder.sv skew_buffer.sv neuroflow_core.sv systolic_array.sv pe.sv vector_unit.sv tb/tb_top_axi.cpp -o sim_top_axi
	@echo "=== Running NeuroFlow Top AXI Hardware Simulation ==="
	./obj_dir/sim_top_axi

# --- Fluxo 3: Síntese Lógica Open-Source com Yosys ---
synth_nano: synth_nano.ys neuroflow_nano.sv neuroflow_core.sv systolic_array.sv pe.sv vector_unit.sv
	@mkdir -p obj_dir
	@echo "=== Running Yosys Generic Logic Synthesis for NeuroFlow Nano (16 PEs) ==="
	$(YOSYS) -s synth_nano.ys

synth_pe: synth_pe.ys pe.sv
	@mkdir -p obj_dir
	@echo "=== Running Yosys Generic Logic Synthesis for PE ==="
	$(YOSYS) -s synth_pe.ys

synth_ice40: synth_ice40.ys pe.sv
	@mkdir -p obj_dir
	@echo "=== Synthesizing PE for Lattice iCE40 FPGA ==="
	$(YOSYS) -s synth_ice40.ys

synth_ecp5: synth_ecp5.ys pe.sv
	@mkdir -p obj_dir
	@echo "=== Synthesizing PE for Lattice ECP5 FPGA ==="
	$(YOSYS) -s synth_ecp5.ys

# --- Fluxo 3: Síntese Física ASIC com OpenLane (SkyWater 130nm) ---
OPENLANE = ../.venv/bin/openlane

asic_pe:
	@echo "=== Running OpenLane 2 ASIC Tapeout Flow for PE (Sky130) ==="
	$(OPENLANE) --dockerized ../openlane/pe/config.json

asic_nano:
	@echo "=== Running OpenLane 2 ASIC Tapeout Flow for NeuroFlow Nano (Sky130) ==="
	$(OPENLANE) --dockerized ../openlane/nano/config.json

clean:
	rm -rf obj_dir *.vcd

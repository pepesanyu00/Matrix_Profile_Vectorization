# Scalable Vectorization of Time Series Similarity: A Matrix Profile. Implementation on RISC-V and ARM

This repository implements and benchmarks vectorized versions of Matrix Profile algorithms on vector length agnostic architectures, specifically RISC-V and ARM.

---

## Repository Structure

### 📁 `Algorithms`

Contains the source code for both **SCAMP** and **SCRIMP** algorithms. Each has four versions:

- `scamp/scrimp.cpp`: Base version, no vectorization.
- `scamp/scrimp-v.cpp`: Vectorized version for RISC-V.
- `scamp/scrimp-v_no_horizontal.cpp`: Vectorized version without horizontal max.
- `scamp/scrimp_initial`: Initial version (see Section 4.2 in the paper).

Includes a `Makefile` to build all versions, with additional options to compile SCAMP for execution under QEMU (including specific flags).

---

### 📁 `Timeseries`

Directory containing time series data used for benchmarking. All series and their required window sizes are defined in the `launch.sh` script.

---

### 📁 `Stats`

Execution statistics are saved in the `stats` directory, following this structure:

        stats-|
              |
              |
              ---- <arch>-<mode>-<timestamp>-|
                                             |
                                             |
                                             ------scamp-|
                                             |           |
                                             |           -audio-|
                                             |           |      |
                                             |           |      ---stats.txt
                                             |           |
                                             |           - human_activity-|
                                             |                            |
                                             |                            ----stats.txt
                                             |           ...
                                             |
                                             |----scrimp-|
                                                         ...

---

## Simulation Configuration

### ⚙️ `gem5_config_riscv.py`

Configuration script for **RISC-V** simulation using a `riscvO3` processor. Parameters:

- `Binary`: Path to the executable with its arguments  
- `-c`, `--cores`: Number of cores (default: 16)  
- `-v`, `--vlen`: Vector register length (default: 256)  
- `-e`, `--elen`: Element width (default: 64)  
- `--l1i_size`: L1 instruction cache size (default: 32KiB)  
- `--lid_size`: L1 data cache size (default: 64KiB)  
- `--l2_size`: L2 cache size (default: 256KiB)  
- `--l3_size`: L3 cache size (default: 16MiB)  

---

### ⚙️ `gem5_config_arm.py`

Same as the RISC-V configuration but for **ARM** architecture. Parameters:

- `Binary`: Path to the executable with its arguments  
- `-c`, `--cores`: Number of cores (default: 16)  
- `--l1i_size`: L1 instruction cache size (default: 32KiB)  
- `--lid_size`: L1 data cache size (default: 64KiB)  
- `--l2_size`: L2 cache size (default: 256KiB)  
- `--l3_size`: L3 cache size (default: 16MiB)  

---

## Execution & Visualization

### 🚀 `launch.sh`

Script to launch SCAMP simulations. Takes the following arguments:

- Architecture: `arm` or `riscv`  
- Mode: `sec` (sequential) or `vect` (vectorized)  
- Algorithm: `scamp` or `scrimp`  
- Threads: number of threads to use  

### 🚀 `run.sh`

Specifies and configure all the required arguments for launch.sh.

### 📊 `generate_graphics.py`

Script that generates plots from the statistics collected in the stats directory.

- `Vlen plots`: line plots showing speedup evolution across different vlen values. Each line corresponds to one architecture. You must specify the folder from which to read the statistics (inside the stats directory).
- `Threads plots`: Comparison plots in terms of threads for both configurations. It is configured to plot 2k vlen for both configurations and 16k vlen for RISC-V. It measures the speedup of two architectures and plots them as lines, one per each.
- `Algorithm option`: specifies which algorithm to execute.
- `--no_vect`: include no vectorized lines in the plot.

FLAGS = -march=rv64gcv -O2 -fopenmp -mabi=lp64d -static -Wall
M5FLAGS_RISCV = -I/home/jsanchez/gem5/include -L/home/jsanchez/gem5/util/m5/build/riscv/out -lm5
ROIFLAGS = -DENABLE_GEM5_ROI



all: scamp scamp-qemu scamp-v scamp-v-qemu scrimp scrimp-qemu scrimp-v scrimp-v-qemu prueba_ligera

launch: scamp scamp-v scrimp scrimp-v

qemu: scamp-qemu scamp-v-qemu scrimp-qemu scrimp-v-qemu


# ---------------------------------------------------------- SCAMP ---------------------------------------------------------#

# Risc-v secuential scamp
scamp: scamp.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) $(ROIFLAGS) -o scamp scamp.cpp $(M5FLAGS_RISCV)

# Risc-v sequential scamp with QEMU
scamp-qemu: scamp.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) -o scamp-qemu scamp.cpp

scamp-v_no_horizontal: scamp-v_no_horizontal.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) $(ROIFLAGS) -o scamp-v_no_horizontal scamp-v_no_horizontal.cpp $(M5FLAGS_RISCV)

# Risc-v vectorized scamp
scamp-v: scamp-v.cpp
	riscv64-unknown-linux-gnu-g++ $(ROIFLAGS) $(FLAGS) -o scamp-v scamp-v.cpp $(M5FLAGS_RISCV)
	#riscv64-unknown-linux-gnu-g++ $(FLAGS) -o scamp-v scamp-v.cpp
# Risc-v vectorized scamp with QEMU
scamp-v-qemu: scamp-v.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) -o scamp-v-qemu scamp-v.cpp



#-------------------------------------------------------- SCRIMP --------------------------------------------------------#

# Risc-v sequential scrimp
scrimp: scrimp.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) $(ROIFLAGS) -o scrimp scrimp.cpp $(M5FLAGS_RISCV)

# Risc-v sequential scrimp with QEMU
scrimp-qemu: scrimp.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) -o scrimp-qemu scrimp.cpp


# Risc-v vectorized scrimp
scrimp-v: scrimp-v.cpp
	riscv64-unknown-linux-gnu-g++ $(ROIFLAGS) $(FLAGS) -o scrimp-v scrimp-v.cpp $(M5FLAGS_RISCV)

# Risc-v vectorized scrimp with QEMU
scrimp-v-qemu: scrimp-v.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) -o scrimp-v-qemu scrimp-v.cpp
	 
scrimp-v_no_horizontal: scrimp-v_no_horizontal.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) -o scrimp-v_no_horizontal scrimp-v_no_horizontal.cpp $(ROIFLAGS) $(M5FLAGS_RISCV)



#-------------------------------------------------------- OMP --------------------------------------------------------#

# Scamp with OpenMP autovectorization
scamp-vomp: scamp-vomp.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) $(ROIFLAGS) -o scamp-vomp scamp-vomp.cpp $(M5FLAGS_RISCV)

# Scrimp with OpenMP autovectorization
scrimp-vomp: scrimp-vomp.cpp
	riscv64-unknown-linux-gnu-g++ $(FLAGS) $(ROIFLAGS) -o scrimp-vomp scrimp-vomp.cpp $(M5FLAGS_RISCV)


clean:
	rm -f scamp scamp-qemu scamp-v scamp-v-qemu scrimp scrimp-qemu scrimp-v scrimp-v-qemu scamp-vomp scrimp-vomp

FLAGS = -O2 -fopenmp -static -Wall -std=c++17 -march=armv8.2-a+sve
M5FLAGS_ARM = -I/home/jsanchez/gem5/include -L/home/jsanchez/gem5/util/m5/build/arm64/out -lm5
ROIFLAGS = -DENABLE_GEM5_ROI



all: scamp scamp-qemu scamp-v scamp-v-qemu scrimp scrimp-qemu scrimp-v scrimp-v-qemu

launch: scamp scamp-v scrimp scrimp-v

qemu: scamp-qemu scamp-v-qemu scrimp-qemu scrimp-v-qemu


# ---------------------------------------------------------- SCAMP ---------------------------------------------------------#

# ARM secuential scamp
scamp: scamp.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scamp scamp.cpp $(ROIFLAGS) $(M5FLAGS_ARM)

# ARM sequential scamp with QEMU
scamp-qemu: scamp.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scamp-qemu scamp.cpp

scamp-v_no_horizontal: scamp-v_no_horizontal.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scamp-v_no_horizontal scamp-v_no_horizontal.cpp $(ROIFLAGS) $(M5FLAGS_ARM)

# ARM vectorized scamp
scamp-v: scamp-v.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scamp-v scamp-v.cpp $(ROIFLAGS) $(M5FLAGS_ARM)

# ARM vectorized scamp with QEMU
scamp-v-qemu: scamp-v.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scamp-v-qemu scamp-v.cpp

scamp-v-initial: scamp-v_initial.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scamp-v_initial scamp-v_initial.cpp $(ROIFLAGS) $(M5FLAGS_ARM)

#-------------------------------------------------------- SCRIMP --------------------------------------------------------#

# ARM sequential scrimp
scrimp: scrimp.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scrimp scrimp.cpp $(ROIFLAGS) $(M5FLAGS_ARM)

# ARM sequential scrimp with QEMU
scrimp-qemu: scrimp.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scrimp-qemu scrimp.cpp


# ARM vectorized scrimp
scrimp-v: scrimp-v.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scrimp-v scrimp-v.cpp $(ROIFLAGS) $(M5FLAGS_ARM)

# ARM vectorized scrimp with QEMU
scrimp-v-qemu: scrimp-v.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scrimp-v-qemu scrimp-v.cpp

scrimp-v_no_horizontal: scrimp-v_no_horizontal.cpp
	aarch64-linux-gnu-g++ $(FLAGS) -o scrimp-v_no_horizontal scrimp-v_no_horizontal.cpp $(ROIFLAGS) $(M5FLAGS_ARM)



clean:
	rm -f scamp scamp-qemu scamp-v scamp-v-qemu scrimp scrimp-qemu scrimp-v scrimp-v-qemu

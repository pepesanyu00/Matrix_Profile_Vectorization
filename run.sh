# This file calls launch.sh file with diferent parameters to run the simulations

#threads=(1 2 4 8 16 32 64)
threads=(1)

arch="riscv"

mode="no_horizontal"

alg="scamp"

diags=10

for i in "${threads[@]}"; do
    ./launch.sh "$arch" "$mode" "$alg" "$i" "$diags"
done

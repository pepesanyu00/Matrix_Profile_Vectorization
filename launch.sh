#!/bin/bash


# Usage: ./launch.sh <arch> <mode>
#   <arch>: intel | riscv
#   <mode>: sec   | vect
#   <alg>:  scamp | scrimp
#   <threads>: number of threads to use
# This script launches different SCAMP executions depending on the architecture
# (intel or riscv) and whether it's sequential (sec) or vectorized (vect).
# It uses 'screen' to run multiple sessions in parallel.

arch="$1"    # First argument: architecture
mode="$2"    # Second argument: execution mode (sequential or vectorized)
alg="$3"     # Third argument: algorithm to run (scamp or scrimp)
threads="$4" # Fourth argument: number of threads to use
diags="$5"   # Fifth argument: diagonal percent to compute

# List of time series with their associated window size
 benchs=(#"timeseries/power-MPIII-SVF_n180000.txt 1325"
    #"timeseries/seismology-MPIII-SVE_n180000.txt 50"
    #"timeseries/e0103_n180000.txt 500"
    "timeseries/penguin_sample_TutorialMPweb.txt 50"
    #"timeseries/audio-MPIII-SVD.txt 200"
    #"timeseries/human_activity-MPIII-SVC.txt 120"
#   series XL
#   "timeseries/power-MPIII-SVF.txt 1325"
#   "timeseries/e0103.txt 500"
#   "timeseries/seismology-MPIII-SVE.txt 50"
)

# Possible vector length configurations
#vlen_values=(16384 8192 4096 2048 1024 512 256)
vlen_values=(2048 1024 512 256)
#vlen_values=(2048)
# Generate a timestamp for the output directories
TIMESTAMP=$(date +"%d-%m-%Y-%H-%M")

# Check if threads argument is provided
if [ -z "$threads" ]; then
    echo "Usage: ./launch.sh <arch> <mode> <alg> <threads> <diags>(arch: arm|riscv, mode: sec|vect, alg: scamp|scrimp, threads: number of threads, diags: percent of diagonals to compute)"
    exit 1
fi

# Check if the architecture argument is valid
if [ "$alg" != "scamp" ] && [ "$alg" != "scrimp" ]; then
    echo "Usage: ./launch.sh <arch> <mode> <alg> <threads> <diags> (arch: arm|riscv|arm, mode: sec|vect|no_horizontal, alg: scamp|scrimp, threads: number of threads, diags: percent of diagonals to compute)"
    exit 1
fi

# Determine which execution command to run based on arch-mode
case "$arch-$mode" in
    # ---------------------- ARM sequential execution -----------------------
    "arm-sec")

    # Execution directory
    exec_dir="stats/arm-sec"
    mkdir -p "$exec_dir"
    mkdir -p "$exec_dir/${alg}"

    # Loop through each time series in the benchs array
    for ts_file in "${benchs[@]}"
    do
        # Extract the base name of the time series file
        ts_basename=$(echo "$ts_file" | sed -E 's#.*/(.*)\.txt.*#\1#')

        # Stats directory
        SIM_DIR="$exec_dir/${alg}/${ts_basename}"
        mkdir -p "$SIM_DIR"

        THREAD_DIR="$SIM_DIR/${threads}_threads/"
        mkdir -p "$THREAD_DIR"

        # Simulation directory path and screen session name
        screen_name="exec${ts_basename}-${alg}-arm-sec-threads-${threads}"        
        # Launch the command in a detached screen session
        screen -dmS "$screen_name" bash -c "
            cd $(pwd);
            /home/jsanchez/gem5/build/ARM/gem5.opt --outdir \"$THREAD_DIR\" \
            gem5_config_arm.py -c $threads algorithms/${arch}/${alg} $ts_file $threads $diags $THREAD_DIR;"
    done
    ;;
    # ---------------------- ARM vectorized execution----------------------
    "arm-vect")

    # Execution directory
    exec_dir="stats_prueba/arm-vect"
    mkdir -p "$exec_dir"
    mkdir -p "$exec_dir/${alg}"

    # Loop through each time series in the benchs array
    for ts_file in "${benchs[@]}"
    do
        # Extract the base name of the time series file
        ts_basename=$(echo "$ts_file" | sed -E 's#.*/(.*)\.txt.*#\1#')
        TS_DIR="$exec_dir/${alg}/${ts_basename}"
        mkdir -p "$TS_DIR"

        for vlen in "${vlen_values[@]}"
        do
            VLEN_DIR="$TS_DIR/vlen-$vlen"
            mkdir -p "$VLEN_DIR"

            THREAD_DIR="$VLEN_DIR/${threads}_threads/"
            mkdir -p "$THREAD_DIR"

            screen_name="exec${ts_basename}-${alg}-arm-vect-vlen-${vlen}-threads-${threads}"
            screen -dmS "$screen_name" bash -c "
                cd $(pwd);
                /home/jsanchez/gem5/build/ARM/gem5.opt --outdir \"$THREAD_DIR\" \
                gem5_config_arm.py -c $threads algorithms/${arch}/${alg}-v $ts_file $threads $diags $THREAD_DIR --vlen $vlen;"
        done
    done
    ;;
    # -------------------- RISC-V sequential execution -----------------------
    "riscv-sec")

    # Execution directory
    exec_dir="stats/riscv-sec"
    mkdir -p "$exec_dir"
    mkdir -p "$exec_dir/${alg}"
    
    # Loop through each time series in the benchs array
    for ts_file in "${benchs[@]}"
    do
        # Extract the base name of the time series file
        ts_basename=$(echo "$ts_file" | sed -E 's#.*/(.*)\.txt.*#\1#')

        # Stats directory
        SIM_DIR="$exec_dir/${alg}/${ts_basename}"
        mkdir -p "$SIM_DIR"

        THREAD_DIR="$SIM_DIR/${threads}_threads/"
        mkdir -p "$THREAD_DIR"

        screen_name="exec${ts_basename}-${alg}-riscv-sec-threads-${threads}"
        screen -dmS "$screen_name" bash -c "
            cd $(pwd);
            /home/jsanchez/gem5/build/RISCV/gem5.opt --outdir \"$THREAD_DIR\" \
            gem5_config_riscv.py -c $threads algorithms/${arch}/${alg} $ts_file $threads $diags $THREAD_DIR;"
    done
    ;;
    # -------------------- RISC-V vectorized execution -----------------------
    "riscv-vect")

    # Execution directory
    exec_dir="stats/riscv-vect"
    mkdir -p "$exec_dir"
    mkdir -p "$exec_dir/${alg}"

    # Loop through each time series in the benchs array
    for ts_file in "${benchs[@]}"
    do
        # Extract the base name of the time series file
        ts_basename=$(echo "$ts_file" | sed -E 's#.*/(.*)\.txt.*#\1#')
        TS_DIR="$exec_dir/${alg}/${ts_basename}"
        mkdir -p "$TS_DIR"

        for vlen in "${vlen_values[@]}"
        do
            VLEN_DIR="$TS_DIR/vlen-$vlen"
            mkdir -p "$VLEN_DIR"

            THREAD_DIR="$VLEN_DIR/${threads}_threads/"
            mkdir -p "$THREAD_DIR"

            screen_name="exec${ts_basename}-${alg}-riscv-vect-vlen-${vlen}-threads-${threads}"
            screen -dmS "$screen_name" bash -c "
                cd $(pwd);
                /home/jsanchez/gem5/build/RISCV/gem5.opt --outdir \"$THREAD_DIR\" \
                gem5_config_riscv.py -c $threads algorithms/${arch}/${alg}-v $ts_file $threads $diags $THREAD_DIR --vlen $vlen;"
        done
    done
    ;;
    # -------------------- RISC-V vectorized no horizontal execution -----------------------
    "riscv-no_horizontal")

    # Execution directory
    exec_dir="stats_no_horizontal/riscv-vect"
    mkdir -p "$exec_dir"
    mkdir -p "$exec_dir/${alg}"

    # Loop through each time series in the benchs array
    for ts_file in "${benchs[@]}"
    do
        # Extract the base name of the time series file
        ts_basename=$(echo "$ts_file" | sed -E 's#.*/(.*)\.txt.*#\1#')
        TS_DIR="$exec_dir/${alg}/${ts_basename}"
        mkdir -p "$TS_DIR"

        for vlen in "${vlen_values[@]}"
        do
            VLEN_DIR="$TS_DIR/vlen-$vlen"
            mkdir -p "$VLEN_DIR"

            THREAD_DIR="$VLEN_DIR/${threads}_threads/"
            mkdir -p "$THREAD_DIR"

            screen_name="exec${ts_basename}-${alg}-riscv-no_hor-vlen-${vlen}-threads-${threads}"
            screen -dmS "$screen_name" bash -c "
                cd $(pwd);
                /home/jsanchez/gem5/build/RISCV/gem5.opt --outdir \"$THREAD_DIR\" \
                gem5_config_riscv.py -c $threads algorithms/${arch}/${alg}-v_no_horizontal $ts_file $threads $diags $THREAD_DIR --vlen $vlen;"
        done
    done
    ;;
    # -------------------- ARM vectorized no horizontal execution -----------------------
    "arm-no_horizontal")

    # Execution directory
    exec_dir="stats_no_horizontal/arm-vect"
    mkdir -p "$exec_dir"
    mkdir -p "$exec_dir/${alg}"

    # Loop through each time series in the benchs array
    for ts_file in "${benchs[@]}"
    do
        # Extract the base name of the time series file
        ts_basename=$(echo "$ts_file" | sed -E 's#.*/(.*)\.txt.*#\1#')
        TS_DIR="$exec_dir/${alg}/${ts_basename}"
        mkdir -p "$TS_DIR"

        for vlen in "${vlen_values[@]}"
        do
            VLEN_DIR="$TS_DIR/vlen-$vlen"
            mkdir -p "$VLEN_DIR"

            THREAD_DIR="$VLEN_DIR/${threads}_threads/"
            mkdir -p "$THREAD_DIR"

            screen_name="exec${ts_basename}-${alg}-arm-no_hor-vlen-${vlen}-threads-${threads}"
            screen -dmS "$screen_name" bash -c "
                cd $(pwd);
                /home/jsanchez/gem5/build/ARM/gem5.opt --outdir \"$THREAD_DIR\" \
                gem5_config_arm.py -c $threads algorithms/${arch}/${alg}-v_no_horizontal $ts_file $threads $diags $THREAD_DIR --vlen $vlen;"
        done
    done
    ;;
    *)
    echo "Usage: ./launch.sh <arch> <mode> <alg> <threads> (arch: intel|riscv, mode: sec|vect, alg: scamp|scrimp, threads: number of threads)"
    exit 1
    ;;
esac

echo "All simulation screens launched. Use 'screen -ls' to see running screens."
echo "Use 'screen -r <screen_name>' to attach."

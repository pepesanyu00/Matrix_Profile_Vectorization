import os
import sys
import argparse
import matplotlib.pyplot as plt


# Take the arguments
parser = argparse.ArgumentParser(description="Arguments for graph generation")
parser.add_argument('type', type=str, choices=['vlen','threads','no_horizontal'], help='Type of graph: vlen="vector length on X-axis", threads:"number of threads on X-axis", no_horizontal:"comparison of the version with vectorized horizontal maximum with the one without it."')
parser.add_argument('--stats_dir', required=False, type=str, default='stats', help="directory with statistics. Inner structure: stats1,stats2,stats3...")
parser.add_argument('algorithm', type=str, choices=['scamp','scrimp'], help='algorithm to generate graphs for.')
parser.add_argument('--no_vect', type=bool, help='whether to show lines for non-vectorized versions in threads or not.' )
args = parser.parse_args()


# Structure of the statistics folders:
# Sequential:
# arq-sec |
#         |-- scamp |
#         .         |-- human_act |
#         .         .             |-- 1th
#         .         .             .
# Vectorized:
# arq-vect |
#          |-- scamp |
#          .         |-- human_act |
#          .         .             |-- vlen-256 |
#          .         .             .            |-- 1threads
#          .         .             .            .

arqs = ['riscv', 'vlen']
riscv_vlen = [256,512,1024,2048,4096,8192,16384]
arm_vlen = [256,512,1024,2048]
riscv_vlen_no_hor = [256,512,1024,2048]
arm_vlen_no_hor = [256,512,1024,2048]
threads = [1,2,4,8,16,32,64]
timeseries = ['power-MPIII-SVF_n180000',
    'seismology-MPIII-SVE_n180000',
    'e0103_n180000',
    'penguin_sample_TutorialMPweb',
    'audio-MPIII-SVD',
    'human_activity-MPIII-SVC'
]

arm_no_vect = {}
arm_vect = {}
riscv_no_vect = {}
riscv_vect = {}



# Function that prints a dictionary appropriately.
def print_dict_pretty(dictionary, indent=4, method='json'):
    """
    Print a dictionary in a readable format.
    
    Args:
        dictionary: The dictionary to print
        indent: Number of spaces for indentation (default: 4)
        method: Printing method ('json', 'pprint', or 'custom')
    """
    if not dictionary:
        print("{}")
        return
    
    if method == 'json':
        import json
        print(json.dumps(dictionary, indent=indent, sort_keys=True))
    
    elif method == 'pprint':
        import pprint
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(dictionary)
    
    elif method == 'custom':
        def _print_dict(d, level=0):
            if not isinstance(d, dict):
                print(d)
                return
                
            print("{")
            for i, (key, value) in enumerate(d.items()):
                end_char = "" if i == len(d) - 1 else ","
                
                if isinstance(value, dict):
                    print(f"{' ' * (level+indent)}{repr(key)}: ", end="")
                    _print_dict(value, level + indent)
                    print(f"{end_char}")
                else:
                    print(f"{' ' * (level+indent)}{repr(key)}: {repr(value)}{end_char}")
            print(f"{' ' * level}}}", end="")
        
        _print_dict(dictionary)
        print()
    
    else:
        print("Invalid method. Choose 'json', 'pprint', or 'custom'")




# Start searching for data in the folders
if not os.path.exists(args.stats_dir):
    print(f"The directory {args.stats_dir} does not exist, please enter a valid one.")
    sys.exit(1)


# Collect all data from the folders
for stat_dir in os.listdir(args.stats_dir):
    stats_dir = os.path.join(args.stats_dir,stat_dir)
    for arq_mode in os.listdir(stats_dir):
        if arq_mode != 'riscv-vect-no-hor' and arq_mode != 'arm-vect-no-hor':
            mode = arq_mode.split('-')[1]
            arq = arq_mode.split('-')[0]
            alg_dir = os.path.join(stats_dir,arq_mode,args.algorithm)
            # Directory with TimeSeries
            for ts in os.listdir(alg_dir):
                ts_dir = os.path.join(alg_dir,ts)
                if mode == 'sec':
                    # Directory with threads
                    for thread in os.listdir(ts_dir):
                        thread_number = thread.split('_')[0]
                        # Statistics file
                        stats_file = os.path.join(ts_dir,thread,'stats.txt')
                        with open(stats_file, 'r') as file:
                            for line in file:
                                if line.startswith('simSeconds'):
                                    sim_seconds = float(line.split()[1])
                                    if (arq == 'arm'):
                                        if ts not in arm_no_vect:
                                            arm_no_vect[ts] = {}
                                        if thread_number not in arm_no_vect[ts]:
                                            arm_no_vect[ts][thread_number] = []
                                        arm_no_vect[ts][thread_number].append(sim_seconds)
                                        break
                                    elif (arq == 'riscv'):
                                        if ts not in riscv_no_vect:
                                            riscv_no_vect[ts] = {}
                                        if thread_number not in riscv_no_vect[ts]:
                                            riscv_no_vect[ts][thread_number] = []
                                        riscv_no_vect[ts][thread_number].append(sim_seconds)
                                        break
                elif mode == 'vect':
                    for vlen in os.listdir(ts_dir):
                        vlen_number = vlen.split('-')[1]
                        vlen_dir = os.path.join(ts_dir, vlen)
                        for thread in os.listdir(vlen_dir):
                            thread_number = thread.split('_')[0]
                            stats_file = os.path.join(vlen_dir,thread,'stats.txt')
                            with open(stats_file, 'r') as file:
                                for line in file:
                                    if line.startswith('simSeconds'):
                                        sim_seconds = float(line.split()[1])
                                        if (arq == 'arm'):
                                            if ts not in arm_vect:
                                                arm_vect[ts] = {}
                                            if vlen_number not in arm_vect[ts]:
                                                arm_vect[ts][vlen_number] = {}
                                            if thread_number not in arm_vect[ts][vlen_number]:
                                                arm_vect[ts][vlen_number][thread_number] = []
                                            arm_vect[ts][vlen_number][thread_number].append(sim_seconds)
                                            break
                                        elif (arq == 'riscv'):
                                            if ts not in riscv_vect:
                                                riscv_vect[ts] = {}
                                            if vlen_number not in riscv_vect[ts]:
                                                riscv_vect[ts][vlen_number] = {}
                                            if thread_number not in riscv_vect[ts][vlen_number]:
                                                riscv_vect[ts][vlen_number][thread_number] = []
                                            riscv_vect[ts][vlen_number][thread_number].append(sim_seconds)
                                            break                                  


# Calculate the average of the times
# RISC-V No Vect
for ts_key in riscv_no_vect:
    for thread_key in riscv_no_vect[ts_key]:
        times_list = riscv_no_vect[ts_key][thread_key]
        if times_list and isinstance(times_list, list) and all(isinstance(t, (int, float)) for t in times_list):
            mean_time = sum(times_list) / len(times_list)
            riscv_no_vect[ts_key][thread_key] = [mean_time]
        else:
            riscv_no_vect[ts_key][thread_key] = [None] # Mark as invalid/missing

# ARM No Vect
for ts_key in arm_no_vect:
    for thread_key in arm_no_vect[ts_key]:
        times_list = arm_no_vect[ts_key][thread_key]
        if times_list and isinstance(times_list, list) and all(isinstance(t, (int, float)) for t in times_list):
            mean_time = sum(times_list) / len(times_list)
            arm_no_vect[ts_key][thread_key] = [mean_time]
        else:
            arm_no_vect[ts_key][thread_key] = [None]

# RISC-V Vect
for ts_key in riscv_vect:
    for vlen_key in riscv_vect[ts_key]:
        for thread_key in riscv_vect[ts_key][vlen_key]:
            times_list = riscv_vect[ts_key][vlen_key][thread_key]
            if times_list and isinstance(times_list, list) and all(isinstance(t, (int, float)) for t in times_list):
                mean_time = sum(times_list) / len(times_list)
                riscv_vect[ts_key][vlen_key][thread_key] = [mean_time]
            else:
                riscv_vect[ts_key][vlen_key][thread_key] = [None]

# ARM Vect
for ts_key in arm_vect:
    for vlen_key in arm_vect[ts_key]:
        for thread_key in arm_vect[ts_key][vlen_key]:
            times_list = arm_vect[ts_key][vlen_key][thread_key]
            if times_list and isinstance(times_list, list) and all(isinstance(t, (int, float)) for t in times_list):
                mean_time = sum(times_list) / len(times_list)
                arm_vect[ts_key][vlen_key][thread_key] = [mean_time]
            else:
                arm_vect[ts_key][vlen_key][thread_key] = [None]


#-------------------------- Generate graphs with VLEN on the X-axis ----------------------------------#

# Type vlen
if args.type == 'vlen':
   # Calculate speedups
    riscv_no_vect_time_map = {} # Renamed to avoid conflict if used elsewhere
    arm_no_vect_time_map = {}   # Renamed
    speedup_riscv = {}
    speedup_arm = {}

    for ts in timeseries:
        speedup_riscv[ts] = {} # Initialize for current timeseries
        speedup_arm[ts] = {}   # Initialize for current timeseries

        base_riscv_time = None
        if ts in riscv_no_vect and "1" in riscv_no_vect[ts] and \
           riscv_no_vect[ts]["1"] and riscv_no_vect[ts]["1"][0] is not None:
            base_riscv_time = riscv_no_vect[ts]["1"][0]
        riscv_no_vect_time_map[ts] = base_riscv_time

        base_arm_time = None
        if ts in arm_no_vect and "1" in arm_no_vect[ts] and \
           arm_no_vect[ts]["1"] and arm_no_vect[ts]["1"][0] is not None:
            base_arm_time = arm_no_vect[ts]["1"][0]
        arm_no_vect_time_map[ts] = base_arm_time

        if base_riscv_time is None and base_arm_time is None:
            # print(f"Warning (vlen): No base time for {ts}. Skipping speedup calculation.")
            continue

        for idx, vlen_val in enumerate(riscv_vlen):
            vlen_str = str(vlen_val)
            
            if base_riscv_time is not None:
                if ts in riscv_vect and vlen_str in riscv_vect[ts] and \
                   "1" in riscv_vect[ts][vlen_str] and \
                   riscv_vect[ts][vlen_str]["1"] and \
                   riscv_vect[ts][vlen_str]["1"][0] is not None:
                    vect_time = riscv_vect[ts][vlen_str]["1"][0]
                    if vect_time > 0:
                        speedup_riscv[ts][vlen_val] = base_riscv_time / vect_time

            if base_arm_time is not None and idx < len(arm_vlen) and vlen_val == arm_vlen[idx]:
                if ts in arm_vect and vlen_str in arm_vect[ts] and \
                   "1" in arm_vect[ts][vlen_str] and \
                   arm_vect[ts][vlen_str]["1"] and \
                   arm_vect[ts][vlen_str]["1"][0] is not None:
                    vect_time = arm_vect[ts][vlen_str]["1"][0]
                    if vect_time > 0:
                        speedup_arm[ts][vlen_val] = base_arm_time / vect_time
    
    # The print_dict_pretty line needs to handle potentially empty dicts
    if timeseries and timeseries[0] in speedup_riscv and speedup_riscv[timeseries[0]]:
        print_dict_pretty(len(speedup_riscv[timeseries[0]]))

    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Generate plots for each timeseries
    for ts in timeseries:
        # Use .get to safely access timeseries data, defaulting to an empty dict
        current_speedup_riscv = speedup_riscv.get(ts, {})
        current_speedup_arm = speedup_arm.get(ts, {})

        if not current_speedup_riscv and not current_speedup_arm:
            # print(f"Skipping plot for {ts} (vlen type): No speedup data available.")
            continue

        plt.figure(figsize=(10, 6)) # Or your common_figure_size
        
        riscv_x_keys_plot = [] # To store keys for x-ticks if RISC-V is plotted

        # Plot RISC-V data
        if current_speedup_riscv:
            riscv_x_keys_plot = sorted(list(current_speedup_riscv.keys()))
            if riscv_x_keys_plot: # Check if there are keys to plot
                riscv_y_values = [current_speedup_riscv[v] for v in riscv_x_keys_plot]
                plt.plot(range(len(riscv_x_keys_plot)), riscv_y_values, 'b-o', linewidth=2, label='RISC-V')
        
        # Plot ARM data
        if current_speedup_arm:
            arm_x_keys_plot = sorted(list(current_speedup_arm.keys()))
            if arm_x_keys_plot: # Check if there are keys to plot
                arm_y_values = [current_speedup_arm[v] for v in arm_x_keys_plot]
                
                if riscv_x_keys_plot: # If RISC-V data exists, map ARM to its x-axis
                    arm_positions = []
                    arm_plot_y = []
                    for v_arm in arm_x_keys_plot:
                        if v_arm in riscv_x_keys_plot: # Ensure vlen from ARM is in RISC-V's list for index()
                            try:
                                idx = riscv_x_keys_plot.index(v_arm)
                                arm_positions.append(idx)
                                arm_plot_y.append(current_speedup_arm[v_arm])
                            except ValueError: # Should not happen if v_arm in riscv_x_keys_plot
                                pass 
                        
                    if arm_positions:
                        plt.plot(arm_positions, arm_plot_y, 'r-s', linewidth=2, label='ARM')
                else: # No RISC-V data, plot ARM on its own categorical axis
                    plt.plot(range(len(arm_x_keys_plot)), arm_y_values, 'r-s', linewidth=2, label='ARM')
        
        # Set x-tick labels (use RISC-V if available, else ARM if available)
        final_x_keys_for_ticks = []
        if riscv_x_keys_plot:
            final_x_keys_for_ticks = riscv_x_keys_plot
        elif current_speedup_arm and sorted(list(current_speedup_arm.keys())): # Check if arm_x_keys_plot would be non-empty
            final_x_keys_for_ticks = sorted(list(current_speedup_arm.keys()))

        if final_x_keys_for_ticks:
            plt.xticks(range(len(final_x_keys_for_ticks)), [str(x) for x in final_x_keys_for_ticks])
        
        # Add labels, title, and legend
        plt.xlabel('Vector Length (VLEN)', fontsize=34)
        plt.ylabel('Speedup', fontsize=34)
        plt.title(f'{ts}', fontsize=36)
        plt.yscale('log',base=10)  # Keep log scale for y-axis
        plt.tick_params(axis='both', which='major', labelsize=26)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=26, frameon=False, labelspacing=0.2)
        
        # Improve appearance
        plt.tight_layout()
        
        # Save plot as PNG
        plt.savefig(f'plots/{ts}_{args.algorithm}_vlen.pdf', dpi=300)
        plt.close()  # Close the figure to free memory

    print(f"Plots saved in the 'plots' directory")

#-------------------------- Generate graphs with THREADS on the X-axis ----------------------------------#

elif args.type == 'threads':
    # Calculate speedups for vectorization with VLEN=2048
    # Compare against the non-vectorized 1-thread version
    speedup_riscv_threads = {}
    speedup_riscv_16k = {}
    speedup_arm_threads = {}
    speedup_riscv_no_vect = {}  # New line for RISC-V non-vectorized
    speedup_arm_no_vect = {}    # New line for ARM non-vectorized
    riscv_no_vect_time = {}
    arm_no_vect_time = {}
    
    # Fixed VLEN for comparisons
    vlen_fixed = "2048"
    vlen_16k = "16384"
    
    # Calculate speedups for each timeseries
    for ts in timeseries:
        # Reference values (sequential, 1 thread)
        if ts in riscv_no_vect and "1" in riscv_no_vect[ts]:
            riscv_no_vect_time[ts] = riscv_no_vect[ts]["1"][0]
        else:
            riscv_no_vect_time[ts] = None

        if ts in arm_no_vect and "1" in arm_no_vect[ts]:
            arm_no_vect_time[ts] = arm_no_vect[ts]["1"][0]
        else:
            arm_no_vect_time[ts] = None

        # Speedups for different numbers of threads
        if riscv_no_vect_time[ts] is not None:
            # Speedup RISC-V vectorized VLEN=2048
            if ts in riscv_vect and vlen_fixed in riscv_vect[ts]:
                speedup_riscv_threads[ts] = {}
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in riscv_vect[ts][vlen_fixed] and riscv_vect[ts][vlen_fixed][thread_str]:
                        speedup_riscv_threads[ts][thread] = riscv_no_vect_time[ts] / riscv_vect[ts][vlen_fixed][thread_str][0]
            
            # Speedup RISC-V vectorized VLEN=16384
            if ts in riscv_vect and vlen_16k in riscv_vect[ts]:
                speedup_riscv_16k[ts] = {}
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in riscv_vect[ts][vlen_16k] and riscv_vect[ts][vlen_16k][thread_str]:
                        speedup_riscv_16k[ts][thread] = riscv_no_vect_time[ts] / riscv_vect[ts][vlen_16k][thread_str][0]
            
            # Speedup RISC-V non-vectorized
            if ts in riscv_no_vect:
                speedup_riscv_no_vect[ts] = {}
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in riscv_no_vect[ts] and riscv_no_vect[ts][thread_str]:
                        speedup_riscv_no_vect[ts][thread] = riscv_no_vect_time[ts] / riscv_no_vect[ts][thread_str][0]

        if arm_no_vect_time[ts] is not None:
            # Speedup ARM vectorized VLEN=2048
            if ts in arm_vect and vlen_fixed in arm_vect[ts]:
                speedup_arm_threads[ts] = {}
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in arm_vect[ts][vlen_fixed] and arm_vect[ts][vlen_fixed][thread_str]:
                        speedup_arm_threads[ts][thread] = arm_no_vect_time[ts] / arm_vect[ts][vlen_fixed][thread_str][0]
            
            # Speedup ARM non-vectorized
            if ts in arm_no_vect:
                speedup_arm_no_vect[ts] = {}
                for thread in threads:
                    thread_str = str(thread)
                    if thread_str in arm_no_vect[ts] and arm_no_vect[ts][thread_str]:
                        speedup_arm_no_vect[ts][thread] = arm_no_vect_time[ts] / arm_no_vect[ts][thread_str][0]

    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Generate plots for each timeseries
    for ts in timeseries:
        # Check if we have at least some data for this timeseries
        has_data = False
        if ts in speedup_riscv_threads and speedup_riscv_threads[ts]:
            has_data = True
        if ts in speedup_riscv_16k and speedup_riscv_16k[ts]:
            has_data = True
        if ts in speedup_arm_threads and speedup_arm_threads[ts]:
            has_data = True
        if ts in speedup_riscv_no_vect and speedup_riscv_no_vect[ts]:
            has_data = True
        if ts in speedup_arm_no_vect and speedup_arm_no_vect[ts]:
            has_data = True
            
        if not has_data:
            print(f"Skipping plot for {ts}: No data available.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Plot RISC-V vectorized VLEN=2048 (blue)
        if ts in speedup_riscv_threads and speedup_riscv_threads[ts]:
            riscv_x_keys = sorted(list(speedup_riscv_threads[ts].keys()))
            riscv_y_values = [speedup_riscv_threads[ts][thread] for thread in riscv_x_keys]
            plt.plot(range(len(riscv_x_keys)), riscv_y_values, 'b-o', linewidth=2, label='RVV_2k')

        # Plot RISC-V vectorized VLEN=16384 (yellow)
        if ts in speedup_riscv_16k and speedup_riscv_16k[ts]:
            riscv_16k_x_keys = sorted(list(speedup_riscv_16k[ts].keys()))
            riscv_16k_y_values = [speedup_riscv_16k[ts][thread] for thread in riscv_16k_x_keys]
            plt.plot(range(len(riscv_16k_x_keys)), riscv_16k_y_values, 'y-o', linewidth=2, label='RVV_16k')
        
        # Plot ARM vectorized VLEN=2048 (red)
        if ts in speedup_arm_threads and speedup_arm_threads[ts]:
            arm_x_keys = sorted(list(speedup_arm_threads[ts].keys()))
            arm_y_values = [speedup_arm_threads[ts][thread] for thread in arm_x_keys]
            # Map ARM values to consistent x-axis
            arm_positions = []
            arm_plot_y_values = []
            for i, thread_val in enumerate(riscv_x_keys if 'riscv_x_keys' in locals() else arm_x_keys):
                if thread_val in speedup_arm_threads[ts]:
                    arm_positions.append(i)
                    arm_plot_y_values.append(speedup_arm_threads[ts][thread_val])
            if arm_positions:
                plt.plot(arm_positions, arm_plot_y_values, 'r-s', linewidth=2, label='SVE_2k')
        
        if args.no_vect:
            # Plot RISC-V non-vectorized (green)
            if ts in speedup_riscv_no_vect and speedup_riscv_no_vect[ts]:
                riscv_no_vect_x_keys = sorted(list(speedup_riscv_no_vect[ts].keys()))
                riscv_no_vect_y_values = [speedup_riscv_no_vect[ts][thread] for thread in riscv_no_vect_x_keys]
                plt.plot(range(len(riscv_no_vect_x_keys)), riscv_no_vect_y_values, 'g--^', linewidth=2, label='RISC-V_NV')
            
            # Plot ARM non-vectorized (orange)
            if ts in speedup_arm_no_vect and speedup_arm_no_vect[ts]:
                arm_no_vect_x_keys = sorted(list(speedup_arm_no_vect[ts].keys()))
                arm_no_vect_y_values = [speedup_arm_no_vect[ts][thread] for thread in arm_no_vect_x_keys]
                # Map ARM no vect values to consistent x-axis
                reference_keys = riscv_x_keys if 'riscv_x_keys' in locals() else arm_no_vect_x_keys
                arm_no_vect_positions = []
                arm_no_vect_plot_y_values = []
                for i, thread_val in enumerate(reference_keys):
                    if thread_val in speedup_arm_no_vect[ts]:
                        arm_no_vect_positions.append(i)
                        arm_no_vect_plot_y_values.append(speedup_arm_no_vect[ts][thread_val])
                if arm_no_vect_positions:
                    plt.plot(arm_no_vect_positions, arm_no_vect_plot_y_values, 'orange', linestyle='--', marker='v', linewidth=2, label='ARM_NV')


        # Set x-tick labels (use the most complete set of thread counts available)
        if 'riscv_x_keys' in locals():
            plt.xticks(range(len(riscv_x_keys)), [str(x) for x in riscv_x_keys])
        elif 'arm_x_keys' in locals():
            plt.xticks(range(len(arm_x_keys)), [str(x) for x in arm_x_keys])
        
        # Add labels, title, and legend
        plt.xlabel('Number of Threads', fontsize=34)
        plt.ylabel('Speedup', fontsize=34)
        plt.title(f'{ts}', fontsize=36)
        plt.tick_params(axis='both', which='major', labelsize=26)
        plt.yscale('log',base=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=20, frameon=False, labelspacing=0.1, loc='upper right',ncol=2, columnspacing=0.5, bbox_to_anchor=(1, 0.35))
        
        # Improve appearance
        plt.tight_layout()
        
        # Save plot as PDF
        plt.savefig(f'plots/{ts}_{args.algorithm}_threads.pdf', dpi=300)
        plt.close()

elif args.type == 'no_horizontal':
    # First, we need to load data from the no-hor versions
    arm_vect_no_hor = {}
    riscv_vect_no_hor = {}
    
    # Load data from versions without horizontal optimization
    for stat_dir in os.listdir(args.stats_dir):
        stats_dir_path = os.path.join(args.stats_dir, stat_dir)
        for arq_mode in os.listdir(stats_dir_path):
            if arq_mode.endswith('-vect-no-hor'):
                mode = 'vect-no-hor'
                arq = arq_mode.split('-')[0]
                alg_dir = os.path.join(stats_dir_path, arq_mode, args.algorithm)
                
                # Directory with TimeSeries
                for ts in os.listdir(alg_dir):
                    ts_dir = os.path.join(alg_dir, ts)
                    for vlen in os.listdir(ts_dir):
                        vlen_number = vlen.split('-')[1]
                        vlen_dir = os.path.join(ts_dir, vlen)
                        for thread in os.listdir(vlen_dir):
                            thread_number = thread.split('_')[0]
                            stats_file = os.path.join(vlen_dir, thread, 'stats.txt')
                            
                            try:
                                with open(stats_file, 'r') as file:
                                    for line in file:
                                        if line.startswith('simSeconds'):
                                            sim_seconds = float(line.split()[1])
                                            if arq == 'arm':
                                                if ts not in arm_vect_no_hor:
                                                    arm_vect_no_hor[ts] = {}
                                                if vlen_number not in arm_vect_no_hor[ts]:
                                                    arm_vect_no_hor[ts][vlen_number] = {}
                                                if thread_number not in arm_vect_no_hor[ts][vlen_number]:
                                                    arm_vect_no_hor[ts][vlen_number][thread_number] = []
                                                arm_vect_no_hor[ts][vlen_number][thread_number].append(sim_seconds)
                                                break
                                            elif arq == 'riscv':
                                                if ts not in riscv_vect_no_hor:
                                                    riscv_vect_no_hor[ts] = {}
                                                if vlen_number not in riscv_vect_no_hor[ts]:
                                                    riscv_vect_no_hor[ts][vlen_number] = {}
                                                if thread_number not in riscv_vect_no_hor[ts][vlen_number]:
                                                    riscv_vect_no_hor[ts][vlen_number][thread_number] = []
                                                riscv_vect_no_hor[ts][vlen_number][thread_number].append(sim_seconds)
                                                break
                            except FileNotFoundError:
                                print(f"File not found: {stats_file}")
                                continue
    
    # Calculate means for no-hor versions
    # RISC-V Vect No-Hor
    for ts_key in riscv_vect_no_hor:
        for vlen_key in riscv_vect_no_hor[ts_key]:
            for thread_key in riscv_vect_no_hor[ts_key][vlen_key]:
                times_list = riscv_vect_no_hor[ts_key][vlen_key][thread_key]
                if times_list and isinstance(times_list, list) and all(isinstance(t, (int, float)) for t in times_list):
                    mean_time = sum(times_list) / len(times_list)
                    riscv_vect_no_hor[ts_key][vlen_key][thread_key] = [mean_time]
                else:
                    riscv_vect_no_hor[ts_key][vlen_key][thread_key] = [None]

    # ARM Vect No-Hor
    for ts_key in arm_vect_no_hor:
        for vlen_key in arm_vect_no_hor[ts_key]:
            for thread_key in arm_vect_no_hor[ts_key][vlen_key]:
                times_list = arm_vect_no_hor[ts_key][vlen_key][thread_key]
                if times_list and isinstance(times_list, list) and all(isinstance(t, (int, float)) for t in times_list):
                    mean_time = sum(times_list) / len(times_list)
                    arm_vect_no_hor[ts_key][vlen_key][thread_key] = [mean_time]
                else:
                    arm_vect_no_hor[ts_key][vlen_key][thread_key] = [None]

    # Calculate speedups
    riscv_no_vect_time_map = {}
    arm_no_vect_time_map = {}
    speedup_riscv = {}
    speedup_arm = {}
    speedup_riscv_no_hor = {}
    speedup_arm_no_hor = {}

    for ts in timeseries:
        speedup_riscv[ts] = {}
        speedup_arm[ts] = {}
        speedup_riscv_no_hor[ts] = {}
        speedup_arm_no_hor[ts] = {}

        base_riscv_time = None
        if ts in riscv_no_vect and "1" in riscv_no_vect[ts] and \
           riscv_no_vect[ts]["1"] and riscv_no_vect[ts]["1"][0] is not None:
            base_riscv_time = riscv_no_vect[ts]["1"][0]
        riscv_no_vect_time_map[ts] = base_riscv_time

        base_arm_time = None
        if ts in arm_no_vect and "1" in arm_no_vect[ts] and \
           arm_no_vect[ts]["1"] and arm_no_vect[ts]["1"][0] is not None:
            base_arm_time = arm_no_vect[ts]["1"][0]
        arm_no_vect_time_map[ts] = base_arm_time

        if base_riscv_time is None and base_arm_time is None:
            continue

        for idx, vlen_val in enumerate(riscv_vlen_no_hor):
            vlen_str = str(vlen_val)
            
            # RISC-V optimized (solid line)
            if base_riscv_time is not None:
                if ts in riscv_vect and vlen_str in riscv_vect[ts] and \
                   "1" in riscv_vect[ts][vlen_str] and \
                   riscv_vect[ts][vlen_str]["1"] and \
                   riscv_vect[ts][vlen_str]["1"][0] is not None:
                    vect_time = riscv_vect[ts][vlen_str]["1"][0]
                    if vect_time > 0:
                        speedup_riscv[ts][vlen_val] = base_riscv_time / vect_time
                
                # RISC-V no-hor (dashed line)
                if ts in riscv_vect_no_hor and vlen_str in riscv_vect_no_hor[ts] and \
                   "1" in riscv_vect_no_hor[ts][vlen_str] and \
                   riscv_vect_no_hor[ts][vlen_str]["1"] and \
                   riscv_vect_no_hor[ts][vlen_str]["1"][0] is not None:
                    vect_time_no_hor = riscv_vect_no_hor[ts][vlen_str]["1"][0]
                    if vect_time_no_hor > 0:
                        speedup_riscv_no_hor[ts][vlen_val] = base_riscv_time / vect_time_no_hor

            # ARM optimized and no-hor
            if base_arm_time is not None and idx < len(arm_vlen) and vlen_val == arm_vlen[idx]:
                # ARM optimized (solid line)
                if ts in arm_vect and vlen_str in arm_vect[ts] and \
                   "1" in arm_vect[ts][vlen_str] and \
                   arm_vect[ts][vlen_str]["1"] and \
                   arm_vect[ts][vlen_str]["1"][0] is not None:
                    vect_time = arm_vect[ts][vlen_str]["1"][0]
                    if vect_time > 0:
                        speedup_arm[ts][vlen_val] = base_arm_time / vect_time
                
                # ARM no-hor (dashed line)
                if ts in arm_vect_no_hor and vlen_str in arm_vect_no_hor[ts] and \
                   "1" in arm_vect_no_hor[ts][vlen_str] and \
                   arm_vect_no_hor[ts][vlen_str]["1"] and \
                   arm_vect_no_hor[ts][vlen_str]["1"][0] is not None:
                    vect_time_no_hor = arm_vect_no_hor[ts][vlen_str]["1"][0]
                    if vect_time_no_hor > 0:
                        speedup_arm_no_hor[ts][vlen_val] = base_arm_time / vect_time_no_hor

    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Generate plots for each timeseries
    for ts in timeseries:
        current_speedup_riscv = speedup_riscv.get(ts, {})
        current_speedup_arm = speedup_arm.get(ts, {})
        current_speedup_riscv_no_hor = speedup_riscv_no_hor.get(ts, {})
        current_speedup_arm_no_hor = speedup_arm_no_hor.get(ts, {})

        # Check if we have any data
        if (not current_speedup_riscv and not current_speedup_arm and 
            not current_speedup_riscv_no_hor and not current_speedup_arm_no_hor):
            continue

        plt.figure(figsize=(10, 6))
        
        riscv_x_keys_plot = []

        # Plot RISC-V optimized (solid blue)
        if current_speedup_riscv:
            riscv_x_keys_plot = sorted(list(current_speedup_riscv.keys()))
            if riscv_x_keys_plot:
                riscv_y_values = [current_speedup_riscv[v] for v in riscv_x_keys_plot]
                plt.plot(range(len(riscv_x_keys_plot)), riscv_y_values, 'b-o', linewidth=2, label='RVV')

        # Plot RISC-V no-hor (dashed blue)
        if current_speedup_riscv_no_hor:
            riscv_no_hor_x_keys = sorted(list(current_speedup_riscv_no_hor.keys()))
            if riscv_no_hor_x_keys:
                riscv_no_hor_y_values = [current_speedup_riscv_no_hor[v] for v in riscv_no_hor_x_keys]
                # Si no tenemos datos de RISC-V optimizado, usar estas claves para x-ticks
                if not riscv_x_keys_plot:
                    riscv_x_keys_plot = riscv_no_hor_x_keys
                plt.plot(range(len(riscv_no_hor_x_keys)), riscv_no_hor_y_values, 'b--^', linewidth=2, label='RVV No-Hor')
        
        # Plot ARM optimizado (rojo sólido)
        if current_speedup_arm:
            arm_x_keys_plot = sorted(list(current_speedup_arm.keys()))
            if arm_x_keys_plot:
                arm_y_values = [current_speedup_arm[v] for v in arm_x_keys_plot]
                
                if riscv_x_keys_plot:  # Map to RISC-V x-axis
                    arm_positions = []
                    arm_plot_y = []
                    for v_arm in arm_x_keys_plot:
                        if v_arm in riscv_x_keys_plot:
                            try:
                                idx = riscv_x_keys_plot.index(v_arm)
                                arm_positions.append(idx)
                                arm_plot_y.append(current_speedup_arm[v_arm])
                            except ValueError:
                                pass
                    if arm_positions:
                        plt.plot(arm_positions, arm_plot_y, 'r-s', linewidth=2, label='SVE')
                else:  # No RISC-V data, plot ARM on its own axis
                    plt.plot(range(len(arm_x_keys_plot)), arm_y_values, 'r-s', linewidth=2, label='SVE')

        # Plot ARM no-hor (rojo discontinuo)
        if current_speedup_arm_no_hor:
            arm_no_hor_x_keys = sorted(list(current_speedup_arm_no_hor.keys()))
            if arm_no_hor_x_keys:
                arm_no_hor_y_values = [current_speedup_arm_no_hor[v] for v in arm_no_hor_x_keys]
                
                if riscv_x_keys_plot:  # Map to RISC-V x-axis
                    arm_no_hor_positions = []
                    arm_no_hor_plot_y = []
                    for v_arm in arm_no_hor_x_keys:
                        if v_arm in riscv_x_keys_plot:
                            try:
                                idx = riscv_x_keys_plot.index(v_arm)
                                arm_no_hor_positions.append(idx)
                                arm_no_hor_plot_y.append(current_speedup_arm_no_hor[v_arm])
                            except ValueError:
                                pass
                    if arm_no_hor_positions:
                        plt.plot(arm_no_hor_positions, arm_no_hor_plot_y, 'r--v', linewidth=2, label='SVE No-Hor')
                else:  # No RISC-V data, plot ARM on its own axis
                    plt.plot(range(len(arm_no_hor_x_keys)), arm_no_hor_y_values, 'r--v', linewidth=2, label='SVE No-Hor')

        # Set x-tick labels
        final_x_keys_for_ticks = []
        if riscv_x_keys_plot:
            final_x_keys_for_ticks = riscv_x_keys_plot
        elif current_speedup_arm and sorted(list(current_speedup_arm.keys())):
            final_x_keys_for_ticks = sorted(list(current_speedup_arm.keys()))
        elif current_speedup_arm_no_hor and sorted(list(current_speedup_arm_no_hor.keys())):
            final_x_keys_for_ticks = sorted(list(current_speedup_arm_no_hor.keys()))

        if final_x_keys_for_ticks:
            plt.xticks(range(len(final_x_keys_for_ticks)), [str(x) for x in final_x_keys_for_ticks])
        
        # Add labels, title, and legend
        plt.xlabel('Vector Length (VLEN)', fontsize=34)
        plt.ylabel('Speedup', fontsize=34)
        plt.title(f'{ts}', fontsize=36)
        #plt.yscale('log', base=10)
        plt.tick_params(axis='both', which='major', labelsize=26)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=24, frameon=False, labelspacing=0.2, loc='upper right', bbox_to_anchor=(0.45, 1))
        
        # Improve appearance
        plt.tight_layout(pad=3)
        
        # Save plot as PDF
        plt.savefig(f'plots/{ts}_{args.algorithm}_no_horizontal.pdf', dpi=300)
        plt.close()

    print(f"No-horizontal comparison plots saved in the 'plots' directory")

        

if args.type == 'threads':

    # Calculate thread-by-thread improvements and overall averages
    print("\n=== SPEEDUP ANALYSIS ===")

    # RISC-V: Compare 2k vectorized vs non-vectorized
    riscv_2k_improvements = []
    if speedup_riscv_threads and speedup_riscv_no_vect:
        print("\nRISC-V (VLEN=2048 vs Non-vectorized):")
        for ts in timeseries:
            if ts in speedup_riscv_threads and ts in speedup_riscv_no_vect:
                print(f"  {ts}:")
                for thread in threads:
                    if (thread in speedup_riscv_threads[ts] and 
                        thread in speedup_riscv_no_vect[ts]):
                        
                        vect_speedup = speedup_riscv_threads[ts][thread]
                        no_vect_speedup = speedup_riscv_no_vect[ts][thread]
                        improvement = vect_speedup / no_vect_speedup
                        riscv_2k_improvements.append(improvement)
                        print(f"    {thread} threads: {improvement:.2f}x improvement")

    # RISC-V: Compare 16k vectorized vs non-vectorized
    riscv_16k_improvements = []
    if speedup_riscv_16k and speedup_riscv_no_vect:
        print("\nRISC-V (VLEN=16384 vs Non-vectorized):")
        for ts in timeseries:
            if ts in speedup_riscv_16k and ts in speedup_riscv_no_vect:
                print(f"  {ts}:")
                for thread in threads:
                    if (thread in speedup_riscv_16k[ts] and 
                        thread in speedup_riscv_no_vect[ts]):
                        
                        vect_speedup = speedup_riscv_16k[ts][thread]
                        no_vect_speedup = speedup_riscv_no_vect[ts][thread]
                        improvement = vect_speedup / no_vect_speedup
                        riscv_16k_improvements.append(improvement)
                        print(f"    {thread} threads: {improvement:.2f}x improvement")

    # ARM: Compare 2k vectorized vs non-vectorized  
    arm_improvements = []
    if speedup_arm_threads and speedup_arm_no_vect:
        print("\nARM (VLEN=2048 vs Non-vectorized):")
        for ts in timeseries:
            if ts in speedup_arm_threads and ts in speedup_arm_no_vect:
                print(f"  {ts}:")
                for thread in threads:
                    if (thread in speedup_arm_threads[ts] and 
                        thread in speedup_arm_no_vect[ts]):
                        
                        vect_speedup = speedup_arm_threads[ts][thread]
                        no_vect_speedup = speedup_arm_no_vect[ts][thread]
                        improvement = vect_speedup / no_vect_speedup
                        arm_improvements.append(improvement)
                        print(f"    {thread} threads: {improvement:.2f}x improvement")

    # Calculate and print overall averages
    print("\n=== OVERALL AVERAGES ===")
    if riscv_2k_improvements:
        riscv_2k_mean = sum(riscv_2k_improvements) / len(riscv_2k_improvements)
        print(f"RISC-V average improvement (2k vs non-vect): {riscv_2k_mean:.2f}x")
        print(f"RISC-V 2k measurements count: {len(riscv_2k_improvements)}")
    else:
        print("No RISC-V 2k improvement data available")

    if riscv_16k_improvements:
        riscv_16k_mean = sum(riscv_16k_improvements) / len(riscv_16k_improvements)
        print(f"RISC-V average improvement (16k vs non-vect): {riscv_16k_mean:.2f}x")
        print(f"RISC-V 16k measurements count: {len(riscv_16k_improvements)}")
    else:
        print("No RISC-V 16k improvement data available")

    if arm_improvements:
        arm_mean = sum(arm_improvements) / len(arm_improvements)
        print(f"ARM average improvement (2k vs non-vect): {arm_mean:.2f}x")
        print(f"ARM measurements count: {len(arm_improvements)}")
    else:
        print("No ARM improvement data available")

    print("=" * 40)

    print(f"Thread speedup plots saved in the 'plots' directory")
# Script que genera gráfica de tiempo de ejecución de las simulaciones
# Requisitos: los resultados de estadísticas deben encontrarse en la carpeta stats, y dentro de esta 
# deben existir carpetas con el formato 'scamp-v-vlen-<valor>', donde <valor> es el tamaño de vlen.
# De momento solo ejecuta una sola serie temporal, pero como mejora futura se añadirán más conforme 
# evolucione el proyecto.

import os
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Generar gráficas de tiempos de ejecución.')
parser.add_argument('type', type=str, choices=['vlen', 'comp'], help='Tipo de gráfica a generar.')
parser.add_argument('scamp_scrimp', type=str, choices=['scamp', 'scrimp'], help='Carpeta scamp o scrimp.')
parser.add_argument('--C1', required=False, type=str, help='Nombre de la carpeta dentro de stats.')
parser.add_argument('--vect', required=False, type=str, help='Carpeta de la versión vectorial (obligatorio si type=comp).')
parser.add_argument('--sec', required=False, type=str, help='Carpeta de la versión secuencial (obligatorio si type=comp).')
args = parser.parse_args()

if args.type == 'vlen' and not args.C1:
    print('El argumento C1 es obligatorio cuando type=vlen.')
    exit()

if args.type == 'comp' and (not args.vect or not args.sec):
    print('Los argumentos --vect y --sec son obligatorios cuando type=comp.')
    exit()

if args.type == 'vlen':
    execution_times = {}
    C1_dir = os.path.join(args.C1, args.scamp_scrimp)
    series_names = set()
    vlen_values = [256, 512, 1024, 2048, 4096, 8192, 16384]

    for series_folder in os.listdir(C1_dir):
        series_path = os.path.join(C1_dir, series_folder)
        if os.path.isdir(series_path):
            series_names.add(series_folder)
            for folder_name in os.listdir(series_path):
                if folder_name.startswith('scamp-v-vlen-'):
                    vlen = int(folder_name.split('-')[3])
                    stats_file_path = os.path.join(series_path, folder_name, 'stats.txt')
                    if os.path.isfile(stats_file_path):
                        with open(stats_file_path, 'r') as file:
                            for line in file:
                                if line.startswith('simSeconds'):
                                    sim_seconds = float(line.split()[1])
                                    if series_folder not in execution_times:
                                        execution_times[series_folder] = {}
                                    if vlen not in execution_times[series_folder]:
                                        execution_times[series_folder][vlen] = []
                                    execution_times[series_folder][vlen].append(sim_seconds)
                                    break

    if not execution_times:
        print('No se encontraron resultados de simulación en el directorio base.')
        exit()

    sorted_vlen = sorted({vlen for series in execution_times.values() for vlen in series.keys()})
    plt.figure(figsize=(10, 6))
    
    for series_name in series_names:
        sorted_times = [execution_times[series_name].get(vlen, None) for vlen in vlen_values]
        plt.plot(vlen_values, sorted_times, marker='o', label=f'Serie Temporal {series_name}')

    plt.xlabel('Vlen')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Comparación de tiempos de ejecución según Vlen')
    plt.yscale('log')  # Usar escala logarítmica en el eje Y
    plt.xticks(vlen_values)
    plt.legend()
    plt.tight_layout()


elif args.type == 'comp':
    vect_dir = os.path.join( args.vect, args.scamp_scrimp)
    sec_dir = os.path.join( args.sec, args.scamp_scrimp)

    execution_times_vect = {}
    execution_times_sec = {}

    for series_folder in os.listdir(vect_dir):
        series_path = os.path.join(vect_dir, series_folder)
        if os.path.isdir(series_path):
            stats_file_path = os.path.join(series_path, 'stats.txt')
            if os.path.isfile(stats_file_path):
                with open(stats_file_path, 'r') as file:
                    for line in file:
                        if line.startswith('simSeconds'):
                            sim_seconds = float(line.split()[1])
                            execution_times_vect[series_folder] = sim_seconds
                            break

    for series_folder in os.listdir(sec_dir):
        series_path = os.path.join(sec_dir, series_folder)
        if os.path.isdir(series_path):
            stats_file_path = os.path.join(series_path, 'stats.txt')
            if os.path.isfile(stats_file_path):
                with open(stats_file_path, 'r') as file:
                    for line in file:
                        if line.startswith('simSeconds'):
                            sim_seconds = float(line.split()[1])
                            execution_times_sec[series_folder] = sim_seconds
                            break

    if not execution_times_vect or not execution_times_sec:
        print('No se encontraron resultados de simulación en los directorios especificados.')
        exit()

    sorted_series = sorted(execution_times_vect.keys())
    vect_times = [execution_times_vect[series] for series in sorted_series]
    sec_times = [execution_times_sec[series] for series in sorted_series]

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(sorted_series))

    plt.bar(index, sec_times, bar_width, label='Secuencial')
    plt.bar([i + bar_width for i in index], vect_times, bar_width, label='Vectorial')

    plt.xlabel('Series Temporales')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Comparación de tiempos de ejecución: Secuencial vs Vectorial')
    plt.xticks([i + bar_width / 2 for i in index], sorted_series)
    plt.legend()
    plt.tight_layout()
else:
    print('El tipo de gráfica especificado no es válido.')
    exit()

# Crear directorio con timestamp
timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
new_dir_name = f"stats/graphics/graphic-{args.type}-{timestamp}"
os.makedirs(new_dir_name, exist_ok=True)

# Guardar gráfica en el nuevo directorio
plt.savefig(os.path.join(new_dir_name, 'graphic.png'), dpi=300, bbox_inches='tight')
plt.close()
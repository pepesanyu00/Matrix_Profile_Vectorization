#!/usr/bin/env python3
"""
Ejemplo de configuración de un sistema ARM con extensión SVE usando
las librerías m5 de gem5. En este script se crea una jerarquía de
cachés de tres niveles y se configura el tamaño del vector (vl) a través
del parámetro 'sve_vl_se' del objeto ArmISA.
"""

import m5
from m5.objects import Root, System, SystemXBar, AddrRange,ArmO3CPU, ArmISA, Cache, SnoopFilter, L2XBar

from custom_cache.caches.l1dcache import L1DCache
from custom_cache.caches.l1icache import L1ICache
from custom_cache.caches.l2cache import L2Cache
from custom_cache.caches.l3cache import L3Cache

from m5.objects import MemCtrl, DDR4_2400_16x4  # Update imports
from m5.objects import SEWorkload
from m5.objects import RubySystem
# Intentar importar los procesos específicos de ARM
try:
    # Nueva ruta en versiones recientes
    from m5.objects.arm.ArmProcess import ArmProcess, ArmSEProcess
except ImportError:
    try:
        # Ruta alternativa
        from m5.objects.Process import Process, SEProcess
        # Usar proceso genérico si no está disponible el específico de ARM
        ArmProcess = SEProcess
    except ImportError:
        # Usar proceso base como último recurso
        from m5.objects import Process
        ArmProcess = Process
import argparse
import sys



# Análisis de argumentos
parser = argparse.ArgumentParser(description="Simulación ARM con SVE usando m5")
parser.add_argument("binary", nargs="+", type=str, help="Ruta al binario a simular y sus argumentos")
parser.add_argument("-c", "--cores", type=int, default=64, help="Número de cores (default: 64)")
parser.add_argument("-v", "--vlen", type=int, default=512, help="Tamaño del vector (VL) en unidades (default: 4)")
parser.add_argument("-e", "--elen", type=int, default=64, help="Tamaño del elemento (ELEN) (default: 64)")
parser.add_argument("--l1i_size", default="32KiB", help="Tamaño de la caché L1 de instrucciones (default: 32KiB)")
parser.add_argument("--l1d_size", default="64KiB", help="Tamaño de la caché L1 de datos (default: 64KiB)")
parser.add_argument("--l2_size", default="256KiB", help="Tamaño de la caché L2 (default: 256KiB)")
parser.add_argument("--l3_size", default="16MiB", help="Tamaño de la caché L3 (default: 16MiB)")
args = parser.parse_args()

# Crear el sistema
system = System()
system.clk_domain = m5.objects.SrcClockDomain(clock="2GHz", voltage_domain=m5.objects.VoltageDomain()) 
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange("32GiB")]

# Crear los CPUs con extensión SVE
# Reemplazar la asignación directa con Vector
system.cpu = [ArmO3CPU(cpu_id=i) for i in range(args.cores)]
# Crear la conexión principal (bus de memoria)
system.membus = SystemXBar()

# Configurar cada CPU
for i in range(args.cores):
    # Configuración de registros vectoriales
    system.cpu[i].numPhysVecRegs = 256
    system.cpu[i].numPhysVecPredRegs = 64
    system.cpu[i].createInterruptController()
    # Configurar el ISA con el parámetro de SVE: 'sve_vl_se' (vl expresado en palabras de 128 bits)
    system.cpu[i].isa = ArmISA(sve_vl_se=args.vlen/128)

    #system.cpu[i].icache_port = system.membus.cpu_side_ports
    #system.cpu[i].dcache_port = system.membus.cpu_side_ports


# --- Configuración de la jerarquía de cachés ---

system.cache_line_size = 64 


# Se crea un bus L3 compartido y una caché L3
system.l3bus = L2XBar()
system.l3cache = L3Cache(size = args.l3_size)


system.l3cache.cpu_side = system.l3bus.mem_side_ports

# L1 y L2 privadas
for i in range(args.cores):
    cpu = system.cpu[i]
    # Configurar L1 caches
    cpu.icache = L1ICache(size = args.l1i_size)
    cpu.dcache = L1DCache(size = args.l1d_size)
    
    # Conectar L1 caches a CPU
    cpu.icache.cpu_side = cpu.icache_port
    cpu.dcache.cpu_side = cpu.dcache_port

    cpu.l2bus = L2XBar(snoop_response_latency=3)

    # Conectar L1 caches al bus privado
    cpu.icache.mem_side = cpu.l2bus.cpu_side_ports
    cpu.dcache.mem_side = cpu.l2bus.cpu_side_ports
    
    cpu.l2cache = L2Cache(size = args.l2_size)

    cpu.l2cache.cpu_side = cpu.l2bus.mem_side_ports

    cpu.l2cache.mem_side = system.l3bus.cpu_side_ports
    
# Conectar la caché L3 al bus de memoria principal
system.l3cache.mem_side = system.membus.cpu_side_ports

# Crear el controlador de memoria DDR4 y conectarlo al bus principal
system.mem_ctrl = MemCtrl(dram=DDR4_2400_16x4(range=system.mem_ranges[0]))
system.mem_ctrl.port = system.membus.mem_side_ports

# --- Fin de la jerarquía de cachés ---



# Crear un proceso para ARM
binary = args.binary[0]
process = ArmProcess()
process.executable = binary
if len(args.binary) > 1:
    process.cmd = [binary] + args.binary[1:]
else:
    process.cmd = [binary]

# Configurar el workload a nivel de sistema primero
system.workload = SEWorkload.init_compatible(binary)

# Asignar el proceso a cada CPU y crear los hilos
for cpu in system.cpu:
    cpu.workload = [process]  # Necesita ser una lista en versiones nuevas
    cpu.createThreads()

# Crear la raíz del sistema y realizar la instanciación
root = Root(full_system=False,system=system)
m5.instantiate()

# Función para cambiar de CPU en las regiones de interés
def roi_begin_handler():
    print("Taking stats from SCAMP")
    m5.stats.reset()  # Reinicia las estadísticas del ROI
    print("stats have been reset in tick {}!".format(m5.curTick()))
    #simulator.save_checkpoint("checkpoints")

# Función para cambiar de vuelta a Timing al final del ROI
def roi_end_handler():
    m5.stats.dump()  # Guarda las estadísticas del ROI
    print("stats have been dumped in tick {}!".format(m5.curTick()))

print("Comenzando simulación...")
exit_event = m5.simulate()

print("Simulación finalizada en tick {} por: {}".format(m5.curTick(), exit_event.getCause()))

# Sigue simulando hasta alcanzar un evento de finalización
checks = 0
while exit_event.getCause() != "exiting with last active thread context":
    if exit_event.getCause() == "checkpoint":
        if checks == 0:
            roi_begin_handler()
            checks += 1
        elif checks == 1:
            roi_end_handler()
    elif exit_event.getCause() == "user interrupt received":
        break
    exit_event = m5.simulate()

print("Simulación finalizada en tick {} por: {}".format(m5.curTick(), exit_event.getCause()))

#m5.checkpoint("checkpoints")
#print("Checkpoint guardado en 'checkpoints'")
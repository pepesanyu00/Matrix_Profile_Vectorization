# MPvect_RISC-V
Vectorización de algoritmos de análisis de series temporales (Matrix profile) en RISC-V.

El repositorio contiene una serie de directorios que contienen lo siguiente:

### Algoritmos

Es la carpeta en la que se encuentran los códigos fuente tanto de SCAMP como de SCRIMP, cada uno contiene 4 versiones, que son las siguientes:

    - scamp/scrimp.cpp #Versión base del algoritmo, sin vectorizar.
    - scamp/scrimp-v.cpp #Versión vectorizada para RISC-V del algoritmo.
    - scamp/scrimp-v-intel.cpp #Versión vectorizada para arquitecturas X86
    - scamp/scrimp-vomp.cpp #Versión autovectorizada con openMP con pragma SIMD.

Además contiene un Makefile para compilar cada una de las opciones, además de incluir opciones para compilar scamp para ejecutar con qemu (con sus flags específicos) y un archivo de prueba ligero para probar que gem5 funciona correctamente.

### Timeseries

Directorio en el que se almacenan las series temporales para los benchmarks. En el archivo launch.sh vienen todas definidas junto con el tamaño de ventana que precisa cada una.

### Stats

En este directorio se almacenan las estadísticas de gem5, su estructura está definida para guardar cada distribución (determinada por la arquitectura y si es secuencial o vectorizada.) en un directorio distinto. Por tanto, su estructura es la siguiente:

<arch> = intel | riscv, <mode> = sec | vect

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

Si la distribución es riscv-vect, dentro del directorio de cada serie temporal, habrá otros 7 directorios, uno por cada valor de vlen con el que se ha probado. Estos se llama scamp/scrimp-v-vlen-{vlen}, y dentro de ellos se encuentra el archivo stats.txt.

### gem5_config_riscv.py

Archivo de configuración gem5 para RISC-V, utiliza un procesador risc-vO3, y los parámetros que recibe son los siguientes:

    Binary: binario del archivo ejecutable junto con sus argumentos.
    -c o --cores: número de cores que va a tener el procesador. (default=16)
    -v o --vlen: tamaño del registro vectorial. (default=256)
    -e o --elen: tamaño del dato con el que se trata. (default=64)
    --l1i_size: tamaño de la caché l1i. (default=32KiB)
    --lid_size: tamaño de la caché l1d. (default=64KiB)
    --l2_size: tamaño de la caché l2. (default=256KiB)
    --l3_size: tamaño de la caché l3. (default=16MiB)

### gem5_config_x86.py

Igual que el archivo de configuración de RISC-V pero para X86. Los parámetros son los siguientes:


    Binary: binario del archivo ejecutable junto con sus argumentos.
    -c o --cores: número de cores que va a tener el procesador. (default=16)
    --l1i_size: tamaño de la caché l1i. (default=32KiB)
    --lid_size: tamaño de la caché l1d. (default=64KiB)
    --l2_size: tamaño de la caché l2. (default=256KiB)
    --l3_size: tamaño de la caché l3. (default=16MiB)

### launch.sh

Archivo que despliega las simulaciones de scamp. Este archivo despliega la simulación que le digas según la arquitectura (intel o riscv), el modo (sec, vect u omp), algoritmo (scamp o scrimp) y el número de threads, todos pasados como argumento. Los modos secuenciales y el modo vectorizado en X86 lanzan 6 simulaciones, ya que hay 6 series temporales con las que hacer las pruebas. En cambio, RISC-V Vectorizado lanza 42 simulaciones, ya que cada serie temporal prueba con 7 vlens distintos.

### generate_graphics.py

Archivo que genera gráficas a raíz de las estadísticas recogidas por gem5. Este coge las estadísticas del directorio stats, y crea gráficas a partir de ellos. Hay dos tipos de gráficas a generar:

- Vlen: son gráficas que estudian la evolución del speedup con las distintas distribuciones de vlen. Es una gráfica de líneas, en la que cada línea corresponde con un valor de vlen. Hay que especificar la carpeta de la que se quieren sacar las estadísticas (dentro de la carpeta stats), mediante el comando --vlen_dir.
- Comp: es una gráfica de comparación de distribuciones, lo que hace es medir el speedup de dos distribuciones y lo mete en una gráfica de líneas, en la que cada línea es una distribución. Ideal para hacer las gráficas de comparación de rendimiento entre arquitecturas, o entre la versión secuencial y la vectorizada.

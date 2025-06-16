#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <omp.h>
#include <unistd.h> //For getpid(), used to get the pid to generate a unique filename
#include <typeinfo> //To obtain type name as string
#include <array>
#include <assert.h> //RIC incluyo assert para hacer comprobaciones de invariantes y condiciones
//#include <immintrin.h>
#include <riscv_vector.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif 

#ifdef ENABLE_GEM5_ROI
#include <gem5/m5ops.h>
#endif


#define SHUFFLE_DIAGS true
#define DTYPE double              /* DATA TYPE */
#define ITYPE uint64_t       /* INDEX TYPE*/

/*******************************************************************/
#define VDTYPE vfloat64m1_t
#define VITYPE vuint64m1_t
#define VMTYPE vbool64_t

//  Reparten un valor por un registro vectorial
#define SETZERO_PD(vl) __riscv_vfmv_v_f_f64m1(0.0, vl)
#define SET1_PD(a, vl) __riscv_vfmv_v_f_f64m1(a, vl)
#define SET1_EPI(a, vl) __riscv_vmv_v_x_u64m1(a, vl)

// Extrae el primer valor de un registro vectorial y lo pone en un escalar float64_t (get), y viceversa (set), coge un escalar float64_t y lo mete en el primer valor del vector.
#define GETFIRST_PD(a) __riscv_vfmv_f_s_f64m1_f64(a)
#define SETFIRST_PD(a,vl) __riscv_vfmv_s_f_f64m1(a, vl)

// Extrae el máximo de un registro vectorial y lo pone en un escalar float64_t, la variable b es el valor inicial del máximo.
#define REDMAX_PD(a, b, vl) __riscv_vfredmax_vs_f64m1_f64m1(a, b, vl)

// Coge el índice del primer valor verdadero de una máscara, si no hay ningún 1 devuelve -1
#define GETFIRST_MASK(mask,vl) __riscv_vfirst_m_b64(mask, vl) 

//  Carga los elementos de un registro (float e int)
#define LOADU_PD(a, vl) __riscv_vle64_v_f64m1(a, vl)
#define LOADU_SI(a, vl) __riscv_vle64_v_u64m1(a, vl)

//  Guarda los elementos de un registro (float e int) en memoria. Para las unaligned, RVV no requiere alineación en la mayoría de sus instrucciones
#define STORE_PD(a, b, vl) __riscv_vse64_v_f64m1(a, b, vl)
#define STORE_SI(a, b, vl) __riscv_vse64_v_u64m1(a, b, vl)

//  hace el multiply-add de dos vectores y lo almacena en un tercero
#define FMADD_PD(a, b, c, vl) __riscv_vfmadd_vv_f64m1(a, b, c, vl)

// suma, resta y multiplicación de dos vectores
#define SUB_PD(a, b, vl) __riscv_vfsub_vv_f64m1(a, b, vl)
#define ADD_PD(a, b, vl) __riscv_vfadd_vv_f64m1(a, b, vl)
#define MUL_PD(a, b, vl) __riscv_vfmul_vv_f64m1(a, b, vl)

// Compara elemento a elemento dos vectores (a mayor que b, y a igual a b) y devuelve una máscara con 1 en los elementos que cumplen la condición
#define CMP_PD_GT(a, b, vl) __riscv_vmfgt_vv_f64m1_b64(a, b, vl)
#define CMP_PD_EQ(a, b, vl) __riscv_vmfeq_vf_f64m1_b64(a, b, vl)

// Combina dos operandos usando una máscara
#define BLEND_EPI(a, b, mask, vl) __riscv_vmerge_vvm_u64m1(a, b, mask, vl)
#define BLEND_PD(a, b, mask, vl) __riscv_vmerge_vvm_f64m1(a, b, mask, vl)

// Guarda elementos de 64 bits en memoria, pero sólo los que cumplen la máscara (PD para punto flotante y EPI para enteros)
#define MASKSTOREU_PD(mask, a, b, vl) __riscv_vse64_v_f64m1_m(mask, a, b, vl)
#define MASKSTOREU_EPI(mask, a, b, vl) __riscv_vse64_v_u64m1_m(mask, a, b, vl)

// Macros para creación de arrays
#define ARRAY_NEW(_type, _var, _elem) _var = new _type[_elem];

#define ARRAY_DEL(_var)      \
  assert(_var != NULL); \
  delete[] _var;

using namespace std;

/* ------------------------------------------------------------------ */

ITYPE numThreads, exclusionZone, windowSize, tSeriesLength, profileLength, percent_diags;
// Número de elementos a procesar en un ciclo de reloj
uint64_t VLMAX = __riscv_vsetvlmax_e64m1();  // Devuelve el máximo número de elementos de 64 bits que entran en el vector
//ITYPE vl = VLMAX;
// Private structures
// RIC las pongo alineadas tb
// vector<DTYPE> profile_tmp(profileLength * numThreads);
// vector<ITYPE> profileIndex_tmp(profileLength * numThreads);
DTYPE *profile_tmp = NULL;
ITYPE *profileIndex_tmp = NULL;


// Computes all required statistics for SCAMP, populating info with these values
void preprocess(DTYPE *tSeries, DTYPE *means, DTYPE *norms, DTYPE *df, DTYPE *dg, ITYPE tSeriesLength)
{

  vector<DTYPE> prefix_sum(tSeriesLength);
  vector<DTYPE> prefix_sum_sq(tSeriesLength);

  // Calculates prefix sum and square sum vectors
  prefix_sum[0] = tSeries[0];
  prefix_sum_sq[0] = tSeries[0] * tSeries[0];
  for (ITYPE i = 1; i < tSeriesLength; ++i)
  {
    prefix_sum[i] = tSeries[i] + prefix_sum[i - 1];
    prefix_sum_sq[i] = tSeries[i] * tSeries[i] + prefix_sum_sq[i - 1];
  }

  // Prefix sum value is used to calculate mean value of a given window, taking last value
  // of the window minus the first one and dividing by window size
  means[0] = prefix_sum[windowSize - 1] / static_cast<DTYPE>(windowSize);
  for (ITYPE i = 1; i < profileLength; ++i)
    means[i] = (prefix_sum[i + windowSize - 1] - prefix_sum[i - 1]) / static_cast<DTYPE>(windowSize);

  DTYPE sum = 0;
  for (ITYPE i = 0; i < windowSize; ++i)
  {
    DTYPE val = tSeries[i] - means[0];
    sum += val * val;
  }
  norms[0] = sum;

  // Calculates L2-norms (euclidean norm, euclidean distance)
  for (ITYPE i = 1; i < profileLength; ++i)
    norms[i] = norms[i - 1] + ((tSeries[i - 1] - means[i - 1]) + (tSeries[i + windowSize - 1] - means[i])) *
                                  (tSeries[i + windowSize - 1] - tSeries[i - 1]);
  for (ITYPE i = 0; i < profileLength; ++i)
    norms[i] = 1.0 / sqrt(norms[i]);

  // Calculates df and dg vectors
  for (ITYPE i = 0; i < profileLength - 1; ++i)
  {
    df[i] = (tSeries[i + windowSize] - tSeries[i]) / 2.0;
    dg[i] = (tSeries[i + windowSize] - means[i + 1]) + (tSeries[i] - means[i]);
  }
}

void scamp(DTYPE *tSeries, vector<ITYPE> &diags, DTYPE *means, DTYPE *norms, DTYPE *df, DTYPE *dg, DTYPE *profile, ITYPE *profileIndex)
{

#pragma omp parallel //proc_bind(spread)
  {
    // ITYPE
    ITYPE my_offset = omp_get_thread_num() * profileLength;
    ITYPE vlOuter = VLMAX,vlInner = VLMAX,vlRed = VLMAX;
    ITYPE Ndiags = (ITYPE)diags.size() * percent_diags / 100;

// Go through diagonals (dynamic)
#pragma omp for schedule(dynamic)
    for (ITYPE ri = 0; ri < Ndiags; ri++)
    /* Cada iteración lleva VLMAX elementos a la vez con las instrucciones vectoriales*/
    {
      ITYPE diag = diags[ri];
      vlOuter = min(VLMAX, profileLength - diag); // Ajusto el VL para que no se pase del tamaño del vector
      // covariance = 0; RIC tengo que crear una variable AVX para la covarianza
      VDTYPE covariance_v = SETZERO_PD(vlOuter); // 4 covarianzas empaquetadas
      /* En este bucle se realiza el producto vectorial de cada elemento de las ventanas
         que están en las posiciones diag y 0 (centrado en la media, de ahí el - mean()
         En nuestro caso, realizaremos el producto vectorial de la [diag, diag+1, diag+2, diag+3]
         por la 0, en paralelo, vectorizando.
      */
      VDTYPE meansWinDiag_v = LOADU_PD(&means[diag], vlOuter);
      VDTYPE meansWin0_v = SET1_PD(means[0], vlOuter);
      for (ITYPE i = 0; i < windowSize; i++)
      {
        //vlWin = min(VLMAX, windowSize - i); // Ajusto el VL para que no se pase del tamaño del vector
        // covariance += ((tSeries[diag + i] - means[diag]) * (tSeries[i] - means[0]));
        // assert(((uintptr_t)&tSeries[diag + i] & (uintptr_t)(VLEN - 1)) == 0);
        VDTYPE tSeriesWinDiag_v = LOADU_PD(&tSeries[diag + i], vlOuter); // Los afectados por diag se cargan de 4 en 4
        VDTYPE tSeriesWin0_v = SET1_PD(tSeries[i], vlOuter); // Los no afectados se replican
        // res = fma(a,b,c) -> res = a*b+c
        covariance_v = FMADD_PD(SUB_PD(tSeriesWinDiag_v, meansWinDiag_v, vlOuter), SUB_PD(tSeriesWin0_v, meansWin0_v, vlOuter), covariance_v, vlOuter);
      }

      ITYPE i = 0;
      ITYPE j = diag;
      // RIC La j es diag en realidad por lo que las normas j se cargan con load y las i con set1
      // correlation = covariance * norms[i] * norms[j];
      VDTYPE normsi_v = SET1_PD(norms[i], vlOuter);
      VDTYPE normsj_v = LOADU_PD(&norms[j], vlOuter);
      VDTYPE correlation_v = MUL_PD(MUL_PD(covariance_v, normsi_v, vlOuter), normsj_v, vlOuter);

      // Búsqueda horizontal del máximo
      // Coge el primer valor del vector resultante al aplicar el máximo entre el vector de correlación y el profile en la posición i+myoffset
      DTYPE corr_max =  GETFIRST_PD(  REDMAX_PD(correlation_v, SETFIRST_PD(profile_tmp[i + my_offset], vlOuter), vlOuter));
      // Comprobamos si el valor de correlación máxima está en el vector de correlaciones
      VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max,vlOuter);
      // Cogemos el primer elemento positivo de la máscara. Devuelve -1 si la máscara está completa a 0.
      long index_max = GETFIRST_MASK(mask,vlOuter);

      if(index_max != -1){
        profile_tmp[i + my_offset] = corr_max;
        profileIndex_tmp[i + my_offset] = j + index_max;      
      }

      VDTYPE profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlOuter);
      mask = CMP_PD_GT(correlation_v, profilej_v, vlOuter);
      MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], correlation_v, vlOuter);
      MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset], SET1_EPI(i, vlOuter), vlOuter);

      i = 1;
      for (ITYPE j = diag + 1; j < profileLength; j++)
      {
        vlInner = min(VLMAX, profileLength - j); // Ajusto el VL para que no se pase del tamaño del vector
        // empieza por i = 0, j = diag y continúa
        // covariance += (df[i - 1] * dg[j - 1] + df[j - 1] * dg[i - 1]); // paralelizable
        // RIC seguimos con que las js son diag y son las que se empaquetan, las ies se replican
        VDTYPE dfj_v = LOADU_PD(&df[j - 1], vlInner); // Los afectados por diag se cargan de 4 en 4
        VDTYPE dgj_v = LOADU_PD(&dg[j - 1], vlInner);
        VDTYPE dfi_v = SET1_PD(df[i - 1], vlInner); // Los no afectados se replican
        VDTYPE dgi_v = SET1_PD(dg[i - 1], vlInner);
        // res = fma(a,b,c) -> res = a*b+c
        covariance_v = ADD_PD(covariance_v, FMADD_PD(dfi_v, dgj_v, MUL_PD(dfj_v, dgi_v, vlInner), vlInner), vlInner);

        // correlation = covariance * norms[i] * norms[j];                // mas complicado
        // RIC como antes, lo que va afectado de i se replica y lo que va afectado de j se carga
        normsi_v = SET1_PD(norms[i], vlInner);
        normsj_v = LOADU_PD(&norms[j], vlInner);
        correlation_v = MUL_PD(MUL_PD(covariance_v, normsi_v, vlInner), normsj_v, vlInner);

        // Coge el primer valor del vector resultante al aplicar el máximo entre el vector de correlación y el profile en la posición i+myoffset
        corr_max =  GETFIRST_PD(  REDMAX_PD(correlation_v, SETFIRST_PD(profile_tmp[i + my_offset], vlInner), vlInner));
        // Comprobamos si el valor de correlación máxima está en el vector de correlaciones
        mask = CMP_PD_EQ(correlation_v, corr_max,vlInner);
        // Cogemos el primer elemento positivo de la máscara. Devuelve -1 si la máscara está completa a 0.
        index_max = GETFIRST_MASK(mask,vlInner);

        if(index_max != -1){
          profile_tmp[i + my_offset] = corr_max;
          profileIndex_tmp[i + my_offset] = j + index_max;      
        }


        profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlInner);
        mask = CMP_PD_GT(correlation_v, profilej_v, vlInner);
        MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], correlation_v, vlInner);
        MASKSTOREU_EPI(mask,&profileIndex_tmp[j + my_offset], SET1_EPI(i, vlInner), vlInner);

        i++;
      }
    } //'pragma omp for' places here a barrier unless 'no wait' is specified

// Reduction. RIC Vectorizo el bucle externo (colum++ por colum+=VLMAX)
#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < profileLength; colum += VLMAX)
    {
      // max_corr = -numeric_limits<DTYPE>::infinity();
      vlRed = min(VLMAX, profileLength - colum); // Ajusto el VL para que no se pase del tamaño del vector
      VDTYPE max_corr_v = SET1_PD(-numeric_limits<DTYPE>::infinity(), vlRed);
      VITYPE max_indices_v = SET1_EPI(-1, vlRed);
      for (ITYPE th = 0; th < numThreads; th++)
      {
        // RIC Como el profileLength puede tener cualquier valor no puedo hacer los load alineados
        VDTYPE profile_tmp_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE profileIndex_tmp_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_GT(profile_tmp_v, max_corr_v, vlRed);
        max_indices_v = BLEND_EPI(max_indices_v, profileIndex_tmp_v, mask, vlRed); // Update con máscara de los índices
        max_corr_v = BLEND_PD(max_corr_v, profile_tmp_v, mask, vlRed);             // Update con máscara de las correlaciones
      }
      STORE_PD(&profile[colum], max_corr_v, vlRed);
      STORE_SI(&profileIndex[colum], max_indices_v, vlRed);
    }
  }
}

int main(int argc, char *argv[])
{
  try
  {
    // Creation of time meassure structures
    chrono::steady_clock::time_point tstart, tend;
    chrono::duration<double> telapsed;
    cout << "VLMAX: " << VLMAX << endl;
    if (argc != 6)
    {
      cout << "[ERROR] usage: ./scamp input_file win_size num_threads percent_diags out_directory" << endl;
      return 0;
    }

    windowSize = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    percent_diags = atoi(argv[4]);
    string outdir = argv[5];
    // Set the exclusion zone to 0.25
    exclusionZone = (ITYPE)(windowSize * 0.25);
    omp_set_num_threads(numThreads);

    // vector<DTYPE> tSeriesV; Usamos arrays
    string inputfilename = argv[1];
    string alg = argv[0];
    alg = alg.substr(2);
    stringstream tmp;
    // RIC Ahora permito que la timeseries se introduzca con el directorio
    // RIC Quito el directorio de la cadena de resultados con rfind('/') y meto el nombre del programa al principio del nombre del fichero de resultados
    tmp << outdir << alg.substr(alg.rfind('/') +1) << "_" << inputfilename.substr(inputfilename.rfind('/') + 1, inputfilename.size() - 4 - inputfilename.rfind('/') - 1) << "_w" << windowSize << "_t" << numThreads << "_pdiags" << percent_diags << "_" << getpid() << ".csv";
    string outfilename = tmp.str();

    // Display info through console
    cout << endl;
    cout << "############################################################" << endl;
    cout << "///////////////////////// SCAMP ////////////////////////////" << endl;
    cout << "############################################################" << endl;
    cout << endl;
    cout << "[>>] Reading File: " << inputfilename << "..." << endl;

    /* ------------------------------------------------------------------ */
    /* Count file lines */
    tstart = chrono::steady_clock::now();

    fstream tSeriesFile(inputfilename, ios_base::in);
    tSeriesLength = 0;
    cout << "[>>] Counting lines ... " << endl;
    string line;
    while (getline(tSeriesFile, line)) // Cuento el número de líneas
      tSeriesLength++;

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Lines: " << tSeriesLength << " Time: " << telapsed.count() << "s." << endl;
    /* ------------------------------------------------------------------ */
    /* Read time series file */
    cout << "[>>] Reading values ... " << endl;
    tstart = chrono::steady_clock::now();
    DTYPE *tSeries = NULL; // Defino la serie temporal como un puntero a DTYPE
    // Le sumo a la longitud el VLMAX para que cuando se trabaje en los límites de la serie se cojan elementos reservados (aunque luego no sirvan los cálculos que se hacen con ellos)
    ARRAY_NEW(DTYPE, tSeries, tSeriesLength);
    tSeriesFile.clear();                // Limpio el stream
    tSeriesFile.seekg(tSeriesFile.beg); // Y lo reinicio a beginning
    DTYPE tempval, tSeriesMin = numeric_limits<DTYPE>::infinity(), tSeriesMax = -numeric_limits<double>::infinity();
    // Comprobar si el fichero tiene algún NaN y quitarlo, porque esto podría fallar
    for (int i = 0; tSeriesFile >> tempval; i++)
    {
      tSeries[i] = tempval;
      if (tempval < tSeriesMin)
        tSeriesMin = tempval;
      if (tempval > tSeriesMax)
        tSeriesMax = tempval;
    }
    tSeriesFile.close();
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    /* ------------------------------------------------------------------ */
    // Set Matrix Profile Length
    profileLength = tSeriesLength - windowSize + 1;

    // Defino los arrays con las macros creadas para que salgan alineados (aunque finalmente uso loadu, y probablemente no haga falta usarlas)
    // Sumo a la longitud VLMAX para que en los límites se trabaje con memoria reservada aunque no se utilicen los datos
    DTYPE *norms = NULL, *means = NULL, *df = NULL, *dg = NULL, *profile = NULL;
    ITYPE *profileIndex = NULL;
    vector<ITYPE> diags;
    ARRAY_NEW(DTYPE, norms, profileLength);
    ARRAY_NEW(DTYPE, means, profileLength);
    ARRAY_NEW(DTYPE, df, profileLength);
    ARRAY_NEW(DTYPE, dg, profileLength);
    ARRAY_NEW(DTYPE, profile, profileLength);
    ARRAY_NEW(ITYPE, profileIndex, profileLength);
    // RIC sumo VLMAX al profileLength
    ARRAY_NEW(DTYPE, profile_tmp, profileLength * numThreads);
    ARRAY_NEW(ITYPE, profileIndex_tmp, profileLength * numThreads);
    // Private profile initialization
    for (ITYPE i = 0; i < (profileLength * numThreads); i++)
      profile_tmp[i] = -numeric_limits<DTYPE>::infinity();

    cout << "[OK] Time: " << telapsed.count() << "s." << endl;

    // Display info through console
    cout << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "************************** INFO ****************************" << endl;
    cout << endl;
    cout << " Series/MP data type: " << typeid(tSeries[0]).name() << "(" << sizeof(tSeries[0]) << "B)" << endl;
    cout << " Index data type:     " << typeid(profileIndex[0]).name() << "(" << sizeof(profileIndex[0]) << "B)" << endl;
    cout << " Time series length:  " << tSeriesLength << endl;
    cout << " Window size:         " << windowSize << endl;
    cout << " Time series min:     " << tSeriesMin << endl;
    cout << " Time series max:     " << tSeriesMax << endl;
    cout << " Number of threads:   " << numThreads << endl;
    cout << " Exclusion zone:      " << exclusionZone << endl;
    cout << " Profile length:      " << profileLength << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << endl;

    /***************** Preprocess ******************/
    cout << "[>>] Preprocessing..." << endl;
    tstart = chrono::steady_clock::now();
    preprocess(tSeries, means, norms, df, dg, tSeriesLength);
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Preprocessing Time:         " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    /***********************************************/

    // Random shuffle the diagonals
    diags.clear();
    for (ITYPE i = exclusionZone + 1; i < profileLength; i += VLMAX)
      diags.push_back(i);

    if (SHUFFLE_DIAGS){
      //random_device rd;
      mt19937 g(0);
      shuffle(diags.begin(), diags.end(), g);
    }
    

    /******************** SCAMP ********************/
    cout << "[>>] Executing SCAMP..." << endl;
    tstart = chrono::steady_clock::now();

    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_begin();
    #endif


    // Establish begining of ROI
    #ifdef ENABLE_GEM5_ROI
    m5_work_begin(0,0);
    #endif

    scamp(tSeries, diags ,means, norms, df, dg, profile, profileIndex);
    
    // Establish end of ROI    
    #ifdef ENABLE_GEM5_ROI
    m5_work_end(0,0);
    #endif
    
    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_end();
    #endif
    
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] SCAMP Time:              " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    /***********************************************/

    cout << "[>>] Saving result: " << outfilename << " ..." << endl;
    fstream statsFile(outfilename, ios_base::out);
    statsFile << "# Time (s)" << endl;
    statsFile << setprecision(9) << telapsed.count() << endl;
    statsFile << "# Profile Length" << endl;
    statsFile << profileLength << endl;
    statsFile << "# i,tseries,profile,index" << endl;
    for (ITYPE i = 0; i < profileLength; i++)
      statsFile << i << "," << tSeries[i] << ","  << (DTYPE)sqrt(2 * windowSize * (1 - profile[i])) << "," << profileIndex[i] << endl;
    statsFile.close();

    cout << endl;

    // Libero memoria
    ARRAY_DEL(tSeries);
    ARRAY_DEL(norms);
    ARRAY_DEL(means);
    ARRAY_DEL(df);
    ARRAY_DEL(dg);
    ARRAY_DEL(profile);
    ARRAY_DEL(profileIndex);
    ARRAY_DEL(profile_tmp);
    ARRAY_DEL(profileIndex_tmp);
  }
  catch (exception &e)
  {
    cout << "Exception: " << e.what() << endl;
  }
}

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
#include <unistd.h>      // Para getpid()
#include <typeinfo>
#include <array>
#include <cassert>
#include <arm_sve.h>     // Incluir intrínsecos SVE

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif 

#ifdef ENABLE_GEM5_ROI
#include <gem5/m5ops.h>
#endif

#define SHUFFLE_DIAGS true
#define DTYPE double              /* DATA TYPE */
#define ITYPE uint64_t       /* INDEX TYPE */

// Tipos SVE: para double, enteros sin signo de 64 bits y máscaras
#define VDTYPE svfloat64_t
#define VITYPE svuint64_t
#define VMTYPE svbool_t

// ================================================================
// Funciones auxiliares para SVE

// Obtiene el número máximo de elementos de 64 bits que contiene cada registro vectorial
uint64_t VLMAX = svcntd();

// Crea un vector de double con todos los elementos iguales a 0.0
#define SETZERO_PD() svdup_f64(0.0)

// Crea un vector de double con todos los elementos iguales a "a"
static inline svfloat64_t set1_pd(double a,int vl) {
  // Vector en el que se guardará el elemento a
  VDTYPE inactive = svdup_f64(0.0);
  // Se crea el vector con valor a y predicado para procesar vl elementos
  return svdup_f64_m(inactive,svwhilelt_b64(0, vl),a);

}
#define SET1_PD(a, vl) set1_pd(a,vl)

// Crea un vector de enteros de 64 bits con todos los elementos iguales a "a"
static inline svuint64_t set1_epi(double a, int vl) {
  // Vector en el que se guardará el elemento a
  VITYPE inactive = svdup_u64(0.0);
  // Se crea el vector con valor a y predicado para procesar vl elementos
  return svdup_u64_m(inactive,svwhilelt_b64(0, vl),a);

}
#define SET1_EPI(a, vl) set1_epi(a,vl)

// Extrae el primer valor de un vector SVE de doubles mediante una función inline
static inline double get_first_pd(svfloat64_t vec) {
    int n = svcntd(); // número de lanes para double
    DTYPE tmp[n];
    // Se almacena el vector completo; usamos la máscara “todos activos”
    svst1_f64(svptrue_b64(), tmp, vec);
    return tmp[0];
}
#define GETFIRST_PD(a) get_first_pd(a)



// Realiza una reducción horizontal de máximo; se combina el vector "a" con un vector duplicado de "b" y se reduce
#define REDMAX_PD(a, b, vl) svmaxv_f64(svwhilelt_b64(0, vl), svmax_f64_m(svwhilelt_b64(0, vl), a, svdup_f64(b)))

// Función auxiliar para extraer el primer índice activo (con valor 1) de una máscara SVE.
// Debido a que SVE no ofrece una extracción directa de la máscara, se almacena la máscara en un arreglo y se recorre.
static inline long get_first_mask(svbool_t mask, int vl) {
    // Simple, direct approach to find first true element
    for (int i = 0; i < vl; i++) {
        uint64_t is_true = svptest_first(svwhilelt_b64(i, i+1), mask);
        if (is_true) return i;
    }
    return -1;
}
#define GETFIRST_MASK(mask, vl) get_first_mask(mask, vl)

// Carga en un vector SVE de doubles (carga “unalineada”) usando una máscara que activa los "vl" elementos
#define LOADU_PD(a, vl) svld1_f64(svwhilelt_b64(0, vl), a)

// Carga en un vector SVE de enteros de 64 bits
#define LOADU_SI(a, vl) svld1_u64(svwhilelt_b64(0, vl), a)

// Almacena en memoria desde un vector SVE de doubles
#define STORE_PD(a, b, vl) svst1_f64(svwhilelt_b64(0, vl), a, b)

// Almacena en memoria desde un vector SVE de enteros de 64 bits
#define STORE_SI(a, b, vl) svst1_u64(svwhilelt_b64(0, vl), a, b)

// Realiza la operación fused multiply-add: calcula (a * b + c)
#define FMADD_PD(a, b, c, vl) svmla_f64_z(svwhilelt_b64(0, vl), c, a, b)

// Suma, resta y multiplicación elemental de dos vectores de doubles
#define ADD_PD(a, b, vl) svadd_f64_z(svwhilelt_b64(0, vl), a, b)
#define SUB_PD(a, b, vl) svsub_f64_z(svwhilelt_b64(0, vl), a, b)
#define MUL_PD(a, b, vl) svmul_f64_z(svwhilelt_b64(0, vl), a, b)

// Compara elemento a elemento dos vectores de doubles: mayor que
#define CMP_PD_GT(a, b, vl) svcmpgt_f64(svwhilelt_b64(0, vl),a, b)

// Compara elemento a elemento: igualdad con un escalar (se duplica el escalar "b")
#define CMP_PD_EQ(a, b, vl) svcmpeq_f64(svwhilelt_b64(0, vl) ,a, svdup_f64(b))

// Combina dos vectores según una máscara: para enteros
#define BLEND_EPI(a, b, mask) svsel(mask, b, a)
// Combina dos vectores según una máscara: para doubles
#define BLEND_PD(a, b, mask) svsel(mask, b, a)

// Almacena en memoria solo los elementos de un vector de doubles que cumplen la máscara
#define MASKSTOREU_PD(mask, a, b) svst1_f64(mask, a, b)
// Almacena en memoria solo los elementos de un vector de enteros que cumplen la máscara
#define MASKSTOREU_EPI(mask, a, b) svst1_u64(mask, a, b)


// Macros para creación y liberación de arrays (se usan igual)
#define ARRAY_NEW(_type, _var, _elem) _var = new _type[_elem];
#define ARRAY_DEL(_var)      \
  assert(_var != NULL);      \
  delete[] _var;

using namespace std;

// ------------------------------------------------------------------
// Variables globales
ITYPE numThreads, exclusionZone, windowSize, tSeriesLength, profileLength, percent_diags;

// Variables temporales privadas (se definen posteriormente en main)
DTYPE *profile_tmp = NULL;
ITYPE *profileIndex_tmp = NULL;

// ------------------------------------------------------------------
// Función de preprocesamiento: calcula estadísticas necesarias para SCAMP
void preprocess(DTYPE *tSeries, DTYPE *means, DTYPE *norms, DTYPE *df, DTYPE *dg, ITYPE tSeriesLength)
{
  vector<DTYPE> prefix_sum(tSeriesLength);
  vector<DTYPE> prefix_sum_sq(tSeriesLength);

  prefix_sum[0] = tSeries[0];
  prefix_sum_sq[0] = tSeries[0] * tSeries[0];
  for (ITYPE i = 1; i < tSeriesLength; ++i) {
    prefix_sum[i] = tSeries[i] + prefix_sum[i - 1];
    prefix_sum_sq[i] = tSeries[i] * tSeries[i] + prefix_sum_sq[i - 1];
  }

  means[0] = prefix_sum[windowSize - 1] / static_cast<DTYPE>(windowSize);
  for (ITYPE i = 1; i < profileLength; ++i)
    means[i] = (prefix_sum[i + windowSize - 1] - prefix_sum[i - 1]) / static_cast<DTYPE>(windowSize);

  DTYPE sum = 0;
  for (ITYPE i = 0; i < windowSize; ++i) {
    DTYPE val = tSeries[i] - means[0];
    sum += val * val;
  }
  norms[0] = sum;

  for (ITYPE i = 1; i < profileLength; ++i)
    norms[i] = norms[i - 1] + ((tSeries[i - 1] - means[i - 1]) + (tSeries[i + windowSize - 1] - means[i])) *
                              (tSeries[i + windowSize - 1] - tSeries[i - 1]);
  for (ITYPE i = 0; i < profileLength; ++i)
    norms[i] = 1.0 / sqrt(norms[i]);

  for (ITYPE i = 0; i < profileLength - 1; ++i) {
    df[i] = (tSeries[i + windowSize] - tSeries[i]) / 2.0;
    dg[i] = (tSeries[i + windowSize] - means[i + 1]) + (tSeries[i] - means[i]);
  }
}

// ------------------------------------------------------------------
// Función SCAMP: cálculo vectorizado con intrínsecos SVE
void scamp(DTYPE *tSeries, vector<ITYPE> &diags, DTYPE *means, DTYPE *norms, DTYPE *df, DTYPE *dg, DTYPE *profile, ITYPE *profileIndex)
{
#pragma omp parallel
  {
    ITYPE my_offset = omp_get_thread_num() * profileLength;
    int vlOuter = VLMAX, vlInner = VLMAX, vlRed = VLMAX;
    ITYPE Ndiags = (ITYPE)diags.size() * percent_diags / 100;

    // Recorre las diagonales (dinámicamente)
#pragma omp for schedule(dynamic)
    for (ITYPE ri = 0; ri < Ndiags; ri++) {
      ITYPE diag = diags[ri];
      vlOuter = min((ITYPE)VLMAX, profileLength - diag);
      VDTYPE covariance_v = SETZERO_PD();
      for (ITYPE i = 0; i < windowSize; i++) {
        VDTYPE tSeriesWinDiag_v = LOADU_PD(&tSeries[diag + i], vlOuter);
        VDTYPE meansWinDiag_v   = LOADU_PD(&means[diag], vlOuter);
        VDTYPE tSeriesWin0_v    = SET1_PD(tSeries[i], vlOuter);
        VDTYPE meansWin0_v      = SET1_PD(means[0], vlOuter);
        covariance_v = FMADD_PD( SUB_PD(tSeriesWinDiag_v, meansWinDiag_v, vlOuter),
                                 SUB_PD(tSeriesWin0_v, meansWin0_v, vlOuter),
                                 covariance_v, vlOuter);
      }

      ITYPE i = 0;
      // j se inicializa con diag
      ITYPE j = diag;
      VDTYPE normsi_v = SET1_PD(norms[i], vlOuter);
      VDTYPE normsj_v = LOADU_PD(&norms[j], vlOuter);
      VDTYPE correlation_v = MUL_PD( MUL_PD(covariance_v, normsi_v, vlOuter), normsj_v, vlOuter);

      // // Reducción horizontal: hallar el máximo
      DTYPE corr_max = REDMAX_PD(correlation_v, profile_tmp[i + my_offset], vlOuter);
      VMTYPE mask = CMP_PD_EQ(correlation_v, corr_max, vlOuter);
      long index_max = GETFIRST_MASK(mask, vlOuter);

      if(index_max != -1) {
        profile_tmp[i + my_offset] = corr_max;
        profileIndex_tmp[i + my_offset] = j + index_max;      
      }

      VDTYPE profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlOuter);
      mask = CMP_PD_GT(correlation_v, profilej_v, vlOuter);
      MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], correlation_v);
      MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset], SET1_EPI(i, vlOuter));

      i = 1;
      for (ITYPE j = diag + 1; j < profileLength; j++) {
        vlInner = min((ITYPE)VLMAX, profileLength - j);
        VDTYPE dfj_v = LOADU_PD(&df[j - 1], vlInner);
        VDTYPE dgj_v = LOADU_PD(&dg[j - 1], vlInner);
        VDTYPE dfi_v = SET1_PD(df[i - 1], vlInner);
        VDTYPE dgi_v = SET1_PD(dg[i - 1], vlInner);
        covariance_v = ADD_PD(covariance_v,
                              FMADD_PD(dfi_v, dgj_v, MUL_PD(dfj_v, dgi_v, vlInner), vlInner),
                              vlInner);
        normsi_v = SET1_PD(norms[i], vlInner);
        normsj_v = LOADU_PD(&norms[j], vlInner);
        correlation_v = MUL_PD( MUL_PD(covariance_v, normsi_v, vlInner), normsj_v, vlInner);

         corr_max = REDMAX_PD(correlation_v, profile_tmp[i + my_offset], vlInner);
         mask = CMP_PD_EQ(correlation_v, corr_max, vlInner);
         index_max = GETFIRST_MASK(mask, vlInner);
         
         if(index_max != -1) {
           profile_tmp[i + my_offset] = corr_max;
           profileIndex_tmp[i + my_offset] = j + index_max;      
         }

        profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlInner);
        mask = CMP_PD_GT(correlation_v, profilej_v, vlInner);
        MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], correlation_v);
        MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset], SET1_EPI(i, vlInner));

        i++;
      }
    } // Fin de omp for (implícita barrera)

    // Reducción final en paralelo: se recorre la columna en pasos de VLMAX
#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < profileLength; colum += VLMAX) {
      vlRed = min((ITYPE)VLMAX, profileLength - colum);
      VDTYPE max_corr_v = SET1_PD(-numeric_limits<DTYPE>::infinity(), vlRed);
      VITYPE max_indices_v = SET1_EPI(-1, vlRed);
      for (ITYPE th = 0; th < numThreads; th++) {
        VDTYPE profile_tmp_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE profileIndex_tmp_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_GT(profile_tmp_v, max_corr_v, vlRed);
        max_indices_v = BLEND_EPI(max_indices_v, profileIndex_tmp_v, mask);
        max_corr_v = BLEND_PD(max_corr_v, profile_tmp_v, mask);
      }
      STORE_PD(&profile[colum], max_corr_v, vlRed);
      STORE_SI(&profileIndex[colum], max_indices_v, vlRed);
    }
  }
}

// ------------------------------------------------------------------
// Función principal
int main(int argc, char *argv[])
{
  try {
    chrono::steady_clock::time_point tstart, tend;
    chrono::duration<double> telapsed;

    if (argc != 6) {
      cout << "[ERROR] usage: ./scamp input_file win_size num_threads percent_diags out_directory" << endl;
      return 0;
    }

    windowSize = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    percent_diags = atoi(argv[4]);
    string outdir = argv[5];
    exclusionZone = (ITYPE)(windowSize * 0.25);
    omp_set_num_threads(numThreads);

    string inputfilename = argv[1];
    string alg = argv[0];
    alg = alg.substr(2);
    stringstream tmp;
    tmp << outdir << alg.substr(alg.rfind('/') +1) << "_" 
        << inputfilename.substr(inputfilename.rfind('/') + 1, inputfilename.size() - 4 - inputfilename.rfind('/') - 1)
        << "_w" << windowSize << "_t" << numThreads << "_pdiags" << percent_diags 
        << "_" << getpid() << ".csv";
    string outfilename = tmp.str();
    cout << "VLMAX" << VLMAX << endl;
    cout << endl;
    cout << "############################################################" << endl;
    cout << "///////////////////////// SCAMP ////////////////////////////" << endl;
    cout << "############################################################" << endl;
    cout << endl;
    cout << "[>>] Reading File: " << inputfilename << "..." << endl;

    tstart = chrono::steady_clock::now();
    fstream tSeriesFile(inputfilename, ios_base::in);
    tSeriesLength = 0;
    cout << "[>>] Counting lines ... " << endl;
    string line;
    while (getline(tSeriesFile, line))
      tSeriesLength++;
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Lines: " << tSeriesLength << " Time: " << telapsed.count() << "s." << endl;

    cout << "[>>] Reading values ... " << endl;
    tstart = chrono::steady_clock::now();
    DTYPE *tSeries = NULL;
    ARRAY_NEW(DTYPE, tSeries, tSeriesLength);
    tSeriesFile.clear();
    tSeriesFile.seekg(tSeriesFile.beg);
    DTYPE tempval, tSeriesMin = numeric_limits<DTYPE>::infinity(), tSeriesMax = -numeric_limits<DTYPE>::infinity();
    for (int i = 0; tSeriesFile >> tempval; i++) {
      tSeries[i] = tempval;
      if (tempval < tSeriesMin)
        tSeriesMin = tempval;
      if (tempval > tSeriesMax)
        tSeriesMax = tempval;
    }
    tSeriesFile.close();
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;

    profileLength = tSeriesLength - windowSize + 1;

    DTYPE *norms = NULL, *means = NULL, *df = NULL, *dg = NULL, *profile = NULL;
    ITYPE *profileIndex = NULL;
    vector<ITYPE> diags;
    ARRAY_NEW(DTYPE, norms, profileLength);
    ARRAY_NEW(DTYPE, means, profileLength);
    ARRAY_NEW(DTYPE, df, profileLength);
    ARRAY_NEW(DTYPE, dg, profileLength);
    ARRAY_NEW(DTYPE, profile, profileLength);
    ARRAY_NEW(ITYPE, profileIndex, profileLength);
    ARRAY_NEW(DTYPE, profile_tmp, profileLength * numThreads);
    ARRAY_NEW(ITYPE, profileIndex_tmp, profileLength * numThreads);

    for (ITYPE i = 0; i < (profileLength * numThreads); i++)
      profile_tmp[i] = -numeric_limits<DTYPE>::infinity();

    cout << "[OK] Time: " << telapsed.count() << "s." << endl;
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

    cout << "[>>] Preprocessing..." << endl;
    tstart = chrono::steady_clock::now();
    preprocess(tSeries, means, norms, df, dg, tSeriesLength);
    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Preprocessing Time:         " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    diags.clear();
    for (ITYPE i = exclusionZone + 1; i < profileLength; i+=VLMAX)
      diags.push_back(i);
    if (SHUFFLE_DIAGS) {
      //random_device rd;
      mt19937 g(0);
      shuffle(diags.begin(), diags.end(), g);
    }
    
    cout << "[>>] Executing SCAMP..." << endl;
    tstart = chrono::steady_clock::now();

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

// Para RISC-V se hace con m5_work_begin 
#ifdef ENABLE_GEM5_ROI
    m5_checkpoint(0,0);
#endif

    scamp(tSeries, diags, means, norms, df, dg, profile, profileIndex);
    
#ifdef ENABLE_GEM5_ROI
    m5_checkpoint(0,0);
#endif
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] SCAMP Time:              " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;

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
  catch (exception &e) {
    cout << "Exception: " << e.what() << endl;
  }
}
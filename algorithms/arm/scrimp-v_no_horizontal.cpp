#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <unistd.h> //For getpid(), used to get the pid to generate a unique filename
#include <typeinfo> //To obtain type name as string
#include <array>
#include <assert.h> //RIC incluyo assert para hacer comprobaciones de invariantes y condiciones
//#include <immintrin.h>
#include <arm_sve.h>     // Incluir intrínsecos SVE

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif 

#ifdef ENABLE_GEM5_ROI
#include <gem5/m5ops.h>
#endif

#define SHUFFLE_DIAGS true
#define DTYPE double        /* DATA TYPE */
#define ITYPE uint64_t /* INDEX TYPE: RIC pongo long long int para que tanto el double como el int sean de 64 bits (facilita la vectorización) */

/*******************************************************************/ 
#define VDTYPE svfloat64_t
#define VITYPE svuint64_t
#define VMTYPE svbool_t


// Obtiene el número máximo de elementos de 64 bits (número de lanes para double)
uint64_t VLMAX = svcntd();

//  Reparten un valor por un registro vectorial
#define SETZERO_PD() svdup_f64(0.0)


// Crea un vector de double con todos los elementos iguales a "a"
static inline svfloat64_t set1_pd(double a,svbool_t vl) {
  // Vector en el que se guardará el elemento a
  VDTYPE inactive = svdup_f64(0.0);
  // Se crea el vector con valor a y predicado para procesar vl elementos
  return svdup_f64_m(inactive,vl,a);

}
#define SET1_PD(a, vl) set1_pd(a,vl)

// Crea un vector de enteros de 64 bits con todos los elementos iguales a "a"
static inline svuint64_t set1_epi(double a, svbool_t vl) {
  // Vector en el que se guardará el elemento a
  VITYPE inactive = svdup_u64(0.0);
  // Se crea el vector con valor a y predicado para procesar vl elementos
  return svdup_u64_m(inactive,vl,a);

}
#define SET1_EPI(a, vl) set1_epi(a,vl)

//  Carga los elementos de un registro (float e int)
#define LOADU_PD(a, vl) svld1_f64(vl, a)
#define LOADU_SI(a, vl) svld1_u64(vl, a)

// Extrae el primer valor de un registro vectorial y lo pone en un escalar float64_t (get), y viceversa (set), coge un escalar float64_t y lo mete en el primer valor del vector.
// Extrae el primer valor de un vector SVE de doubles mediante una función inline
static inline double get_first_pd(svfloat64_t vec) {
    int n = svcntd(); // número de lanes para double
    DTYPE tmp[n];
    // Se almacena el vector completo; usamos la máscara “todos activos”
    svst1_f64(svptrue_b64(), tmp, vec);
    return tmp[0];
}
#define GETFIRST_PD(a) get_first_pd(a)

// Extrae el mínimo de un registro vectorial y lo pone en un escalar float64_t, la variable b es el valor inicial del máximo.
#define REDMIN_PD(a, b, vl) svminv_f64(vl, svmin_f64_m(vl, a, svdup_f64(b)))

// Coge el índice del primer valor verdadero de una máscara, si no hay ningún 1 devuelve -1
static inline uint64_t get_first_mask(svbool_t mask, svbool_t vl) {
  svbool_t first_true = svbrkb_z(vl, mask);
  ITYPE index = svcntp_b64(vl, first_true);
  if (index == svcntp_b64(vl,vl))
    return -1;
  return index;
}
#define GETFIRST_MASK(mask, vl) get_first_mask(mask, vl)

//  Guarda los elementos de un registro (float e int) en memoria. Para las unaligned, RVV no requiere alineación en la mayoría de sus instrucciones
#define STORE_PD(a, b, vl) svst1_f64(vl, a, b)
#define STORE_SI(a, b, vl) svst1_u64(vl, a, b)
//  hace el multiply-add de dos vectores y lo almacena en un tercero
#define FMADD_PD(a, b, c, vl) svmla_f64_z(vl, c, a, b)
#define FMSUB_PD(a,b,c, vl) svsub_f64_z(vl, svmul_f64_z(vl, a, b), c)
// suma, resta y multiplicación de dos vectores
#define SUB_PD(a, b, vl) svsub_f64_z(vl, a, b)
#define ADD_PD(a, b, vl) svadd_f64_z(vl, a, b)
#define MUL_PD(a, b, vl) svmul_f64_z(vl, a, b)
#define DIV_PD(a, b, vl) svdiv_f64_z(vl, a, b)
// Compara elemento a elemento dos vectores (a mayor que b) y devuelve una máscara con 1 en los elementos que cumplen la condición
#define CMP_PD_LT(a, b, vl) svcmplt_f64(vl,a, b)
#define CMP_PD_EQ(a, b, vl) svcmpeq_f64(vl ,a, svdup_f64(b))
// Combina dos operandos usando una máscara
#define BLEND_EPI(a, b, mask) svsel(mask, b, a)
#define BLEND_PD(a, b, mask) svsel(mask, b, a)
// Guarda elementos de 64 bits en memoria, pero sólo los que cumplen la máscara (PD para punto flotante y EPI para enteros)
#define MASKSTOREU_PD(mask, a, b) svst1_f64(mask, a, b)
#define MASKSTOREU_EPI(mask, a, b) svst1_u64(mask, a, b)



// Macros para creación de arrays
#define ARRAY_NEW(_type, _var, _elem) _var = new _type[_elem];


#define ARRAY_DEL(_var)      \
  assert(_var != NULL); \
  delete[] _var;

using namespace std;

ITYPE numThreads, exclusionZone;
ITYPE windowSize, tSeriesLength, profileLength, percent_diags;


// Private structures
// vector<DTYPE> profile_tmp(profileLength * numThreads);
// vector<ITYPE> profileIndex_tmp(profileLength * numThreads);
DTYPE *profile_tmp = NULL;
ITYPE *profileIndex_tmp = NULL;

void preprocess(DTYPE *tSeries, DTYPE *means,
                DTYPE *devs)
{
  DTYPE *ACumSum = new DTYPE[tSeriesLength];
  DTYPE *ASqCumSum = new DTYPE[tSeriesLength];
  DTYPE *ASum = new DTYPE[profileLength];
  DTYPE *ASumSq = new DTYPE[profileLength];
  DTYPE *ASigmaSq = new DTYPE[profileLength];

  ACumSum[0] = tSeries[0];
  ASqCumSum[0] = tSeries[0] * tSeries[0];

  // means.clear();
  // devs.clear();
  // RIC No hace falta inicializarlos porque se machacan
  /*for (ITYPE i = 0; i < profileLength + VLMAX; i++)
  {
    means[i] = 0;
    devs[i] = 0;
  }*/

  // Cummulate sum
  for (ITYPE i = 1; i < tSeriesLength; i++)
  {
    ACumSum[i] = tSeries[i] + ACumSum[i - 1];
    ASqCumSum[i] = tSeries[i] * tSeries[i] + ASqCumSum[i - 1];
  }

  ASum[0] = ACumSum[windowSize - 1];
  ASumSq[0] = ASqCumSum[windowSize - 1];

  for (ITYPE i = 0; i < tSeriesLength - windowSize; i++)
  {
    ASum[i + 1] = ACumSum[windowSize + i] - ACumSum[i];
    ASumSq[i + 1] = ASqCumSum[windowSize + i] - ASqCumSum[i];
  }

  for (ITYPE i = 0; i < profileLength; i++)
  {
    // means.push_back(ASum[i] / windowSize);
    means[i] = (ASum[i] / windowSize);
    ASigmaSq[i] = ASumSq[i] / windowSize - means[i] * means[i];
    // devs.push_back(sqrt(ASigmaSq[i]));
    devs[i] = sqrt(ASigmaSq[i]);
  }

  delete[] ACumSum;
  delete[] ASqCumSum;
  delete[] ASum;
  delete[] ASumSq;
  delete[] ASigmaSq;
}

void scrimp(DTYPE *tSeries, vector<ITYPE> &idx,
            DTYPE *means, DTYPE *devs, DTYPE *profile,
            ITYPE *profileIndex)
{

#pragma omp parallel //proc_bind(spread)
  {
    ITYPE my_offset = omp_get_thread_num() * (profileLength);
    svbool_t vlOuter = svptrue_b64(), vlInner = svptrue_b64(), vlRed = svptrue_b64();
    ITYPE Ndiags = (ITYPE)idx.size()*percent_diags/100;

// Go through diagonals (dynamic)
#pragma omp for schedule(dynamic)
    for (ITYPE ri = 0; ri < Ndiags; ri++)
    //for (ITYPE diag = exclusionZone + 1; diag < profileLength; diag += VLMAX)
    {
      ITYPE diag = idx[ri];
      ITYPE tam_vl = min((ITYPE)VLMAX, profileLength - diag);
      vlOuter = svwhilelt_b64((ITYPE)0,tam_vl);
      VDTYPE dotProd_v = SETZERO_PD();


      for (ITYPE j = diag; j < windowSize + diag; j++)
      {
        VDTYPE tSeriesj_v = LOADU_PD(&tSeries[j], vlOuter);
        VDTYPE tSeriesMinusDiag_v = SET1_PD(tSeries[j - diag], vlOuter);
        dotProd_v = FMADD_PD(tSeriesj_v, tSeriesMinusDiag_v, dotProd_v, vlOuter);
      }

      // j is the column index, i is the row index of the current distance value in the distance matrix
      ITYPE j = diag;
      ITYPE i = 0;

      // __m256d distance_v = _mm256_setzero_pd();
      VDTYPE distance_v;
      VDTYPE meansj_v = LOADU_PD(&means[j], vlOuter),
             devsj_v = LOADU_PD(&devs[j], vlOuter),
             meansi_v = SET1_PD(means[i], vlOuter),
             devsi_v = SET1_PD(devs[i], vlOuter);
      VDTYPE windowSize_v = SET1_PD((double)windowSize, vlOuter);

      // Evaluate the distance based on the dot product
      // DTYPE dotProd = 0;
      // DTYPE distance = 2 * (windowSize - (dotProd - windowSize * means[j] * means[i]) / (devs[j] * devs[i]));
      VDTYPE prod_devs_v = MUL_PD(devsi_v, devsj_v, vlOuter);
      VDTYPE triple_product_v = MUL_PD(windowSize_v, MUL_PD(meansi_v, meansj_v, vlOuter), vlOuter);
      VDTYPE division_v = DIV_PD(SUB_PD(dotProd_v, triple_product_v, vlOuter), prod_devs_v, vlOuter);
      distance_v = MUL_PD(SET1_PD(2.0, vlOuter), SUB_PD(windowSize_v, division_v, vlOuter), vlOuter);

      DTYPE distance_outer[tam_vl];
      STORE_PD(distance_outer, distance_v, vlOuter);
      for (ITYPE ii = 0; ii < tam_vl; ii++){
        if (distance_outer[ii] < profile_tmp[i + my_offset]){
          profile_tmp[i + my_offset] = distance_outer[ii];
          profileIndex_tmp[i + my_offset] = j + ii;
        }
      }
      //Búsqueda horizontal del máximo
      // Coge el primer valor del vector resultante al aplicar el mínimo entre el vector de correlación y el profile en la posición i+myoffset
      // DTYPE corr_max = REDMIN_PD(distance_v, profile_tmp[i + my_offset], vlOuter);
      // VMTYPE mask = CMP_PD_EQ(distance_v, corr_max, vlOuter);
      // long index_max = GETFIRST_MASK(mask, vlInner);

      // if(index_max != -1) {
      //   profile_tmp[i + my_offset] = corr_max;
      //   profileIndex_tmp[i + my_offset] = j + index_max;      
      // }

      VDTYPE profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlOuter);
      VMTYPE mask = CMP_PD_LT(distance_v, profilej_v, vlOuter);
      MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], distance_v);
      MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset],  SET1_EPI(i, vlOuter));

      i = 1;
      for (ITYPE j = diag + 1; j < profileLength; j++)
      {
        tam_vl = min((ITYPE)VLMAX, profileLength - j);
        vlInner = svwhilelt_b64((ITYPE)0,tam_vl);
        // RIC seguimos con que las js son diag y son las que se empaquetan, las ies se replican
        VDTYPE tSeriesj0_v = LOADU_PD(&tSeries[j + windowSize - 1], vlInner),
               tSeriesj1_v = LOADU_PD(&tSeries[j - 1], vlInner),
               tSeriesi0_v = SET1_PD(tSeries[i + windowSize - 1], vlInner),
               tSeriesi1_v = SET1_PD(tSeries[i - 1], vlInner);

        // dotProd += (tSeries[j + windowSize - 1] * tSeries[i + windowSize - 1]) - (tSeries[j - 1] * tSeries[i - 1]);
        dotProd_v = ADD_PD(FMSUB_PD(tSeriesj0_v, tSeriesi0_v,
                                    MUL_PD(tSeriesj1_v, tSeriesi1_v, vlInner), vlInner),
                           dotProd_v, vlInner);

        // distance = 2 * (windowSize - (dotProd - means[j] * means[i] * windowSize) / (devs[j] * devs[i]));
        meansj_v = LOADU_PD(&means[j], vlInner);
        devsj_v = LOADU_PD(&devs[j], vlInner);
        meansi_v = SET1_PD(means[i], vlInner);
        devsi_v = SET1_PD(devs[i], vlInner);
        triple_product_v = MUL_PD(windowSize_v, MUL_PD(meansi_v, meansj_v, vlInner), vlInner);
        prod_devs_v = MUL_PD(devsi_v, devsj_v, vlInner);
        division_v = DIV_PD(SUB_PD(dotProd_v, triple_product_v, vlInner), prod_devs_v, vlInner);
        distance_v = MUL_PD(SET1_PD(2.0, vlInner), SUB_PD(windowSize_v, division_v, vlInner), vlInner);

        DTYPE distance_inner[tam_vl];
        STORE_PD(distance_inner, distance_v, vlOuter);
        for (ITYPE jj = 0; jj < tam_vl; jj++){
          if (distance_inner[jj] < profile_tmp[i + my_offset]){
            profile_tmp[i + my_offset] = distance_inner[jj];
            profileIndex_tmp[i + my_offset] = j + jj;
          }
        }

        // Coge el primer valor del vector resultante al aplicar el máximo entre el vector de correlación y el profile en la posición i+myoffset
        // corr_max = REDMIN_PD(distance_v, profile_tmp[i + my_offset], vlInner);
        // mask = CMP_PD_EQ(distance_v, corr_max, vlInner);
        // index_max = GETFIRST_MASK(mask, vlInner);
        
        // if(index_max != -1) {
        //   profile_tmp[i + my_offset] = corr_max;
        //   profileIndex_tmp[i + my_offset] = j + index_max;      
        // }

        profilej_v = LOADU_PD(&profile_tmp[j + my_offset], vlInner);
        mask = CMP_PD_LT(distance_v, profilej_v, vlInner);
        MASKSTOREU_PD(mask, &profile_tmp[j + my_offset], distance_v);
        MASKSTOREU_EPI(mask, &profileIndex_tmp[j + my_offset], SET1_EPI(i, vlInner));

        i++;
      }

    } //'pragma omp for' places here a barrier unless 'no wait' is specified

    // DTYPE min_distance;
    // ITYPE min_index;
// Reduction
#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < profileLength; colum += VLMAX)
    {
      vlRed = svwhilelt_b64((ITYPE)0,min((ITYPE)VLMAX, profileLength - colum));
      // max_corr = -numeric_limits<DTYPE>::infinity();
      VDTYPE min_dist_v = SET1_PD(numeric_limits<DTYPE>::infinity(), vlRed);
      VITYPE min_indices_v = SET1_EPI(1, vlRed);
      for (ITYPE th = 0; th < numThreads; th++)
      {
        VDTYPE profile_tmp_v = LOADU_PD(&profile_tmp[colum + (th * profileLength)], vlRed);
        VITYPE profileIndex_tmp_v = LOADU_SI(&profileIndex_tmp[colum + (th * profileLength)], vlRed);
        VMTYPE mask = CMP_PD_LT(profile_tmp_v, min_dist_v, vlRed);
        min_indices_v = BLEND_EPI(min_indices_v, profileIndex_tmp_v, mask); // Update con máscara de los índices
        min_dist_v = BLEND_PD(min_dist_v, profile_tmp_v, mask);             // Update con máscara de las correlaciones
      }
      // Los stores sí pueden ser alineados
      STORE_PD(&profile[colum], min_dist_v, vlRed);
      STORE_SI(&profileIndex[colum], min_indices_v, vlRed);
    }
  }

  // delete[] profile_tmp;
  // delete[] profileIndex_tmp;
}

int main(int argc, char *argv[])
{
  try
  {
    // Creation of time meassure structures
    chrono::steady_clock::time_point tstart, tprogstart, tend;
    chrono::duration<double> telapsed;

    if (argc != 6)
    {
      cout << "[ERROR] usage: ./scrimp input_file win_size num_threads percent_diags out_directory" << endl;
      return 0;
    }

    windowSize = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    percent_diags = atoi(argv[4]);
    string outdir = argv[5];
    // Set the exclusion zone to 0.25
    exclusionZone = (ITYPE)(windowSize * 0.25);
    omp_set_num_threads(numThreads);

    // vector<DTYPE> tSeries;
    string inputfilename = argv[1];
    string alg = argv[0];
    alg = alg.substr(2);
    stringstream tmp;
    tmp << outdir << alg.substr(alg.rfind('/') +1) << "_" << inputfilename.substr(inputfilename.rfind('/') + 1, inputfilename.size() - 4 - inputfilename.rfind('/') - 1) << "_w" << windowSize << "_t" << numThreads << "_pdiags" << percent_diags << "_" << getpid() << ".csv";
    string outfilename = tmp.str();

    // Display info through console
    cout << endl;
    cout << "############################################################" << endl;
    cout << "///////////////////////// SCRIMP ///////////////////////////" << endl;
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
    cout << "[>>] Reading values..." << endl;
    tstart = chrono::steady_clock::now();
    tprogstart = tstart;
    // fstream tSeriesFile(inputfilename, ios_base::in);
    tSeriesFile.clear();                // Limpio el stream
    tSeriesFile.seekg(tSeriesFile.beg); // Y lo reinicio a beginning
    DTYPE *tSeries = NULL;
    DTYPE tempval, tSeriesMin = numeric_limits<DTYPE>::infinity(),
                   tSeriesMax = -numeric_limits<double>::infinity();
    // alignas(ALIGN) DTYPE tSeries[tSeriesLength + VLMAX];
    // tSeries = new (std::align_val_t(ALIGN)) DTYPE[tSeriesLength + VLMAX];
    ARRAY_NEW(DTYPE, tSeries, tSeriesLength);

    // RIC comprobar si el fichero tiene algún NaN y quitarlo
    for (int i = 0; tSeriesFile >> tempval; i++)
    {
      tSeries[i] = tempval;
      if (tempval < tSeriesMin)
        tSeriesMin = tempval;
      if (tempval > tSeriesMax)
        tSeriesMax = tempval;
    }
    tSeriesFile.close();
    // RIC inicializo los VLMAX sobrantes a NaN, así todas las operaciones que
    //  se hagan con ellos darán NaN, y no se contabilizarán en los resultados
    //  finales
    for (ITYPE i = tSeriesLength; i < tSeriesLength; i++)
      tSeries[i] = numeric_limits<DTYPE>::quiet_NaN();

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Read File Time: " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    // Set Matrix Profile Length
    profileLength = tSeriesLength - windowSize + 1;
    DTYPE *means = NULL, *devs = NULL, *profile = NULL;
    vector<ITYPE> diags;
    ITYPE *profileIndex = NULL;
    ARRAY_NEW(DTYPE, means, profileLength);
    ARRAY_NEW(DTYPE, devs, profileLength);
    ARRAY_NEW(DTYPE, profile, profileLength);
    ARRAY_NEW(ITYPE, profileIndex, profileLength);

    // RIC sumo VLMAX al profileLength
    ARRAY_NEW(DTYPE, profile_tmp, profileLength * numThreads);
    ARRAY_NEW(ITYPE, profileIndex_tmp, profileLength * numThreads);
    // Private profile initialization
    // RIC Pongo + VLMAX para inicializar a infinito los elementos añadidos
    for (ITYPE i = 0; i < (profileLength * numThreads); i++)
      profile_tmp[i] = numeric_limits<DTYPE>::infinity();

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


    cout << "[>>] Preprocessing..." << endl;
    tstart = chrono::steady_clock::now();

    preprocess(tSeries, means, devs);

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Preprocessing Time:         " << setprecision(numeric_limits<double>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    /***********************************************/
    
    // Random shuffle the diagonals
    diags.clear();
    for (ITYPE i = exclusionZone + 1; i < profileLength; i+= VLMAX)
      diags.push_back(i);

    if (SHUFFLE_DIAGS){
      //random_device rd;
      mt19937 g(0);
      shuffle(diags.begin(), diags.end(), g);
    }
    /******************** SCRIMP ********************/
    cout << "[>>] Executing SCRIMP..." << endl;
    tstart = chrono::steady_clock::now();

    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_begin();
    #endif


    // Establish begining of ROI
    #ifdef ENABLE_GEM5_ROI
    m5_checkpoint(0,0);
    #endif

    scrimp(tSeries, diags, means, devs, profile, profileIndex);

    // Establish end of ROI    
    #ifdef ENABLE_GEM5_ROI
    m5_checkpoint(0,0);
    #endif
    
    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_end();
    #endif

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] SCRIMP Time:             " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    // Save profile to file
    cout << "[>>] Saving Profile..." << endl;
    tstart = chrono::steady_clock::now();

    fstream statsFile(outfilename, ios_base::out);
    statsFile << "# Time (s)" << endl;
    statsFile << setprecision(9) << telapsed.count() << endl;
    statsFile << "# Profile Length" << endl;
    statsFile << profileLength << endl;
    statsFile << "# i,tseries,profile,index" << endl;
    for (ITYPE i = 0; i < profileLength; i++)
    {
      statsFile << i << "," << tSeries[i] << "," << (DTYPE)sqrt(profile[i]) << "," << profileIndex[i] << endl;
    }
    statsFile.close();

    ARRAY_DEL(tSeries);
    ARRAY_DEL(means);
    ARRAY_DEL(devs);
    ARRAY_DEL(profile);
    ARRAY_DEL(profileIndex);
    ARRAY_DEL(profile_tmp);
    ARRAY_DEL(profileIndex_tmp);

    tend = chrono::steady_clock::now();
    telapsed = tend - tstart;
    cout << "[OK] Saving Profile Time:       " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;

    // Calculate total time
    telapsed = tend - tprogstart;
    cout << "[OK] Total Time:              " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << telapsed.count() << " seconds." << endl;
    cout << "[OK] Filename: " << outfilename << endl;
    cout << endl;
  }
  catch (exception &e)
  {
    cout << "Exception: " << e.what() << endl;
  }
}

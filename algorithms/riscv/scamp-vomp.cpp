#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <assert.h>
#include <omp.h>
#include <unistd.h> //For getpid(), used to get the pid to generate a unique filename
#include <typeinfo> //To obtain type name as string
#include <pthread.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif 

#ifdef ENABLE_GEM5_ROI
#include <gem5/m5ops.h>
#endif


#define DTYPE double        /* DATA TYPE */
#define ITYPE long long int /* INDEX TYPE */
#define K 8

using namespace std;

int numThreads, exclusionZone;
ITYPE windowSize, timeSeriesLength, ProfileLength;
ITYPE *profileIndex;
DTYPE *profile;
DTYPE *norms, *means, *df, *dg, *cov;
vector<DTYPE> A;

// Private structures
DTYPE *profile_tmp;
ITYPE *profileIndex_tmp;


// Computes all required statistics for SCAMP, populating info with these values
void preprocess()
{
  vector<DTYPE> prefix_sum(A.size());
  vector<DTYPE> prefix_sum_sq(A.size());

  prefix_sum[0] = A[0];
  prefix_sum_sq[0] = A[0] * A[0];
  for (ITYPE i = 1; i < timeSeriesLength; ++i)
  {
    prefix_sum[i] = A[i] + prefix_sum[i - 1];
    prefix_sum_sq[i] = A[i] * A[i] + prefix_sum_sq[i - 1];
  }

  means[0] = prefix_sum[windowSize - 1] / static_cast<double>(windowSize);
  for (ITYPE i = 1; i < ProfileLength; ++i)
  {
    means[i] =
        (prefix_sum[i + windowSize - 1] - prefix_sum[i - 1]) / static_cast<DTYPE>(windowSize);
  }

  DTYPE sum = 0;
  for (ITYPE i = 0; i < windowSize; ++i)
  {
    DTYPE val = A[i] - means[0];
    sum += val * val;
  }
  norms[0] = sum;

  for (ITYPE i = 1; i < ProfileLength; ++i)
  {
    norms[i] =
        norms[i - 1] + ((A[i - 1] - means[i - 1]) + (A[i + windowSize - 1] - means[i])) *
                           (A[i + windowSize - 1] - A[i - 1]);
  }
  for (ITYPE i = 0; i < ProfileLength; ++i)
  {
    norms[i] = 1.0 / sqrt(norms[i]);
  }

  for (ITYPE i = 0; i < ProfileLength - 1; ++i)
  {
    df[i] = (A[i + windowSize] - A[i]) / 2.0;
    dg[i] = (A[i + windowSize] - means[i + 1]) + (A[i] - means[i]);
  }
}

void scamp()
{

#pragma omp parallel //proc_bind(spread)
  {
    ITYPE my_offset = omp_get_thread_num() * ProfileLength;

    DTYPE *covs, *corrs, lastcov, lastcorr;

    covs = new DTYPE[K];
    corrs = new DTYPE[K];

#pragma omp for schedule(dynamic)
    // Go through diagonals
    for (ITYPE diag = exclusionZone + 1; diag < ProfileLength; diag++)
    {
      // for(int i = 0; i < K; i++)
      lastcov = 0;

#pragma omp simd
      for (ITYPE i = 0; i < windowSize; i++)
      {
        lastcov += ((A[diag + i] - means[diag]) * (A[i] - means[0]));
      }

      ITYPE i = 0;
      ITYPE j = diag;

      lastcorr = lastcov * norms[i] * norms[j];

      if (lastcorr > profile_tmp[i + my_offset])
      {
        profile_tmp[i + my_offset] = lastcorr;
        profileIndex_tmp[i + my_offset] = j;
      }
      if (lastcorr > profile_tmp[j + my_offset])
      {
        profile_tmp[j + my_offset] = lastcorr;
        profileIndex_tmp[j + my_offset] = i;
      }

      i = 1;

      for (ITYPE j = diag + 1; j < ProfileLength; j += K)
      {
#pragma omp simd
        for (int k = 0; k < K; k++)
          covs[k] = df[k + i - 1] * dg[k + j - 1] + df[k + j - 1] * dg[k + i - 1];

        covs[0] += lastcov;
        for (int k = 1; k < K; k++)
        {
          covs[k] += covs[k - 1];
        }

        lastcov = covs[K - 1];

#pragma omp simd
        for (int k = 0; k < K; k++)
          corrs[k] = covs[k] * norms[k + i] * norms[k + j];

#pragma omp simd
        for (int k = 0; k < K; k++)
        {

          if (corrs[k] > profile_tmp[k + i + my_offset])
          {
            profile_tmp[k + i + my_offset] = corrs[k];
            profileIndex_tmp[k + i + my_offset] = k + j;
          }

          if (corrs[k] > profile_tmp[k + j + my_offset])
          {
            profile_tmp[k + j + my_offset] = corrs[k];
            profileIndex_tmp[k + j + my_offset] = k + i;
          }
        }
        i += K;
      }
    }

    DTYPE max_distance;
    ITYPE max_index = 0;

#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < ProfileLength; colum++)
    {
      max_distance = -numeric_limits<DTYPE>::infinity();
      ;
      for (ITYPE row = 0; row < numThreads; row++)
      {
        if (profile_tmp[colum + (row * ProfileLength)] > max_distance)
        {
          max_distance = profile_tmp[colum + (row * ProfileLength)];
          max_index = profileIndex_tmp[colum + (row * ProfileLength)];
        }
      }
      profile[colum] = max_distance;
      profileIndex[colum] = max_index;
    }
  }
}

int main(int argc, char *argv[])
{
  try
  {
    // Creation of time meassure structures
    chrono::steady_clock::time_point tprogstart, tstart, tend;
    chrono::duration<double> time_elapsed;

    if (argc != 5)
    {
      cout << "[ERROR] usage: ./scamp input_file win_size num_threads out_directory" << endl;
      return 0;
    }

    windowSize = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    string outdir = argv[4];
    // Set the exclusion zone to 0.25
    exclusionZone = (ITYPE)(windowSize * 0.25);
    omp_set_num_threads(numThreads);

    string inputfilename = argv[1];
    string alg = argv[0];
    alg = alg.substr(2);
    stringstream tmp;
    tmp << outdir << alg.substr(alg.rfind('/') +1) << "_" << inputfilename.substr(inputfilename.rfind('/') + 1, inputfilename.size() - 4 - inputfilename.rfind('/') - 1) << "_w" << windowSize << "_t" << numThreads << "_" << getpid() << ".csv";
    string outfilename = tmp.str();

    // Display info through console
    cout << endl;
    cout << "############################################################" << endl;
    cout << "///////////////////////// SCAMP ////////////////////////////" << endl;
    cout << "############################################################" << endl;
    cout << endl;
    cout << "[>>] Reading File: " << inputfilename << "..." << endl;

    /* ------------------------------------------------------------------ */
    /* Read time series file */
    tstart = chrono::steady_clock::now();

    fstream timeSeriesFile(inputfilename, ios_base::in);

    DTYPE tempval, tSeriesMin = numeric_limits<DTYPE>::infinity(), tSeriesMax = -numeric_limits<double>::infinity();

    timeSeriesLength = 0;
    while (timeSeriesFile >> tempval)
    {
      A.push_back(tempval);

      if (tempval < tSeriesMin)
        tSeriesMin = tempval;
      if (tempval > tSeriesMax)
        tSeriesMax = tempval;
      timeSeriesLength++;
    }
    timeSeriesFile.close();

    tend = chrono::steady_clock::now();
    time_elapsed = tend - tstart;
    cout << "[OK] Read File Time: " << setprecision(numeric_limits<double>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;

    // Set Matrix Profile Length
    ProfileLength = timeSeriesLength - windowSize + 1;

    profile = new DTYPE[ProfileLength];
    profileIndex = new ITYPE[ProfileLength];

    profile_tmp = new DTYPE[ProfileLength * numThreads];
    profileIndex_tmp = new ITYPE[ProfileLength * numThreads];
    // Private profile initialization
    for (ITYPE i = 0; i < (ProfileLength * numThreads); i++)
      profile_tmp[i] = -numeric_limits<DTYPE>::infinity();

    // Display info through console
    cout << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "************************** INFO ****************************" << endl;
    cout << endl;
    cout << " Series/MP data type: " << typeid(A[0]).name() << "(" << sizeof(A[0]) << "B)" << endl;
    cout << " Index data type:     " << typeid(profileIndex[0]).name() << "(" << sizeof(profileIndex[0]) << "B)" << endl;
    cout << " Time series length:  " << timeSeriesLength << endl;
    cout << " Window size:         " << windowSize << endl;
    cout << " Time series min:     " << tSeriesMin << endl;
    cout << " Time series max:     " << tSeriesMax << endl;
    cout << " Number of threads:   " << numThreads << endl;
    cout << " Exclusion zone:      " << exclusionZone << endl;
    cout << " Profile length:      " << ProfileLength << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << endl;

    // Auxiliary vectors
    norms = new DTYPE[timeSeriesLength];
    means = new DTYPE[timeSeriesLength];
    df = new DTYPE[timeSeriesLength];
    dg = new DTYPE[timeSeriesLength];
    cov = new DTYPE[timeSeriesLength];

    preprocess();

    tend = chrono::steady_clock::now();
    time_elapsed = tend - tstart;
    cout << "[OK] Preprocess Time:         " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;

    /******************** SCAMP ********************/
    cout << "[>>] Performing SCAMP..." << endl;
    tstart = chrono::steady_clock::now();

    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_begin();
    #endif


    // Establish begining of ROI
    #ifdef ENABLE_GEM5_ROI
    m5_work_begin(0,0);
    #endif

    scamp();


    // Establish end of ROI    
    #ifdef ENABLE_GEM5_ROI
    m5_work_end(0,0);
    #endif
    
    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_end();
    #endif

    tend = chrono::steady_clock::now();
    time_elapsed = tend - tstart;
    cout << "[OK] SCAMP Time:              " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;

    cout << "[>>] Saving result: " << outfilename << " ..." << endl;
    fstream statsFile(outfilename, ios_base::out);
    statsFile << "# Time (s)" << endl;
    statsFile << setprecision(9) << time_elapsed.count() << endl;
    statsFile << "# Profile Length" << endl;
    statsFile << ProfileLength << endl;
    statsFile << "# i,tseries,profile,index" << endl;
    for (ITYPE i = 0; i < ProfileLength; i++)
    {
      statsFile << i << "," << A[i] << "," << (DTYPE)sqrt(2 * windowSize * (1 - profile[i])) << "," << profileIndex[i] << endl;
    }
    statsFile.close();

    delete[] norms;
    delete[] means;
    delete[] df;
    delete[] dg;
    delete[] cov;
    delete[] profile_tmp;
    delete[] profileIndex_tmp;
    cout << endl;
  }
  catch (exception &e)
  {
    cout << "Exception: " << e.what() << endl;
  }
}

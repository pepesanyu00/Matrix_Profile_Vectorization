#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <unistd.h> //For getpid(), used to get the pid to generate a unique filename
#include <typeinfo> //To obtain type name as string
#include <array>
#include <assert.h> //RIC incluyo assert para hacer comprobaciones de invariantes y condiciones


#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif 

#ifdef ENABLE_GEM5_ROI
#include <gem5/m5ops.h>
#endif



#define DTYPE double        /* DATA TYPE */
#define ITYPE long long int /* INDEX TYPE: RIC pongo long long int para que tanto el double como el int sean de 64 bits (facilita la vectorización) */

#define ARIT_FACT 8

using namespace std;

int numThreads, exclusionZone;
int windowSize, timeSeriesLength, ProfileLength;
ITYPE *profileIndex;
DTYPE *AMean, *ASigma, *profile;
vector<ITYPE> idx;
vector<DTYPE> A;

// Private structures
DTYPE *profile_tmp;
ITYPE *profileIndex_tmp;


void preprocess()
{
  DTYPE *ACumSum = new DTYPE[timeSeriesLength];
  DTYPE *ASqCumSum = new DTYPE[timeSeriesLength];
  DTYPE *ASum = new DTYPE[ProfileLength];
  DTYPE *ASumSq = new DTYPE[ProfileLength];
  DTYPE *ASigmaSq = new DTYPE[ProfileLength];

  AMean = new DTYPE[ProfileLength];
  ASigma = new DTYPE[ProfileLength];

  ACumSum[0] = A[0];
  ASqCumSum[0] = A[0] * A[0];

  for (ITYPE i = 1; i < timeSeriesLength; i++)
  {
    ACumSum[i] = A[i] + ACumSum[i - 1];
    ASqCumSum[i] = A[i] * A[i] + ASqCumSum[i - 1];
  }

  ASum[0] = ACumSum[windowSize - 1];
  ASumSq[0] = ASqCumSum[windowSize - 1];

  for (ITYPE i = 0; i < timeSeriesLength - windowSize; i++)
  {
    ASum[i + 1] = ACumSum[windowSize + i] - ACumSum[i];
    ASumSq[i + 1] = ASqCumSum[windowSize + i] - ASqCumSum[i];
  }

  for (ITYPE i = 0; i < ProfileLength; i++)
  {
    AMean[i] = ASum[i] / windowSize;
    ASigmaSq[i] = ASumSq[i] / windowSize - AMean[i] * AMean[i];
    ASigma[i] = sqrt(ASigmaSq[i]);
  }

  delete[] ACumSum;
  delete[] ASqCumSum;
  delete[] ASum;
  delete[] ASumSq;
  delete[] ASigmaSq;
}

void scrimp()
{

#pragma omp parallel //proc_bind(spread)
  {
    DTYPE lastz, distance, windowSizeDouble;
    DTYPE *distances, *lastzs;
    ITYPE diag, my_offset, i, j;
    long unsigned int ri;

    distances = new DTYPE[ARIT_FACT];
    lastzs = new DTYPE[ARIT_FACT];

    windowSizeDouble = (DTYPE)windowSize;

    my_offset = omp_get_thread_num() * ProfileLength;

#pragma omp for schedule(dynamic)
    for (ri = 0; ri < idx.size(); ri++)
    {
      // select a diagonal
      diag = idx[ri];

      lastz = 0;

// calculate the dot product of every two time series values that ar diag away
#pragma omp simd
      for (j = diag; j < windowSize + diag; j++)
      {
        lastz += A[j] * A[j - diag];
      }

      // j is the column index, i is the row index of the current distance value in the distance matrix
      j = diag;
      i = 0;

      // evaluate the distance based on the dot product
      distance = 2 * (windowSizeDouble - (lastz - windowSizeDouble * AMean[j] * AMean[i]) / (ASigma[j] * ASigma[i]));

      // update matrix profile and matrix profile index if the current distance value is smaller
      if (distance < profile_tmp[my_offset + j])
      {
        profile_tmp[my_offset + j] = distance;
        profileIndex_tmp[my_offset + j] = i;
      }

      if (distance < profile_tmp[my_offset + i])
      {
        profile_tmp[my_offset + i] = distance;
        profileIndex_tmp[my_offset + i] = j;
      }
      i = 1;
      j = diag + 1;

      while (j < (ProfileLength - ARIT_FACT))
      {
#pragma omp simd
        for (int k = 0; k < ARIT_FACT; k++)
        {
          lastzs[k] = (A[k + j + windowSize - 1] * A[k + i + windowSize - 1]) - (A[k + j - 1] * A[k + i - 1]);
        }

        lastzs[0] += lastz;
        //#pragma unroll(ARIT_FACT - 1)
        for (int k = 1; k < ARIT_FACT; k++)
        {
          lastzs[k] += lastzs[k - 1];
        }
        lastz = lastzs[ARIT_FACT - 1];

#pragma omp simd
        for (int k = 0; k < ARIT_FACT; k++)
        {
          distances[k] = 2 * (windowSizeDouble - (lastzs[k] - AMean[k + j] * AMean[k + i] * windowSizeDouble) / (ASigma[k + j] * ASigma[k + i]));
        }

#pragma omp simd
        for (int k = 0; k < ARIT_FACT; k++)
        {
          if (distances[k] < profile_tmp[k + my_offset + j])
          {
            profile_tmp[k + my_offset + j] = distances[k];
            profileIndex_tmp[k + my_offset + j] = i + k;
          }

          if (distances[k] < profile_tmp[k + my_offset + i])
          {
            profile_tmp[k + my_offset + i] = distances[k];
            profileIndex_tmp[k + my_offset + i] = j + k;
          }
        }
        i += ARIT_FACT;
        j += ARIT_FACT;
      }

      while (j < ProfileLength)
      {
        lastz = lastz + (A[j + windowSize - 1] * A[i + windowSize - 1]) - (A[j - 1] * A[i - 1]);
        distance = 2 * (windowSizeDouble - (lastz - AMean[j] * AMean[i] * windowSizeDouble) / (ASigma[j] * ASigma[i]));

        if (distance < profile_tmp[my_offset + j])
        {
          profile_tmp[my_offset + j] = distance;
          profileIndex_tmp[my_offset + j] = i;
        }

        if (distance < profile_tmp[my_offset + i])
        {
          profile_tmp[my_offset + i] = distance;
          profileIndex_tmp[my_offset + i] = j;
        }
        i++;
        j++;
      }
    }


    // Reduce the (partial) result
    DTYPE min_distance;
    ITYPE min_index;

#pragma omp for schedule(static)
    for (ITYPE colum = 0; colum < ProfileLength; colum++)
    {
      min_distance = std::numeric_limits<DTYPE>::infinity();
      min_index = 0;
      //      #pragma unroll(256)
      for (ITYPE row = 0; row < numThreads; row++)
      {
        if (profile_tmp[colum + (row * ProfileLength)] < min_distance)
        {
          min_distance = profile_tmp[colum + (row * ProfileLength)];
          min_index = profileIndex_tmp[colum + (row * ProfileLength)];
        }
      }
      profile[colum] = min_distance;
      profileIndex[colum] = min_index;
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
    // Defaults: random computational order and hbm allocation
    // bool sequentialDiags = false;

    if (argc != 5)
    {
      cout << "[ERROR] usage: ./scrimp input_file win_size num_threads out_directory" << endl;
      return 0;
    }

    windowSize = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    string outdir = argv[4];
    // Set the exclusion zone
    exclusionZone = (ITYPE)(windowSize * 0.25);
    omp_set_num_threads(numThreads);

    // Set computational order
    /*if(argc > 4)
          sequentialDiags = (strcmp(argv[4], "-s") == 0);*/

    string inputfilename = argv[1];
    string alg = argv[0];
    alg = alg.substr(2);
    stringstream tmp;
    tmp << outdir << alg.substr(alg.rfind('/') +1) << "_" << inputfilename.substr(inputfilename.rfind('/') + 1, inputfilename.size() - 4 - inputfilename.rfind('/') - 1) << "_w" << windowSize << "_t" << numThreads << "_" << getpid() << ".csv";
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
    timeSeriesLength = 0;
    cout << "[>>] Counting lines ... " << endl;
    string line;
    while (getline(tSeriesFile, line)) // Cuento el número de líneas
      timeSeriesLength++;

    tend = chrono::steady_clock::now();
    time_elapsed = tend - tstart;
    cout << "[OK] Lines: " << timeSeriesLength << " Time: " << time_elapsed.count() << "s." << endl;
    /* ------------------------------------------------------------------ */
    /* Read time series file */
    cout << "[>>] Reading values..." << endl;
    tstart = chrono::steady_clock::now();
    tprogstart = tstart;
    // fstream tSeriesFile(inputfilename, ios_base::in);
    tSeriesFile.clear();                // Limpio el stream
    tSeriesFile.seekg(tSeriesFile.beg); // Y lo reinicio a beginning

    DTYPE tempval, tSeriesMin = numeric_limits<DTYPE>::infinity(),
                   tSeriesMax = -numeric_limits<double>::infinity();

    // RIC comprobar si el fichero tiene algún NaN y quitarlo
    for (ITYPE i = 0; tSeriesFile >> tempval; i++)
    {
      A.push_back(tempval);
      if (tempval < tSeriesMin)
        tSeriesMin = tempval;
      if (tempval > tSeriesMax)
        tSeriesMax = tempval;
    }
    tSeriesFile.close();

    tend = chrono::steady_clock::now();
    time_elapsed = tend - tstart;
    cout << "[OK] Read File Time: " << setprecision(numeric_limits<double>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;

    // Set Matrix Profile Length
    ProfileLength = timeSeriesLength - windowSize + 1;
    // Display info through console
    cout << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "************************** INFO ****************************" << endl;
    cout << endl;
    cout << " Series/MP data type: " << typeid(A[0]).name() << "(" << sizeof(A[0]) << "B)" << endl;
    cout << " Index data type:     " << typeid(profileIndex[0]).name() << "(" << sizeof(profileIndex[0]) << "B)" << endl;
    cout << " Time series length: " << timeSeriesLength << endl;
    cout << " Window size:        " << windowSize << endl;
    cout << " Time series min:     " << tSeriesMin << endl;
    cout << " Time series max:     " << tSeriesMax << endl;
    cout << " Number of threads:  " << numThreads << endl;
    cout << " Exclusion zone:     " << exclusionZone << endl;
    cout << " Profile length:     " << ProfileLength << endl;
    /* cout << " Sequential order:   ";
    if (sequentialDiags)
      cout << "true" << endl;
    else
      cout << "false" << endl;
    cout << endl; */
    cout << "------------------------------------------------------------" << endl;
    cout << endl;

    // Preprocess, statistics, get the mean and standard deviation of every subsequence in the time series
    cout << "[>>] Preprocessing..." << endl;
    tstart = chrono::steady_clock::now();

    preprocess();

    tend = std::chrono::steady_clock::now();
    time_elapsed = tend - tstart;
    cout << "[OK] Preprocess Time:         " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;

    // Initialize Matrix Profile and Matrix Profile Index
    cout << "[>>] Initializing Profile..." << endl;
    tstart = chrono::steady_clock::now();

    profile = new DTYPE[ProfileLength];
    profileIndex = new ITYPE[ProfileLength];

    profile_tmp = new DTYPE[ProfileLength * numThreads];
    profileIndex_tmp = new ITYPE[ProfileLength * numThreads];

    // Private profile initialization
    for (ITYPE i = 0; i < (ProfileLength * numThreads); i++)
      profile_tmp[i] = numeric_limits<DTYPE>::infinity();

    tend = chrono::steady_clock::now();
    time_elapsed = tend - tstart;
    cout << "[OK] Initialize Profile Time: " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;

    // Random shuffle the diagonals
    idx.clear();
    for (int i = exclusionZone + 1; i < ProfileLength; i++)
      idx.push_back(i);

    // if (!sequentialDiags)
    //   std::random_shuffle(idx.begin(), idx.end());

    /******************** SCRIMP ********************/
    cout << "[>>] Performing SCRIMP..." << endl;
    tstart = chrono::steady_clock::now();

    // ROI de Iván
    #ifdef ENABLE_PARSEC_HOOKS
      __parsec_roi_begin();
    #endif


    // Establish begining of ROI
    #ifdef ENABLE_GEM5_ROI
    m5_work_begin(0,0);
    #endif

    scrimp();

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
    cout << "[OK] SCRIMP Time:             " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;

    // Save profile to file
    cout << "[>>] Saving Profile..." << endl;
    tstart = chrono::steady_clock::now();

    fstream statsFile(outfilename, ios_base::out);
    statsFile << "# Time (s)" << endl;
    statsFile << setprecision(9) << time_elapsed.count() << endl;
    statsFile << "# Profile Length" << endl;
    statsFile << ProfileLength << endl;
    statsFile << "# i,tseries,profile,index" << endl;
    for (ITYPE i = 0; i < ProfileLength; i++)
    {
      statsFile << i << "," << A[i] << "," << (DTYPE)sqrt(profile[i]) << "," << profileIndex[i] << endl;
    }
    statsFile.close();
    delete[] profile;
    delete[] profileIndex;
    delete[] profile_tmp;
    delete[] profileIndex_tmp;

    tend = chrono::steady_clock::now();
    time_elapsed = tend - tstart;
    cout << "[OK] Save Profile Time:       " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;

    // Calculate total time
    time_elapsed = tend - tprogstart;
    cout << "[OK] Total Time:              " << setprecision(numeric_limits<DTYPE>::digits10 + 2) << time_elapsed.count() << " seconds." << endl;
    cout << endl;
  }
  catch (exception &e)
  {
    cout << "Exception: " << e.what() << endl;
  }
}

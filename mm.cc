#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <pmmintrin.h> // for SSE3
#include "timer.c"

#define N_ 4096
#define K_ 4096
#define M_ 4096

typedef double dtype;

void print_mat(dtype *A, int N, int M)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j){
      printf("%lf, ", A[i * M  + j]);
    }
    printf("\n");
  }
}

void verify(dtype *C, dtype *C_ans, int N, int M)
{
  int i, cnt;
  cnt = 0;
  for(i = 0; i < N * M; i++) {
    if(abs (C[i] - C_ans[i]) > 1e-6) cnt++;
  }
  if(cnt != 0) printf("ERROR\n"); else printf("SUCCESS\n");
}

void mm_serial (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  int i, j, k;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      for(int k = 0; k < K; k++) {
        C[i * M + j] += A[i * K + k] * B[k * M + j];
      }
    }
  }
}
void block_multiply(dtype *C, dtype *A, dtype *B,
  int blocksize)
{
  for(int i = 0; i < blocksize; ++i){
    for(int j = 0; j < blocksize; ++j){
      for(int k = 0; k < blocksize; ++k){
        C[i * blocksize + j] += A[i * blocksize + k] * B[k * blocksize + j];
      }
    }
  }
}

void mm_cb (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  /* =======================================================+ */
  /* Implement your own cache-blocked matrix-matrix multiply  */
  /* =======================================================+ */
  int _BLOCKSIZE_ = 64;       //SQRT of block size

  if (N < _BLOCKSIZE_ || M < _BLOCKSIZE_ || K < _BLOCKSIZE_) _BLOCKSIZE_ = 1;
  int i, j, k;
  int N_blocks = (int) N / _BLOCKSIZE_;
  int M_blocks = (int) M / _BLOCKSIZE_;
  int K_blocks = (int) K / _BLOCKSIZE_;

  dtype *A_block = (dtype*) malloc (_BLOCKSIZE_ * _BLOCKSIZE_ * sizeof (dtype));
  dtype *B_block = (dtype*) malloc (_BLOCKSIZE_ * _BLOCKSIZE_ * sizeof (dtype));
  dtype *C_block = (dtype*) malloc (_BLOCKSIZE_ * _BLOCKSIZE_ * sizeof (dtype));

  for(int i = 0; i < N_blocks; i++) {
    for(int j = 0; j < M_blocks; j++) {
      //READ C(i,j)
      for (int read_i = 0; read_i < _BLOCKSIZE_; ++read_i) {
        for (int read_j = 0; read_j < _BLOCKSIZE_; ++read_j) {
          C_block[read_i * _BLOCKSIZE_ + read_j] = C[(i * _BLOCKSIZE_ + read_i) * M  + (j * _BLOCKSIZE_ + read_j)];
        }
      }

      for(int k = 0; k < K_blocks; k++) {
        //Read A,B
        for (int read_i = 0; read_i < _BLOCKSIZE_; ++read_i) {
          for (int read_j = 0; read_j < _BLOCKSIZE_; ++read_j) {
            A_block[read_i * _BLOCKSIZE_ + read_j] = A[(i * _BLOCKSIZE_ + read_i) * K  + (k * _BLOCKSIZE_ + read_j)];
            B_block[read_i * _BLOCKSIZE_ + read_j] = B[(k * _BLOCKSIZE_ + read_i) * M  + (j * _BLOCKSIZE_ + read_j)];
          }
        }
        block_multiply(C_block, A_block, B_block,_BLOCKSIZE_);
      }
      //Write C block back
      for (int read_i = 0; read_i < _BLOCKSIZE_; ++read_i) {
        for (int read_j = 0; read_j < _BLOCKSIZE_; ++read_j) {
          C[(i * _BLOCKSIZE_ + read_i) * M  + (j * _BLOCKSIZE_ + read_j)] = C_block[read_i * _BLOCKSIZE_ + read_j];
        }
      }
    }
  }
}

// gcc -O3 -march=native mm.cc -o mm
void mm_sv (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
	/* =======================================================+ */
	/* Implement your own SIMD-vectorized matrix-matrix multiply  */
	/* =======================================================+ */
	// C++ vectors are stored in row-major order, so need to transpose second matrix for SIMD
	// using K, M values based on homework image
	dtype *Btemparray = (dtype*) malloc (K * M * sizeof (dtype));
	for (int i = 0; i < K; i++) {
		for (int j = i; j < M; j++) {
			Btemparray[i * K + j] = B[j * M + i];
			Btemparray[j * M + i] = B[i * K + j];
		}
	}

	// 3 128 bit registers used for multiplication and summation
	for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
		__m128d Z = _mm_setzero_pd();
          for(int k = 0; k < K; k += 2) {
					__m128d X, Y;
    				// load 2 numbers in each register
    				X = _mm_load_pd(&A[i * K + k]);
    				Y = _mm_load_pd(&Btemparray[j * K + k]);
    				// multiply the numbers and add together
    				X = _mm_mul_pd(X, Y);
    				Z = _mm_add_pd(X, Z);
  			}
		// sum all values in Z
		Z = _mm_hadd_pd(Z, Z);
		// store in C array
		_mm_store_sd(&C[i * N + j], Z);
		}
	}

	free(Btemparray);
}

int main(int argc, char** argv)
{
  int i, j, k;
  int N, K, M;

  if(argc == 4) {
    N = atoi (argv[1]);
    K = atoi (argv[2]);
    M = atoi (argv[3]);
    printf("N: %d K: %d M: %d\n", N, K, M);
  } else {
    N = N_;
    K = K_;
    M = M_;
    printf("N: %d K: %d M: %d\n", N, K, M);
  }

  dtype *A = (dtype*) malloc (N * K * sizeof (dtype));
  dtype *B = (dtype*) malloc (K * M * sizeof (dtype));
  dtype *C = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_cb = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_sv = (dtype*) malloc (N * M * sizeof (dtype));
  assert (A && B && C);

  /* initialize A, B, C */
  srand48 (time (NULL));
  for(i = 0; i < N; i++) {
    for(j = 0; j < K; j++) {
      A[i * K + j] = drand48 ();
    }
  }
  for(i = 0; i < K; i++) {
    for(j = 0; j < M; j++) {
      B[i * M + j] = drand48 ();
    }
  }
  bzero(C, N * M * sizeof (dtype));
  bzero(C_cb, N * M * sizeof (dtype));
  bzero(C_sv, N * M * sizeof (dtype));

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create ();
  assert (timer);
  long double t;

  printf("Naive matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_serial (C, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for naive implementation: %Lg seconds\n\n", t);


  printf("Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_cb (C_cb, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for cache-blocked implementation: %Lg seconds\n", t);

  /* verify answer */
  verify (C_cb, C, N, M);

  printf("SIMD-vectorized Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_sv (C_sv, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for SIMD-vectorized cache-blocked implementation: %Lg seconds\n", t);

  /* verify answer */
  verify (C_sv, C, N, M);

  return 0;
}

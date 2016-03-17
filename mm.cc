#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

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
  printf("A:\n");
  print_mat(A, N, K);
  printf("B:\n");
  print_mat(B, K, M);
  int i, j, k;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      for(int k = 0; k < K; k++) {
        C[i * M + j] += A[i * K + k] * B[k * M + j];
      }
    }
  }
  printf("C:\n");
  print_mat(C, N, M);
}
void block_multiply(dtype *C, dtype *A, dtype *B,
  int blocksize ,int in_i, int in_j, int in_k, int N, int K, int M)
{
  int i = in_i * blocksize;
  int j = in_j * blocksize;
  int k = in_k * blocksize;
  for(i; i < in_i + blocksize; ++i){
    for(j; j < in_j + blocksize; ++j){
      for(k; k < in_k + blocksize; ++k){
        C[i * M + j] += A[i * K + k] * B[k * M + j];
      }
    }
  }
}

void mm_cb (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  /* =======================================================+ */
  /* Implement your own cache-blocked matrix-matrix multiply  */
  /* =======================================================+ */
  int _BLOCKSIZE_ = 2;
  int i, j, k;
  int N_blocks = N / _BLOCKSIZE_;
  int M_blocks = M / _BLOCKSIZE_;
  int K_blocks = K / _BLOCKSIZE_;
  printf("A:\n");
  print_mat(A, N, K);
  printf("B:\n");
  print_mat(B, K, M);
  for(int i = 0; i < N_blocks; i++) {
    for(int j = 0; j < M_blocks; j++) {
      //READ C(i,j)
      for(int k = 0; k < K_blocks; k++) {
        //READ A(i,k) B(k,j)
        block_multiply(C, A, B,_BLOCKSIZE_, i, j, k, N, K, M);
      }
    }
  }
  printf("C:\n");
  print_mat(C, N, M);
}

void mm_sv (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  /* =======================================================+ */
  /* Implement your own SIMD-vectorized matrix-matrix multiply  */
  /* =======================================================+ */
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

/* The Chunks and Tasks Matrix Library (CHTML), version 2.1. A sparse
 * matrix library based on the Chunks and Tasks programming model.
 * 
 * For copyright and license information, see below under "Copyright and
 * license".
 * 
 * Primary academic reference:
 * Locality-aware parallel block-sparse matrix-matrix multiplication
 * using the Chunks and Tasks programming model,
 * Emanuel H. Rubensson and Elias Rudberg,
 * Parallel Comput. 57, 87 (2016),
 * <http://dx.doi.org/10.1016/j.parco.2016.06.005>
 * 
 * For further information about Chunks and Tasks and CHTML, see
 * <http://www.chunks-and-tasks.org>.
 * 
 * === Copyright and license ===
 * 
 * Copyright (c) 2012-2021 Emanuel H. Rubensson and Elias Rudberg. All
 *                         rights reserved.
 * Copyright (c) 2016-2021 Anastasia Kruchinina and Anton Artemov. All
 *                         rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer listed
 *   in this license in the documentation and/or other materials provided
 *   with the distribution.
 * 
 * - Neither the name of the copyright holders nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 * 
 * The copyright holders provide no reassurances that the source code
 * provided does not infringe any patent, copyright, or any other
 * intellectual property rights of third parties. The copyright holders
 * disclaim any liability to any recipient for claims brought against
 * recipient by any third party for infringement of that parties
 * intellectual property rights.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef GBLAS_HEADER
#define GBLAS_HEADER
#if BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

extern "C" void dgemm_(const char *ta,const char *tb,
                       const int *n, const int *k, const int *l,
                       const double *alpha,const double *A,const int *lda,
                       const double *B, const int *ldb,
                       const double *beta, double *C, const int *ldc);
extern "C" void sgemm_(const char *ta,const char *tb,
                       const int *n, const int *k, const int *l,
                       const float *alpha,const float *A,const int *lda,
                       const float *B, const int *ldb,
                       const float *beta, float *C, const int *ldc);
extern "C" void dsymm_(const char *side,const char *uplo,
                       const int *m,const int *n,
                       const double *alpha,const double *A,const int *lda,
                       const double *B,const int *ldb, const double* beta,
                       double *C,const int *ldc);
extern "C" void ssymm_(const char *side,const char *uplo,
                       const int *m,const int *n,
                       const float *alpha,const float *A,const int *lda,
                       const float *B,const int *ldb, const float* beta,
                       float *C,const int *ldc);
extern "C" void dsyrk_(const char *uplo, const char *trans, const int *n,
                       const int *k, const double *alpha, const double *A,
                       const int *lda, const double *beta,
                       double *C, const int *ldc);
extern "C" void ssyrk_(const char *uplo, const char *trans, const int *n,
                       const int *k, const float *alpha, const float *A,
                       const int *lda, const float *beta,
                       float *C, const int *ldc);

extern "C" void dtrmm_(const char *side,const char *uplo,const char *transa,
                       const char *diag,const int *m,const int *n,
                       const double *alpha,const double *A,const int *lda,
                       double *B,const int *ldb);
extern "C" void strmm_(const char *side,const char *uplo,const char *transa,
                       const char *diag,const int *m,const int *n,
                       const float *alpha,const float *A,const int *lda,
                       float *B,const int *ldb);
extern "C" void daxpy_(const int* n, const double* da, const double* dx,
		       const int* incx, double* dy,const int* incy);
extern "C" void saxpy_(const int* n, const float* da, const float* dx,
		       const int* incx, float* dy,const int* incy);
extern "C" void dsyev_(const char *jobz, const char *uplo, const int *n,
		       double *a, const int *lda, double *w, double *work,
		       const int *lwork, int *info);
extern "C" void dgemv_(const char *ta, const int *m, const int *n,
		       const double *alpha, const double *A, const int *lda,
		       const double *x, const int *incx, const double *beta,
		       double *y, const int *incy);
extern "C" void sgemv_(const char *ta, const int *m, const int *n,
		       const float *alpha, const float *A, const int *lda,
		       const float *x, const int *incx, const float *beta,
		       float *y, const int *incy);

extern "C" void dsymv_(const char *uplo, const int *n,
		       const double *alpha, const double *A, const int *lda,
		       const double *x, const int *incx, const double *beta,
		       double *y, const int *incy);
extern "C" void ssymv_(const char *uplo, const int *n,
		       const float *alpha, const float *A, const int *lda,
		       const float *x, const int *incx, const float *beta,
		       float *y, const int *incy);

extern "C" double ddot_(const int *n, const double *dx, const int *incx,
                      const double *dy, const int *incy);

extern "C" float sdot_(const int *n, const float *dx, const int *incx,
                      const float *dy, const int *incy);

extern "C" void dpstrf_(const char *uplo, const int *n, double *A, 
      const int *lda,  int *piv, int *rank, const double *tol, double *work, 
      int *info); //Added by William Samuelsson

extern "C" void dtrtri_(const char *uplo, const char *diag, const int *N, double *A, 
              const int *lda, int *info); //Added by William Samuelsson

extern "C" void dpotrf_(const char *uplo, const int *N, double *A, const int *lda, 
                        int *info); //Added by William Samuelsson



inline void gemm(const char *ta,const char *tb,
		 const int *n, const int *k, const int *l,
		 const double *alpha,const double *A,const int *lda,
		 const double *B, const int *ldb,
		 const double *beta, double *C, const int *ldc) {
  dgemm_(ta,tb,n,k,l,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void gemm(const char *ta,const char *tb,
		 const int *n, const int *k, const int *l,
		 const float *alpha,const float *A,const int *lda,
		 const float *B, const int *ldb,
		 const float *beta, float *C, const int *ldc) {
  sgemm_(ta,tb,n,k,l,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void symm(const char *side,const char *uplo,
		 const int *m,const int *n,
		 const double *alpha,const double *A,const int *lda,
		 const double *B,const int *ldb, const double* beta,
		 double *C,const int *ldc) {
  dsymm_(side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void symm(const char *side,const char *uplo,
		 const int *m,const int *n,
		 const float *alpha,const float *A,const int *lda,
		 const float *B,const int *ldb, const float* beta,
		 float *C,const int *ldc) {
  ssymm_(side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void syrk(const char *uplo, const char *trans, const int *n,
		 const int *k, const double *alpha, const double *A,
		 const int *lda, const double *beta,
		 double *C, const int *ldc) {
  dsyrk_(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
inline void syrk(const char *uplo, const char *trans, const int *n,
		 const int *k, const float *alpha, const float *A,
		 const int *lda, const float *beta,
		 float *C, const int *ldc) {
  ssyrk_(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
inline void trmm(const char *side,const char *uplo,const char *transa,
		 const char *diag,const int *m,const int *n,
		 const double *alpha,const double *A,const int *lda,
		 double *B,const int *ldb) {
  dtrmm_(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb);
}
inline void trmm(const char *side,const char *uplo,const char *transa,
		 const char *diag,const int *m,const int *n,
		 const float *alpha,const float *A,const int *lda,
		 float *B,const int *ldb) {
  strmm_(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb);
}
inline void axpy(const int* n, const double* da, const double* dx,
		 const int* incx, double* dy,const int* incy) {
  daxpy_(n, da, dx, incx, dy, incy);
}
inline void axpy(const int* n, const float* da, const float* dx,
		 const int* incx, float* dy,const int* incy) {
  saxpy_(n, da, dx, incx, dy, incy);
}


inline void syev(const char *jobz, const char *uplo, const int *n,
		       double *a, const int *lda, double *w, double *work,
		       const int *lwork, int *info)
{
  dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}


inline void gemv(const char *ta, const int *m, const int *n,
		       const double *alpha, const double *A, const int *lda,
		       const double *x, const int *incx, const double *beta,
		       double *y, const int *incy){
    dgemv_(ta, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void gemv(const char *ta, const int *m, const int *n,
		       const float *alpha, const float *A, const int *lda,
		       const float *x, const int *incx, const float *beta,
		       float *y, const int *incy){
    sgemv_(ta, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void symv(const char *uplo, const int *n,
		       const double *alpha, const double *A, const int *lda,
		       const double *x, const int *incx, const double *beta,
		       double *y, const int *incy){
   dsymv_(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void symv(const char *uplo, const int *n,
		       const float *alpha, const float *A, const int *lda,
		       const float *x, const int *incx, const float *beta,
		       float *y, const int *incy){
    ssymv_(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline double ddot(const int *n, const double *dx, const int *incx,
                      const double *dy, const int *incy){
    return ddot_(n,dx,incx,dy,incy);
}

inline float ddot(const int *n, const float *dx, const int *incx,
                      const float *dy, const int *incy){
    return sdot_(n,dx,incx,dy,incy);
}

inline void dpstrf(const char *uplo, const int *n, double *A,
              const int *lda,  int *piv, int *rank, const double *tol, double *work, 
              int *info){
        return dpstrf_(uplo,n,A,lda,piv,rank,tol,work, info); //Added by William Samuelsson
        }

inline void dtrtri(const char *uplo, const char *diag, const int *N, double *A, 
              const int *lda, int *info){
                return dtrtri_(uplo, diag, N, A, lda, info); //Added by William Samuelsson
              }
inline void dpotrf(const char *uplo, const int *N, double *A, const int *lda, 
                        int *info){
                      return dpotrf_(uplo, N, A, lda, info); //Added by William Samuelsson
                      }
#if BUILD_WITH_CUDA
inline cublasStatus_t cublasgemm(cublasHandle_t handle,
				 cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
				 const float *alpha, const float *A, int lda,
				 const float *B, int ldb, const float *beta,
				 float *C, int ldc)
{
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasgemm(cublasHandle_t handle,
				 cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
				 const double *alpha, const double *A, int lda,
				 const double *B, int ldb, const double *beta,
				 double *C, int ldc)
{
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasgemmBatched(cublasHandle_t handle,
					cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
					const float *alpha, const float *Aarray[], int lda,
					const float *Barray[], int ldb, const float *beta,
					float *Carray[], int ldc, int batchCount)
{
#if CUDART_VERSION >= 4010
#if 1
  // Workaround since batch sizes larger than 256^2 results in
  // degraded performance for most matrix sizes that have been tested
  // on K20m (on Erik, Lunarc).
  cublasStatus_t status1;
  int max_count = 64000;
  int no_of_calls = batchCount / max_count;
  for (unsigned int ind = 0; ind < no_of_calls; ind++ ) {
    status1 = cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, max_count);
    if (status1 != CUBLAS_STATUS_SUCCESS)
      return status1;
    batchCount -= max_count;
    Aarray += max_count;
    Barray += max_count;
    Carray += max_count;
  }
  if ( batchCount > 0)
    status1 = cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
  return status1;
#else
  return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#endif
#else
  return CUBLAS_STATUS_SUCCESS;
#endif
}

inline cublasStatus_t cublasgemmBatched(cublasHandle_t handle,
					cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
					const double *alpha, const double *Aarray[], int lda,
					const double *Barray[], int ldb, const double *beta,
					double *Carray[], int ldc,
					int batchCount)
{
#if CUDART_VERSION >= 4010
#if 1
  // Workaround since batch sizes larger than 256^2 results in
  // degraded performance for most matrix sizes that have been tested
  // on K20m (on Erik, Lunarc).
  cublasStatus_t status1;
  int max_count = 64000;
  int no_of_calls = batchCount / max_count;
  for (unsigned int ind = 0; ind < no_of_calls; ind++ ) {
    status1 = cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, max_count);
    if (status1 != CUBLAS_STATUS_SUCCESS)
      return status1;
    batchCount -= max_count;
    Aarray += max_count;
    Barray += max_count;
    Carray += max_count;
  }
  if ( batchCount > 0)
    status1 = cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
  return status1;
#else
  return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#endif
#else
  return CUBLAS_STATUS_SUCCESS;
#endif
}
#endif

#endif

#ifndef _BLOCKMATRIX_
#define _BLOCKMATRIX_

#include<vector>
#include<fstream>
#include<iostream>
#include<string>
#include<random>
#include<algorithm>
#include<cstdlib>
#include<cmath>
#include<stdexcept>
#include<cassert>
#include"gblas.h"


template<class classType>
class Matrix{
    public:
        //Constructors
        Matrix(){n_rows = 0; n_cols = 0;};
        Matrix(int i, int j);
        Matrix(int i, int j, std::vector<classType> entries); //Vector containing columns
        
        
        //Matrix generators
        static void diagonal_matrix(Matrix<classType> &A, std::vector<classType> const &diags);
        static void random_matrix(Matrix<classType> &A);
        static void identity_matrix(Matrix<classType> &A);
        static void orthonormal_matrix(Matrix<classType> &A);
        static void zeros(Matrix<classType> &A);
        static void chol(Matrix<classType> &A);
        
        //Matrix operations
        static void invert_lower_triangular(Matrix<classType> const &A, Matrix<classType> &B); // B = inv(A)
        static void transpose(Matrix <classType> const&A , Matrix<classType> &B);
        static void multiply(Matrix<classType> const &A, Matrix<classType> const&B, Matrix<classType> &C, bool ta = false, bool tb = false);
        static void add(Matrix<classType> const &A, classType const &Acoeff, Matrix<classType> const&B, classType const &Bcoeff, Matrix<classType> &C);
        static void schur_multiply(Matrix<classType> const &A, Matrix<classType> const &B, Matrix<classType> const &C);
        static classType trace(Matrix<classType> const &A);
        static void set_matrix_data(std::vector<classType> const &inData, Matrix<classType> &B);
        static void scale_matrix_data(const classType alpha, Matrix<classType> &A);
        static void commutator(Matrix<classType> const &A, Matrix<classType> const&B, Matrix<classType>  &C);
        void reshape(int new_rows, int new_cols){n_rows = new_rows;n_cols = new_cols; data.resize(n_rows*n_cols);}
        static void hstack(Matrix<classType> const &A, Matrix<classType> const &B, Matrix<classType> &C);
        static void vstack(Matrix<classType> const &A, Matrix<classType> const &B, Matrix<classType> &C);
        void clear(){n_rows = 0; n_cols = 0; data.clear();}
        Matrix<classType> get_upper_right_block()const{throw std::runtime_error("Error in Matrix::get_upper_right_block: Not implemented. ");};
        static classType first_order_frobenius_error(Matrix<classType> const &A){throw std::runtime_error("Error in Matrix::first_order_frobenius_error: Not implemented");};
        //Copy move
        void copy(const Matrix<classType> &other);
        //Norm
        static classType frobenius_norm_squared(Matrix<classType> const &A);
        static classType idempotency_error(Matrix<classType> const &A);
        //Selectors
        int get_n_rows()const;
        int get_n_cols()const;
        const classType * get_data_ptr() const {return &data[0];};
        classType * get_fluid_data_ptr() {return &data[0];};
        const classType * get_column_ptr(const int colnum) const {return &data[colnum * n_rows];};
        classType * get_fluid_column_ptr(const int colnum) {return &data[colnum * n_rows];};
        classType get_matrix_element(const int &ROW, const int &COL){assert(ROW < n_rows && COL < n_cols); return data[COL * n_rows + ROW];};
        void set_matrix_element(const int &ROW, const int &COL, classType const &VAL){assert(ROW < n_rows && COL < n_cols); data[COL * n_rows + ROW] = VAL;}
        void lower_triangular();
        const classType get_const_matrix_element(const int &ROW, const int &COL) const {return data[COL * n_rows + ROW];};
        std::vector<classType> get_diagonal_elements();
        std::vector<classType> get_lower_triangular_matrix();
        void write_matrix()const; //Write contents of Matrix in terminal
        void write_to_file(std::string filename)const; //Write contents of Matrix to filename
        void gresgorin(classType &e_min, classType &e_max);
        //Inner product of columns 
        static classType dot(int const& colnum1, int const& colnum2, Matrix<classType> const &A);
        //Add columns. Performs Col2 = Alpha*Col1 + Col2
        static void add_columns(int const &colnum1, int const& colnum2, classType const ALPHA, Matrix<classType> &A); 

    private: 
    //Matrix elements stored in vector (column format!)
        std::vector<classType> data;
    //No of rows and columns in matrix
        int n_rows;
        int n_cols;
    //For use in blas
        struct Transpose{
            const char * transposeptr;
            Transpose(const char *transposeptr): transposeptr(transposeptr) {}
            static Transpose N(){return 'N';}
            
        };
    
        
};

template<class classType>
void Matrix<classType>::orthonormal_matrix(Matrix<classType> &A){
    int Nrows = A.get_n_rows();
    int Ncols = A.get_n_cols();
    classType normsq = 0.0;
    classType ALPHA = 0.0;

    for (int i = 0; i < Ncols; i++){
        for (int j = 0; j < i; j++){
            classType proj = -1.0 * Matrix<classType>::dot(i,j,A); //proj_cj(ci)
            Matrix<classType>::add_columns(j,i,proj, A); // u_i = u_i - proj*u_j
        }
        //Normalize
        normsq = Matrix<classType>::dot(i,i,A);
        ALPHA = 1/std::sqrt(normsq) - 1;
        Matrix<classType>::add_columns(i,i,ALPHA,A);
    }

}

template<class classType>
void Matrix<classType>::copy(const Matrix<classType> &other){
    n_rows = other.get_n_rows();
    n_cols = other.get_n_cols();

    data.resize(n_rows*n_cols);

    const classType * otherptr = other.get_data_ptr();

    for (int i = 0; i < n_rows*n_cols; i++){
        data[i] = otherptr[i];
    }       
}

template<class classType>
classType Matrix<classType>::dot(int const& colnum1, int const& colnum2, Matrix<classType> const &A){
    const int Ncols = A.get_n_cols();
    const int Nrows = A.get_n_rows();
    
    assert(Ncols * Nrows > 0);
    
    const int INC = 1;

    if (colnum1 < Ncols && colnum2 < Ncols){
        const classType * Xptr = A.get_column_ptr(colnum1);
        const classType * Yptr = A.get_column_ptr(colnum2);

        
        return ddot_(&Nrows, Xptr, &INC, Yptr, &INC);
    }
    else{
        throw std::runtime_error("Error in Matrix::dot: Index out of bounds");
    }
    
}
template<class classType>
void Matrix<classType>::add_columns(int const &colnum1, int const& colnum2, classType const ALPHA, Matrix<classType> &A){
    
    const int Ncols = A.get_n_cols();
    const int Nrows = A.get_n_rows();

    assert(Ncols * Nrows > 0);
    
    const int INC = 1;

    if (colnum1 < Ncols && colnum2 < Ncols){
        const classType * ALPHAptr = &ALPHA;
        const classType * Xptr = A.get_column_ptr(colnum1);
        classType * Yptr = A.get_fluid_column_ptr(colnum2);

        return daxpy_(&Nrows, ALPHAptr, Xptr, &INC, Yptr, &INC);
    }
    else{
        throw std::runtime_error("Error in Matrix::dot: Index out of bounds");
    }

}
template<class classType>
void Matrix<classType>::multiply(Matrix<classType> const &A, Matrix<classType> const&B, Matrix<classType> &C, bool ta, bool tb){
    int m = A.get_n_rows();
    int n = B.get_n_rows();
    int k = A.get_n_cols();
    int l = B.get_n_cols();

    assert(m > 0 && n > 0 && k>0 && l>0);
    if(k == n && ta==false && tb == false){ 
        //duplicate rows in A, B, Ci nto lda ldb ldc for readability
        const int lda = m;
        const int ldb = k;
        const int ldc = m;

        char transposeState = 'N';
        Transpose transa(&transposeState);
        Transpose transb(&transposeState);
        
        classType alpha = 1.0;
        classType beta = 0.0;

        C.data.resize(m * l);

        const classType * A_data_ptr = A.get_data_ptr();
        const classType * B_data_ptr = B.get_data_ptr();
        classType * C_data_ptr = C.get_fluid_data_ptr();

        gemm(transa.transposeptr, transb.transposeptr, &m, &n, &k, &alpha, A_data_ptr, &lda, B_data_ptr, &ldb, &beta, C_data_ptr, &ldc); 
    }
    else if (m == l && ta == true && tb == true)
    {
        //duplicate rows in A, B, Ci nto lda ldb ldc for readability
        const int lda = m;
        const int ldb = k;
        const int ldc = m;

        char transposeState = 'T';
        Transpose transa(&transposeState);
        Transpose transb(&transposeState);
        
        classType alpha = 1.0;
        classType beta = 0.0;

        C.data.resize(k * n);

        const classType * A_data_ptr = A.get_data_ptr();
        const classType * B_data_ptr = B.get_data_ptr();
        classType * C_data_ptr = C.get_fluid_data_ptr();

        gemm(transa.transposeptr, transb.transposeptr, &m, &n, &k, &alpha, A_data_ptr, &lda, B_data_ptr, &ldb, &beta, C_data_ptr, &ldc); 
    }

    else if (m == n && ta == true && tb == false)
    {
        //duplicate rows in A, B, Ci nto lda ldb ldc for readability
        const int lda = m;
        const int ldb = k;
        const int ldc = m;

        char transposeStateA = 'T';
        Transpose transa(&transposeStateA);
        char transposeStateB = 'N'; 
        Transpose transb(&transposeStateB);
        
        classType alpha = 1.0;
        classType beta = 0.0;

        C.data.resize(k * l);

        const classType * A_data_ptr = A.get_data_ptr();
        const classType * B_data_ptr = B.get_data_ptr();
        classType * C_data_ptr = C.get_fluid_data_ptr();

        gemm(transa.transposeptr, transb.transposeptr, &m, &n, &k, &alpha, A_data_ptr, &lda, B_data_ptr, &ldb, &beta, C_data_ptr, &ldc); 
    }

    else if (k == l && ta == false && tb == true)
    {
        //duplicate rows in A, B, Ci nto lda ldb ldc for readability
        const int lda = m;
        const int ldb = k;
        const int ldc = m;

        char transposeStateA = 'N';
        Transpose transa(&transposeStateA);
        char transposeStateB = 'T'; 
        Transpose transb(&transposeStateB);
        
        classType alpha = 1.0;
        classType beta = 0.0;

        C.data.resize(m * n);

        const classType * A_data_ptr = A.get_data_ptr();
        const classType * B_data_ptr = B.get_data_ptr();
        classType * C_data_ptr = C.get_fluid_data_ptr();

        gemm(transa.transposeptr, transb.transposeptr, &m, &n, &k, &alpha, A_data_ptr, &lda, B_data_ptr, &ldb, &beta, C_data_ptr, &ldc); 
    }
    
    
    
    else{
        throw std::runtime_error("Error in multiply: Dimensions must agree.");
    }
}
template<class classType> 
void schur_multiply(Matrix<classType> const &A, Matrix<classType> const &B, Matrix<classType> const &C){
    int m = A.get_n_rows();
    int n = B.get_n_rows();
    int k = A.get_n_cols();
    int l = B.get_n_cols();

    assert(m > 0 && n > 0 && k>0 && l>0);

    if (m == n && k == l){
        C.data.resize(m*k);

        const classType * A_data_ptr = A.get_data_ptr();
        const classType * B_data_ptr = B.get_data_ptr();
        classType * C_data_ptr = C.get_fluid_data_ptr();

        for (int i = 0; i < m*k; i++){
            C_data_ptr[i] = A_data_ptr[i] * B_data_ptr[i];
        }
    }
    else{
        throw std::runtime_error("Error in schur_multiply: Matrix dimensions must agree");
    }
}
template<class classType>
void Matrix<classType>::add(Matrix<classType> const &A, classType const &Acoeff, Matrix<classType> const&B, classType const &Bcoeff, Matrix<classType> &C){
    int rowsInA = A.get_n_rows();
    int colsInA = A.get_n_cols();
    int rowsInB = B.get_n_rows();
    int colsInB = B.get_n_cols();

    if(rowsInA == rowsInB && colsInA == colsInB){
        C.data.resize(colsInA*rowsInA);
        for (int i=0; i < colsInA*rowsInA; i++){
            C.data[i] = Acoeff *A.data[i] + Bcoeff * B.data[i];
        }
    }
    else{
        throw std::runtime_error("Error in add: Dimensions must agree.");
    }

}


template<class classType>
int Matrix<classType>::get_n_cols()const{
    return n_cols;
}

template<class classType>
int Matrix<classType>::get_n_rows()const{
    return n_rows;
}

template<class classType>
void Matrix<classType>::write_matrix()const{
    
    for (int i = 0; i < n_rows; i++){
        for(int j = 0; j < n_cols; j++){
            std::cout << data[i + j*n_rows] << "    ";
        }
    std::cout << std::endl;
    }   
}

template<class classType>
void Matrix<classType>::random_matrix(Matrix<classType> &A){
    classType * dataptr = A.get_fluid_data_ptr();
    std::random_device rd{};
    std::mt19937 generator{rd()};
    int iterations = A.get_n_cols() * A.get_n_rows();
    std::normal_distribution<classType> distribrution(0, 1); 
    for (int i = 0; i < iterations; i++){
        dataptr[i] = distribrution(generator);
    }
    
}

template<class classType>
void Matrix<classType>::zeros(Matrix<classType> &A){
    assert(A.get_n_rows() > 0 && A.get_n_cols() > 0);
    classType * dataptr = A.get_fluid_data_ptr();
    for (int i = 0; i < A.get_n_cols() * A.get_n_cols();i++){
        dataptr[i] = 0.0;
    } 
}

template<class classType>
void Matrix<classType>::diagonal_matrix(Matrix<classType> &A, std::vector<classType> const &diags){
    
    assert(A.get_n_cols() == A.get_n_rows());
    if (A.get_n_cols() == diags.size()){
        classType * dataptr = A.get_fluid_data_ptr();
        for (int i = 0; i < A.get_n_rows()*A.get_n_cols(); i++){
            dataptr[i] = 0.0;
            } 
        
        for (int i = 0; i < A.get_n_rows(); i++){
            dataptr[i + i*A.get_n_rows()] = diags[i];
        }
    }
    else{
        throw std::runtime_error("Error in Matrix::diagonal_matrix: Dimensions must match.");
    }
}

template<class classType>
void Matrix<classType>::identity_matrix(Matrix<classType> &A){
    int n_rows = A.get_n_rows();
    int n_cols = A.get_n_cols();

    if (n_cols == n_rows){
        classType * dataptr = A.get_fluid_data_ptr();
        for (int i = 0; i < n_rows*n_cols; i++){
            dataptr[i] = 0.0;
            } 
        
        for (int i = 0; i < n_rows; i++){
            dataptr[i + i*n_rows] = 1.0;
        }
    }
    else{
        throw std::runtime_error("Error in generate_identity_matrix: Must be square dimensions");
    }
}
template<class classType>
classType Matrix<classType>::trace(Matrix<classType> const &A){
    
    classType trace = 0.0;
    int rows = A.get_n_rows();
    int cols = A.get_n_cols();

    assert(rows > 0 && cols > 0);
    
    if(rows == cols){
        for (int i = 0; i < rows; i++){
            trace += A.data[i + i*cols];
        }
        return trace;
    }
    else{
        throw std::runtime_error("Error in trace: Trace only implemented for square matricies.");
    }
    
}

template<class classType>
void Matrix<classType>::transpose(Matrix<classType> const &A, Matrix<classType> &B){
    const int Nrows = A.get_n_rows();
    const int Ncols = A.get_n_cols();

    const classType * Adataptr = A.get_data_ptr();
    classType * Bdataptr = B.get_fluid_data_ptr();

    for (int i = 0; i < Nrows; i++){
        for (int j = 0; j < Ncols; j++){
            Bdataptr[j*Nrows + i] = Adataptr[i * Ncols + j];
        }
    }
};

template<class classType>
Matrix<classType>::Matrix(int i, int j, std::vector<classType> entries){
    data = entries;
    if(entries.size() == i*j){
        n_rows = i;
        n_cols = j;
    }
    else{
        throw std::runtime_error("Error in constructor: Number of matrix entries must be the same as rows*columns");
    }
    
}

template<class classType> 
Matrix<classType>::Matrix(int i, int j){
    assert(i>0 && j>0);
    n_rows = i;
    n_cols = j;
    data.reserve(n_cols * n_rows);
}

template<class classType>
void Matrix<classType>::set_matrix_data(std::vector<classType> const &inData, Matrix<classType> &B){
    if(inData.size() == B.data.size()){
        B.data = inData;}
    else{
        throw std::runtime_error("Error in set_matrix_data: Dimensions must match. ");
    }
}

template<class classType>
void Matrix<classType>::scale_matrix_data(const classType alpha, Matrix<classType> &A){
    int nrows = A.get_n_rows();
    int ncols = A.get_n_cols();

    classType * dataptr = A.get_fluid_data_ptr();

    for (int i = 0; i < nrows*ncols; i++){
        dataptr[i] = alpha*dataptr[i];
    }
}

template<class classType>
classType Matrix<classType>::frobenius_norm_squared(Matrix<classType> const &A){
    classType sum = 0.0;
    int Nrows = A.get_n_rows();
    int Ncols = A.get_n_cols();

    assert(Nrows > 0 && Ncols > 0);
    const classType * dataptr = A.get_data_ptr();
    for (int i = 0; i < Nrows*Ncols; i++){
        sum += dataptr[i] * dataptr[i];
    }
    return sum;
}

template<class classType>
void Matrix<classType>::commutator(Matrix<classType> const &A, Matrix<classType> const&B, Matrix<classType>  &C){
    int Nrows = A.get_n_rows();
    int Ncols = A.get_n_rows();

    Matrix<classType> FirstProduct(Nrows, Ncols);

    classType ONE = 1.0;
    classType MINUS_ONE = -1.0;

    Matrix<classType>::multiply(A, B, FirstProduct);
    Matrix<classType>::multiply(B,A, C);

    Matrix<classType>::add(FirstProduct, ONE, C, MINUS_ONE, C); //C = AB - BA


}

template<class classType>
void Matrix<classType>::hstack(Matrix<classType> const &A, Matrix<classType> const &B, Matrix<classType> &C){
    const classType * Aptr = A.get_data_ptr();
    const classType * Bptr = B.get_data_ptr();

    int Arows = A.get_n_rows();
    int Acols = A.get_n_cols();

    int Brows = B.get_n_rows();
    int Bcols = B.get_n_cols();

   
    if (Arows == Brows){
        C.clear();
        C.reshape(Arows, Acols + Bcols);

        classType * Cptr = C.get_fluid_data_ptr();
        for(int i = 0; i < Arows*Acols; i++){
            Cptr[i] = Aptr[i];
        }
        for(int i = 0; i < Brows*Bcols;i++){
            Cptr[i + Arows*Acols] = Bptr[i]; 
        }
        
        
    }
    else{
        throw std::runtime_error("Error in Matrix::hstack: Number of rows in stacked matricies must match. ");
    }
    

}

template<class classType>
void Matrix<classType>::vstack(Matrix<classType> const &A, Matrix<classType> const &B, Matrix<classType> &C){
    const classType * Aptr = A.get_data_ptr();
    const classType * Bptr = B.get_data_ptr();

    int Arows = A.get_n_rows();
    int Acols = A.get_n_cols();

    int Brows = B.get_n_rows();
    int Bcols = B.get_n_cols();

    if (Acols == Bcols){
        C.clear();
        C.reshape(Arows + Brows, Acols);

        classType * Cptr = C.get_fluid_data_ptr();
        
        
        for(int i = 0; i < Acols; i++){
            for(int j = 0; j < Arows; j++){
                Cptr[j + i*C.n_rows] = Aptr[j + i*Arows];
            }
        }
        for(int i = 0; i < Acols; i++){
            for(int j = 0; j < Arows; j++){
                Cptr[Arows + j + i*C.n_rows] = Bptr[j + i*Arows];
            }
        }

        }
    else{
        throw std::runtime_error("Error in Matrix::vstack: Number of columns in stacked matricies must match. ");
    }
}

template<class classType>
classType Matrix<classType>::idempotency_error(Matrix<classType> const &A){
    Matrix<classType> AA(A.get_n_rows(), A.get_n_cols());
    Matrix<classType>::multiply(A, A, AA);

    classType ONE = 1.0;

    Matrix<classType>::add(A, ONE, AA, -ONE, AA);

    return std::sqrt(Matrix<classType>::frobenius_norm_squared(AA));
}
template<class classType>
std::vector<classType> Matrix<classType>::get_diagonal_elements(){
    std::vector<classType> diagonal_elements;
    if(n_rows == n_cols){
        diagonal_elements.resize(n_rows);
        for (int i = 0; i < n_rows; i++){
            diagonal_elements[i] = get_matrix_element(i,i);
        }
    }
    else{
        throw std::runtime_error("Error in Matrix::get_diagonal_element: Operation only valid for square matrix");
    }
    return diagonal_elements;
}

template<class classType>
void Matrix<classType>::gresgorin(classType &e_min, classType &e_max){
    e_min = 0.0;
    e_max = 0.0;

    classType e = 0.0;
    classType r = 0.0;

    for (int i = 0; i < n_rows; i++){
        e = get_matrix_element(i,i);
        r = 0.0;
       
        for(int j = 0; j < n_rows; j++){
            r += std::abs(get_matrix_element(i,j));
        }
        r = r - std::abs(e);
        if ((e - r) < e_min){
            e_min = e-r;
        }
        else if ((e + r)  > e_max){
            e_max = e+r;
        }
    }
}

template<class classType>
void Matrix<classType>::write_to_file(std::string filename)const{
    std::ofstream fout(filename);

    for (int i = 0; i < n_rows; i++){
        for(int j = 0; j < n_cols; j++){
            fout << data[i + j*n_rows] << "    ";
        }
    fout << std::endl;
    }   

}

template<class classType>
void Matrix<classType>::chol(Matrix<classType> &A){
    //This matrix must be symmetric positive definite

    const int N = A.get_n_rows();
    assert(N == A.get_n_cols()); //Can atleast check that matrix is square

    const char uplo = 'L';
    classType * A_data_ptr = A.get_fluid_data_ptr();

    const int lda = N;
    int INFO;

    dpotrf(&uplo, &N, A_data_ptr, &lda, &INFO);
    if (INFO != 0){
        throw std::runtime_error("Error in Matrix::chol: Incomplete rank in final matrix"
        "or not postitve definite symmetric input");
    }
    A.lower_triangular();
}
template<class classType>
void Matrix<classType>::invert_lower_triangular(Matrix<classType> const &A, Matrix<classType> &B){
    B.copy(A);
    
    const char uplo = 'L';

    const char diag = 'N';

    const int N = A.get_n_rows();

    classType * A_data_ptr = B.get_fluid_data_ptr();

    const int lda = N;

    int INFO;

    dtrtri(&uplo, &diag, &N, A_data_ptr, &lda, &INFO);
    if (INFO != 0){
        throw std::runtime_error("Error in Matrix::invertlowertri: Trying to invert singular matrix");
    }
}
template<class classType>
std::vector<classType> Matrix<classType>::get_lower_triangular_matrix(){
    std::vector<classType> L;
    int N = n_rows;

    assert(N == n_cols);
    int size = N*(N+1) / 2;
    L.resize(size);

    int INDEX = 0;

    for (int i = 0; i < N; i++){
        for (int j = i; j < N; j++){
            L[INDEX] = get_matrix_element(j, i);
            INDEX++;
        }
    }
    return L;
}

template<class classType>
void Matrix<classType>::lower_triangular(){
    if (n_rows == n_cols){
        for (int i = 0;i < n_cols;i++){
            for(int j = 0; j < i; j++){
                set_matrix_element(j,i,0.0);
            }
        }
    }
}
#endif //_BLOCKMATRIX_
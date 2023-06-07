#ifndef __BUTM__
#define __BUTM__
#include<cmath>
template<class BlockMatrix, class classType>
class BlockUpperTriangularMatrix{
    
    
    /*Block Matrix on form: 
     _____________________
    |                     |
    |    A0         A1    |
    |                     |
    |                     |   
    |                     |
    |    NULL       A0    |
    |                     |
    |_____________________|
    */

    public:
        //Constructors
        BlockUpperTriangularMatrix(BlockMatrix const &A0, BlockMatrix const &A1);
        BlockUpperTriangularMatrix(int N, int M);
        BlockUpperTriangularMatrix();
        //Matrix Generators
        static void identity_matrix(BlockUpperTriangularMatrix<BlockMatrix, classType> &A);
        //Matrix operations
        static void multiply(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A,BlockUpperTriangularMatrix<BlockMatrix, classType> const &B, BlockUpperTriangularMatrix<BlockMatrix, classType> &C);
        static void add(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A, classType const &Acoeff ,BlockUpperTriangularMatrix<BlockMatrix, classType > const &B, classType const &Bcoeff, BlockUpperTriangularMatrix<BlockMatrix, classType > &C);
        BlockMatrix get_upper_right_block(){return blocks[1];};
        BlockMatrix get_diagonal_block(){return blocks[0];};
        static classType trace(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A);
        static classType frobenius_norm_squared(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A){return 2*BlockMatrix::frobenius_norm_squared(A.get_constant_blocksptr()[0]) + BlockMatrix::frobenius_norm_squared(A.get_constant_blocksptr()[1]);};
        static classType first_order_frobenius_error(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A);
        static classType idempotency_error(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A);
        void set_blocks(BlockMatrix const &A0, BlockMatrix const &A1);
        void write_butm();

        //Selectors
        const BlockMatrix * get_constant_blocksptr()const{return &blocks[0];};
        BlockMatrix * get_fluid_blocksptr(){return &blocks[0];};
        int get_n_rows()const{return SubMatrixRows;};
        int get_n_cols()const{return SubMatrixCols;};

    private:
        int SubMatrixRows; //These should be the same for A1 and A0
        int SubMatrixCols; 

        std::vector<BlockMatrix> blocks;
        
        

};

template<class BlockMatrix, class classType>
void BlockUpperTriangularMatrix<BlockMatrix, classType>::set_blocks(BlockMatrix const &A0, BlockMatrix const &A1){
    BlockMatrix * blockptr = get_fluid_blocksptr();

    blocks.resize(2);
    blocks[0].copy(A0);
    blocks[1].copy(A1);

    SubMatrixRows = A0.get_n_rows();
    SubMatrixCols = A1.get_n_cols();

    
}
template<class BlockMatrix, class classType>
classType BlockUpperTriangularMatrix<BlockMatrix, classType>::trace(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A){
    const BlockMatrix * Aptr = A.get_constant_blocksptr();
    return 2*BlockMatrix::trace(Aptr[0]);
}
template<class BlockMatrix, class classType>
BlockUpperTriangularMatrix<BlockMatrix, classType>::BlockUpperTriangularMatrix(int N, int M){
    SubMatrixRows = N;
    SubMatrixCols = M;

    blocks.resize(2);
}

template<class BlockMatrix, class classType>
void BlockUpperTriangularMatrix<BlockMatrix, classType>::identity_matrix(BlockUpperTriangularMatrix<BlockMatrix, classType> &A){
    BlockMatrix I(A.get_n_rows(), A.get_n_cols());
    BlockMatrix zeros(A.get_n_rows(), A.get_n_cols());

    BlockMatrix::identity_matrix(I);
    BlockMatrix::zeros(zeros);

    BlockMatrix * Aptr = A.get_fluid_blocksptr();
    Aptr[0].copy(I);
    Aptr[1].copy(zeros);
}

template<class BlockMatrix, class classType >
BlockUpperTriangularMatrix<BlockMatrix, classType>::BlockUpperTriangularMatrix(BlockMatrix const &A0, BlockMatrix const &A1){
    int A0rows = A0.get_n_rows();
    int A1rows = A1.get_n_rows();

    int A0cols = A0.get_n_cols();
    int A1cols = A1.get_n_cols();

    assert(A0rows == A1rows && A0cols == A1cols);

    SubMatrixCols = A0cols;
    SubMatrixRows = A0rows;

    blocks.resize(2);
    blocks[0].copy(A0);
    blocks[1].copy(A1);

}

template<class BlockMatrix, class classType>
BlockUpperTriangularMatrix<BlockMatrix, classType>::BlockUpperTriangularMatrix(){
    SubMatrixCols = 0;
    SubMatrixRows = 0;

    blocks.resize(2);

}

template<class BlockMatrix, class classType>
void BlockUpperTriangularMatrix<BlockMatrix, classType>::write_butm(){
    BlockMatrix printable_matrix;
    BlockMatrix zeros(SubMatrixRows, SubMatrixRows);
    BlockMatrix::zeros(zeros);

    BlockMatrix toprow;
    BlockMatrix::hstack(blocks[0], blocks[1], toprow);

    BlockMatrix bottomrow;
    BlockMatrix::hstack(zeros, blocks[0], bottomrow);

    BlockMatrix::vstack(toprow, bottomrow, printable_matrix);

    printable_matrix.write_matrix();
    
}
    


template<class BlockMatrix, class classType>
 void BlockUpperTriangularMatrix<BlockMatrix, classType>::multiply(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A,BlockUpperTriangularMatrix<BlockMatrix, classType> const &B, BlockUpperTriangularMatrix<BlockMatrix, classType> &C){
    const BlockMatrix * Aptr = A.get_constant_blocksptr();
    const BlockMatrix * Bptr = B.get_constant_blocksptr();
    BlockMatrix * Cptr = C.get_fluid_blocksptr();

    C.SubMatrixCols = Aptr[0].get_n_cols();
    C.SubMatrixRows = Aptr[0].get_n_rows();

    Cptr[0].reshape(C.SubMatrixRows, C.SubMatrixCols);
    Cptr[1].reshape(C.SubMatrixRows, C.SubMatrixCols);

    BlockMatrix::multiply(Aptr[0], Bptr[0], Cptr[0]);

    BlockMatrix C1_temp(Aptr[0].get_n_rows(), Aptr[0].get_n_cols());

    
    BlockMatrix::multiply(Aptr[0], Bptr[1], C1_temp);

   
    BlockMatrix::multiply(Aptr[1], Bptr[0], Cptr[1]);
    const classType ONE = 1.0;
    BlockMatrix::add(C1_temp, ONE, Cptr[1], ONE, Cptr[1]);

}

template<class BlockMatrix, class classType> 
void BlockUpperTriangularMatrix<BlockMatrix, classType>::add(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A, classType const &Acoeff ,BlockUpperTriangularMatrix<BlockMatrix, classType> const &B, classType const &Bcoeff, BlockUpperTriangularMatrix<BlockMatrix, classType> &C){
    const BlockMatrix * Aptr = A.get_constant_blocksptr();
    const BlockMatrix* Bptr = B.get_constant_blocksptr();
    BlockMatrix * Cptr = C.get_fluid_blocksptr();
    C.SubMatrixCols = Aptr[0].get_n_cols();
    C.SubMatrixRows = Aptr[0].get_n_rows();

    Cptr[0].reshape(C.SubMatrixRows, C.SubMatrixCols);
    Cptr[1].reshape(C.SubMatrixRows, C.SubMatrixCols);

    BlockMatrix::add(Aptr[0], Acoeff, Bptr[0], Bcoeff, Cptr[0]);
    BlockMatrix::add(Aptr[1], Acoeff, Bptr[1], Bcoeff, Cptr[1]);
}
template<class BlockMatrix, class classType>
classType BlockUpperTriangularMatrix<BlockMatrix, classType>::idempotency_error(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A){
    BlockUpperTriangularMatrix<BlockMatrix, classType> AA(A.get_n_rows(), A.get_n_cols());

    classType ONE = 1.0; 

    BlockUpperTriangularMatrix<BlockMatrix, classType>::multiply(A, A, AA);

    BlockUpperTriangularMatrix<BlockMatrix, classType>::add(A, ONE, AA, -ONE, AA);

    return std::sqrt(BlockUpperTriangularMatrix<BlockMatrix, classType>::frobenius_norm_squared(AA));

}

template<class BlockMatrix, class classType>
classType BlockUpperTriangularMatrix<BlockMatrix, classType>::first_order_frobenius_error(BlockUpperTriangularMatrix<BlockMatrix, classType> const &A){
    const BlockMatrix * Aptr = A.get_constant_blocksptr();
    

    BlockMatrix error_matrix1(A.get_n_rows(), A.get_n_cols());
    BlockMatrix error_matrix2(A.get_n_rows(), A.get_n_cols());

    classType ONE = 1.0;

    BlockMatrix::multiply(Aptr[0], Aptr[1], error_matrix1);
    BlockMatrix::multiply(Aptr[1], Aptr[0], error_matrix2);

    BlockMatrix::add(error_matrix1, ONE, error_matrix2, ONE, error_matrix1);

    BlockMatrix::add(Aptr[1], ONE, error_matrix1, -ONE, error_matrix1);

    return std::sqrt(BlockMatrix::frobenius_norm_squared(error_matrix1)); 
    
}
#endif
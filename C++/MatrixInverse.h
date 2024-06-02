#pragma once
using namespace std;

// 将矩阵A进行LU分解
void luDecomposition(double** A, double**& L, double**& U, int n) {
    // LU分解
    for (int i = 0; i < n; ++i) {
        L[i][i] = 1;
        for (int j = 0; j < n; ++j) {
            U[i][j] = 0;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += L[i][j] * U[j][k];
            }
            U[i][k] = A[i][k] - sum;
        }

        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += L[k][j] * U[j][i];
            }
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

// 使用LU分解求解方程组AX = B
void luSolve(double** L, double** U, double** B, double**& X, int n) {
    double** Y = new double* [n];
    for (int i = 0; i < n; i++) {
        Y[i] = new double[n];
    }

    // LY=B,求Y
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L[j][k] * Y[k][i];
            }
            Y[j][i] = (B[j][i] - sum) / L[j][j];
        }
    }

    // UX=Y，求X
    for (int i = n - 1; i >= 0; i--) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = i + 1; k < n; k++) {
                sum += U[i][k] * X[k][j];
            }
            X[i][j] = (Y[i][j] - sum) / U[i][i];
        }
    }

    // 释放内存
    for (int i = 0; i < n; i++) {
        delete[] Y[i];
    }
    delete[] Y;
}

// 计算矩阵A的逆矩阵
void inverse(double** A, double**& A_inv, int n) {
    double** L = new double* [n];
    double** U = new double* [n];
    for (int i = 0; i < n; i++) {
        L[i] = new double[n];
        U[i] = new double[n];
    }

    luDecomposition(A, L, U, n);

    // 将B初始化为单位矩阵
    double** B = new double* [n];
    for (int i = 0; i < n; i++) {
        B[i] = new double[n];
        for (int j = 0; j < n; j++) {
            B[i][j] = (i == j) ? 1 : 0;
        }
    }

    luSolve(L, U, B, A_inv, n);

    // 释放内存
    for (int i = 0; i < n; i++) {
        delete[] L[i];
        delete[] U[i];
        delete[] B[i];
    }
    delete[] L;
    delete[] U;
    delete[] B;
}

//int main() {
//    int n = 4;
//    double** A = new double* [n];
//    double** A_inv = new double* [n];
//    for (int i = 0; i < n; i++) {
//        A[i] = new double[n];
//        A_inv[i] = new double[n];
//    }
//
//    A[0][0] = 4; A[0][1] = -1; A[0][2] = 0; A[0][3] = 0;
//    A[1][0] = -1; A[1][1] = 4; A[1][2] = -1; A[1][3] = 0;
//    A[2][0] = 0; A[2][1] = -1; A[2][2] = 4; A[2][3] = -1;
//    A[3][0] = 0; A[3][1] = 0; A[3][2] = -1; A[3][3] = 3;
//
//    inverse(A, A_inv, n);
//
//    cout << "Inverse of A:" << endl;
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            cout << A_inv[i][j] << "\t";
//        }
//        cout << endl;
//    }
//
//    for (int i = 0; i < n; i++) {
//        delete[] A[i];
//        delete[] A_inv[i];
//    }
//    delete[] A;
//    delete[] A_inv;
//
//    return 0;
//}

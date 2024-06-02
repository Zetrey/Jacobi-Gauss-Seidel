#pragma once
#include <cmath>
#include <iostream>
#include"MatrixInverse.h"//包含“求矩阵逆”头文件
//#include"QRDecomposition.h"
//#include<Eigen/Dense>
//#include<Eigen/Eigenvalues>

using namespace std;
//using namespace Eigen;

void multiply_m(double** A, double** B, double**& C, int n)//辅助函数，矩阵乘法(A[][] * B[][] = C[][])
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			C[i][j] = 0;
			for (int k = 0; k < n; ++k)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}
void multiply(double** A, double* b, double*& result, int n) //辅助函数，向量矩阵乘法(A[][] * b[] = result[])
{
	for (int i = 0; i < n; ++i)
	{
		result[i] = 0;
		for (int j = 0; j < n; ++j)
		{
			result[i] += A[i][j] * b[j];
		}
	}
}
//double multiply_vec(double* x, double* u, int n) //辅助函数，向量乘法(x[] * u[] = result)
//{
//	double result = 0;
//	for (int i = 0; i < n; i++)
//	{
//		result += x[i] * u[i];
//	}
//	return result;
//}
double MAX_mat(double* m, int n)//辅助函数，求向量内的最大值
{
	double max = m[0];
	for (int i = 1; i < n; i++)
	{
		if (m[i] > max)
		{
			max = m[i];
		}
	}
	return max;
}
//void normalize(double*& x, double*& u, int n)//辅助函数，向量归一化
//{
//	double norm = 0.0;
//	for (int i = 0; i < n; ++i)
//	{
//		norm += x[i] * x[i];
//	}
//	norm = sqrt(norm);
//	for (int i = 0; i < n; ++i)
//	{
//		u[i] /= norm;
//	}
//}
double spectralRadius(double** A, int n, int max_itrs = 1e5, double min_delta = 1e-9)//幂迭代
{
	double delta = INFINITY;

	double* x = new double[n];
	for (int i = 0; i < n; i++) { x[i] = 1; }

	double* y = new double[n];
	for (int i = 0; i < n; i++) { y[i] = 0.0; }

	/*double* u = new double[n];
	for (int i = 0; i < n; i++) { u[i] = 1; }*/

	double m = 0;
	double m_old = m;

	for (int i = 0; i < max_itrs; i++)
	{
		m_old = m;
		multiply(A, x, y, n);
		m = MAX_mat(y, n);
		for (int j = 0; j < n; j++)
		{
			x[j] = y[j] / m;
		}
		if (abs(m - m_old) <= min_delta) {
			break;
		}
	}
	return m;
}

//double spectralRadius_Eigen(const MatrixXd& matrix) { //使用eigen库验证结果
//	// Compute the eigenvalues of the matrix
//	EigenSolver<MatrixXd> solver(matrix);
//	VectorXcd eigenvalues = solver.eigenvalues();
//
//	// Find the maximum absolute value among the eigenvalues
//	double maxAbsEigenvalue = 0.0;
//	for (int i = 0; i < eigenvalues.size(); ++i) {
//		double absValue = abs(eigenvalues[i]);
//		if (absValue > maxAbsEigenvalue) {
//			maxAbsEigenvalue = absValue;
//		}
//	}
//	return maxAbsEigenvalue;
//}

double J_spectralRadius(double** A, int n)//先将A转化为雅可比迭代矩阵J，然后返回J的谱半径
{
	double** J = new double* [n];
	double** D = new double* [n];
	double** D_inv = new double* [n];
	double** I = new double* [n];
	double** D_invA = new double* [n];
	for (int i = 0; i < n; i++) {
		J[i] = new double[n];
		D[i] = new double[n];
		D_inv[i] = new double[n];
		I[i] = new double[n];
		D_invA[i] = new double[n];
	}//为所需要的矩阵分配内存

	for (int i = 0; i < n; i++)//定义D/L矩阵
	{
		for (int j = 0; j < n; j++)
		{
			if (i != j) {
				D[i][j] = 0.0;
				I[i][j] = 0.0;
			}
			else{
				D[i][i] = A[i][i];
				I[i][i] = 1.0;
			}
		}
	}
	inverse(D, D_inv, n);//求D的逆
	multiply_m(D_inv, A, D_invA, n);//求 D ^ -1 * A
	for (int i = 0; i < n; i++)// J = I - D ^ -1 * A
	{
		for (int j = 0; j < n; j++)
		{
			J[i][j] = I[i][j] - D_invA[i][j];
		}
	}
	//for (int i = 0; i < n; i++)
	//{
	//	cout << endl;
	//	for (int j = 0; j < n; j++)
	//	{
	//		cout <<J[i][j] << ' ';
	//	}
	//	cout << endl;
	//}
	double SR = spectralRadius(J, n);

	for (int i = 0; i < n; i++) {
		delete[]J[i];
		delete[]D[i];
		delete[]D_inv[i];
		delete[]I[i];
		delete[]D_invA[i];
	}
	delete[]J;
	delete[]D;
	delete[]D_inv;
	delete[]I;
	delete[]D_invA;

	return SR;
}
double G_spectralRadius(double** A, int n)//先将A转化为GS迭代矩阵G，然后返回G的谱半径
{
	double** G = new double* [n];
	double** D = new double* [n];
	double** U = new double* [n];
	double** L = new double* [n];
	double** D_L = new double* [n];
	double** D_L_inv = new double* [n];
	for (int i = 0; i < n; i++) {
		G[i] = new double[n];
		D[i] = new double[n];
		U[i] = new double[n];
		L[i] = new double[n];
		D_L[i] = new double[n];
		D_L_inv[i] = new double[n];
	}//为所需要的矩阵分配内存

	for (int i = 0; i < n; i++)//定义 D / L / U 矩阵
	{
		for (int j = 0; j < n; j++)
		{
			if (i == j) {
				D[i][j] = A[i][j];
				U[i][j] = 0.0;
				L[i][j] = 0.0;
			}
			else if (i > j) {
				D[i][j] = 0.0;
				U[i][j] = 0.0;
				L[i][j] = -A[i][j];
			}
			else{
				D[i][j] = 0.0;
				U[i][j] = -A[i][j];
				L[i][j] = 0.0;
			}
		}
	}

	for (int i = 0; i < n; i++)// D - L
	{
		for (int j = 0; j < n; j++)
		{
			D_L[i][j] = D[i][j] - L[i][j];
		}
	}
	inverse(D_L, D_L_inv, n);//求 (D - L) 的逆
	multiply_m(D_L_inv, U, G, n);// G = (D - L) ^ -1 * U
	//for (int i = 0; i < n; i++)//用于输出测试
	//{
	//	cout << endl;
	//	for (int j = 0; j < n; j++)
	//	{
	//		cout << D_L_inv[i][j] << ' ';
	//	}
	//	cout << endl;
	//}
	double SR = spectralRadius(G, n);//求G的最大特征值

	for (int i = 0; i < n; i++) {
		delete[]G[i];
		delete[]D[i];
		delete[]U[i];
		delete[]L[i];
		delete[]D_L[i];
		delete[]D_L_inv[i];
	}
	delete[]G;
	delete[]D;
	delete[]U;
	delete[]L;
	delete[]D_L;
	delete[]D_L_inv;

	return SR;
}
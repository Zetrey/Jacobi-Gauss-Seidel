#include<iostream>
#include<fstream>
#include<cstdlib>
#include<string>
#include<iomanip>
#include"SpectralRadius.h"//包含自写“谱半径”头文件
using namespace std;

double tolerance = 1e-9;//设置误差限
int maxIterations = 1000;//设置迭代最大次数

void Initialize_Output()//存储文件初始化
{
	ofstream file_writer1("data_jacobi.txt", ios_base::out);
	ofstream file_writer2("data_GS.txt", ios_base::out);
	ofstream file_writer3("data_RE_jacobi.txt", ios_base::out);
	ofstream file_writer4("data_RE_GS.txt", ios_base::out);
}
void Output_J(double* x, int col)//用于输出jacobi每步迭代结果（存放进文件）
{
	fstream f;
	f.open("data_jacobi.txt", ios::out | ios::app);
	for (int i = 0; i < col; i++)
	{
		f << fixed << setprecision(10) << x[i] << ' ';
	}
	f << endl;
	f.close();
}
void Output_GS(double* x, int col)//用于输出GS每步迭代结果（文件）
{
	fstream f;
	f.open("data_GS.txt", ios::out | ios::app);
	for (int i = 0; i < col; i++)
	{
		f << fixed << setprecision(10) << x[i] << ' ';
	}
	f << endl;
	f.close();
}
void Output_RE_jacobi(double x, int n)//用于输出jacobi迭代每步的残差（文件）
{
	fstream f;
	f.open("data_RE_jacobi.txt", ios::out | ios::app);
	f << fixed << setprecision(10) << x << endl;
	f.close();
}
void Output_RE_GS(double x, int n)//用于输出GS每步迭代的残差（文件）
{
	fstream f;
	f.open("data_RE_GS.txt", ios::out | ios::app);
	f << fixed << setprecision(10) << x << endl;
	f.close();
}
void Print(double* x, int col)//将向量打印到屏幕
{
	for (int i = 0; i < col; i++)
	{
		cout << fixed << setprecision(10) << x[i] << ' ';
	}
	cout << endl;
}

double residualError(double** A, double* x, double* b, int row)//返回残差|b-Ax|
{
	double e = 0;
	for (int i = 0; i < row; i++)
	{
		double sum = 0;
		for (int j = 0; j < row; j++)
		{
			sum += A[i][j] * x[j];
		}
		e += pow(b[i] - sum, 2);
	}
	return sqrt(e);
}

void jacobi(int n, double** A, double* b)//雅可比迭代
{
	int k = 0;
	double* x = new double[n];
	for (int i = 0; i < n; i++){
		x[i] = 0;
	}//初始化x向量为0
	double* x0 = new double[n];
	cout << "开始进行雅可比迭代" << endl;
	for (k; k < maxIterations; k++)
	{
		for (int i = 0; i < n; i++)
		{
			x0[i] = x[i];//保存旧值
		}
		//进行雅可比迭代
		for (int i = 0; i < n; ++i) 
		{
			double sum = 0;
			for (int j = 0; j < n; ++j)
			{
				if (i != j)
				{
					sum += A[i][j] * x0[j];
				}
			}
			x[i] = (b[i] - sum) / A[i][i];
		}
		double error = 0;
		for (int i = 0; i < n; ++i)
		{
			if (error < abs(x[i] - x0[i]))
				error = abs(x[i] - x0[i]);
		}
		Print(x0, n);
		Output_J(x0, n);
		Output_RE_jacobi(residualError(A, x0, b, n), n);
		if (error <= tolerance)//检查收敛精度
		{
			Print(x, n);
			Output_J(x, n);
			Output_RE_jacobi(residualError(A, x, b, n), n);
			cout << "迭代完毕！迭代次数： " << k + 1 << endl;
			break;
		}
		else if (k == maxIterations - 1)
		{
			cout << "未完成迭代，超出最大迭代次数限制！最大次数限制： " << maxIterations << endl;
		}
	}
	residualError(A, x, b, n);
	delete[]x, x0;
}
void GS(int n, double** A, double* b)//高斯-赛德尔迭代
{
	int k = 0;
	double* x = new double[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
	}//初始化x向量为0
	double* x0 = new double[n];
	cout << "开始进行高斯·赛德尔迭代" << endl;
	for (k; k < maxIterations; k++)
	{
		for (int i = 0; i < n; i++)
		{
			x0[i] = x[i];//保存旧值
		}
		//进行GS迭代
		for (int i = 0; i < n; ++i) 
		{
			double sum1 = 0, sum2 = 0;
			for (int j = 0; j < i; ++j) 
			{ 
				sum1 += A[i][j] * x[j]; 
			}
			for (int j = i + 1; j < n; j++) 
			{ 
				sum2 += A[i][j] * x0[j]; 
			}
			x[i] = (b[i] - sum1 - sum2) / A[i][i];
		}
		double error = 0;
		for (int i = 0; i < n; ++i)
		{
			if(error< abs(x[i] - x0[i]))
				error = abs(x[i] - x0[i]);
		}
		Print(x0, n);
		Output_GS(x0, n);
		Output_RE_GS(residualError(A, x0, b, n), n);
		if (error <= tolerance)//检查收敛精度
		{
			Print(x, n);
			Output_GS(x, n);
			Output_RE_GS(residualError(A, x, b, n), n);
			cout << "迭代完毕！迭代次数： " << k + 1 << endl;
			break;
		}
		if (k == maxIterations - 1)
		{
			cout << "未完成迭代，超出最大迭代次数限制！最大次数限制： " << maxIterations << endl;
		}
	}
	delete[]x, x0;
}
bool testFile(string name)//用于测试文件名是否存在
{
	ifstream f;
	f.open(name);
	if (f) { return 1; }
	else { return 0; }
}
bool getFileName(string& FileName_A,string& FileName_b, string m)//如果读取文件名合法返回true，否则输出“找不到文件”且返回false
{
	if (m == "1")
	{
		FileName_A = "mat_idx_3x3.dat";
		FileName_b = "rhs_3x3.dat";
		return 1;
	}
	else if (m == "2")
	{
		FileName_A = "mat_idx_50x50.dat";
		FileName_b = "rhs_50.dat";
		return 1;
	}
	else if (m == "3")
	{
		cout << "请输入存放A矩阵的文件名：";
		cin >> FileName_A;
		cout << "请输入存放b矩阵的文件名：";
		cin >> FileName_b;
		bool A = testFile(FileName_A),
			b = testFile(FileName_b);
		if (!A) {
			cout << "找不到文件: \"" << FileName_A << "\"" << endl;
		}
		if (!b) {
			cout << "找不到文件: \"" << FileName_b << "\"" << endl;
		}
		if (!A || !b) {
			return 0;
		}else{
			return 1;
		}
	}
	else
	{
		cout << "请重新输入！" << endl;
		return 0;
	}
}
void getMatSize(int& n, int& num, string FileName)//读取A矩阵第一行，存储矩阵信息
{
	ifstream f;
	f.open(FileName);
	if (!f) {
		cout << "找不到文件: \"" << FileName << "\"" << endl;
	}
	else {
		f >> n >> n >> num;
		f.close();
	}
}
void openFile_A(double**& A, int num, string FileName)//打开矩阵A（COO稀疏矩阵）
{
	int r, c;
	ifstream FileA;
	FileA.open(FileName);
	if (!FileA) {
		cout << "找不到文件: \"" << FileName << "\"" << endl;
	}
	else {
		FileA >> r >> r >> r;
		for (int i = 0; i < num; i++) {
			FileA >> r >> c;
			FileA >> A[r][c];
		}
	}
	FileA.close();
}
void openFile_b(double*& b, int n, string FileName)//打开向量b（稠密矩阵）
{
	ifstream Fileb;
	Fileb.open(FileName);
	if (!Fileb) {
		cout << "找不到文件: \"" << FileName << "\"" << endl;
	}
	else {
		for (int i = 0; i < n; i++){
			Fileb >> b[i];
		}
		Fileb.close();
	}
}

int main()
{
	string menu_select();//声明

	int n, num;//定义矩阵大小n及非零元素个数num
	string FileName_A, FileName_b;//定义存放A、b矩阵的文件名

	for (;;) {
		string m = menu_select();
		if (m != "0") {
			if (!getFileName(FileName_A, FileName_b, m))//获取文件名)
			{
				system("pause");
				continue;
			}
			else
			{
				getMatSize(n, num, FileName_A);//获取矩阵大小

				double** A = new double* [n];
				for (int i = 0; i < n; i++) { A[i] = new double[n]; }//根据n值申请动态二维数组a
				for (int i = 0; i < n; i++) {//初始化数组值为0
					for (int j = 0; j < n; j++) {
						A[i][j] = 0;
					}
				}

				double* b = new double[n];//申请动态数组b
				for (int i = 0; i < n; i++) {//初始化数组为0
					b[i] = 0;
				}

				openFile_A(A, num, FileName_A);//将A矩阵填入A数组
				openFile_b(b, n, FileName_b);//将b矩阵填入b数组
				Initialize_Output();//为输出结果文件做初始化

				jacobi(n, A, b);//进行Jacobi迭代
				GS(n, A, b);//进行Guass-Seidel迭代

				cout << "幂迭代法求得Jacobi迭代矩阵谱半径为：" << fixed << setprecision(10) << J_spectralRadius(A, n) << endl
					<< "幂迭代法求得Gauss-Seidel迭代矩阵谱半径为：" << fixed << setprecision(10) << G_spectralRadius(A, n) << endl;

				for (int i = 0; i < n; i++) {
					delete[]A[i];
				}
				delete[]A, b;//释放内存
				system("pause");
			}
		}
		else
		{
			break;
		}
	}
	return 0;
}
string menu_select()//菜单
{
	const char* m[5] =
	{
		"1.直接计算预设3x3矩阵（测试矩阵）",
		"2.直接计算预设50x50矩阵",
		"3.输入其他文件名",
		"0.退出"
	};
	int i, choice;
	string ch;
	system("cls");	//清屏
	for (i = 0; m[i]; i++)
	{
			cout << m[i] << endl;
	}
	cout << "\n";
	cout << "请输入选择：";
	cin >> ch;
	return(ch);
}
#include<iostream>
#include<fstream>
#include<cstdlib>
#include<string>
#include<iomanip>
#include"SpectralRadius.h"//������д���װ뾶��ͷ�ļ�
using namespace std;

double tolerance = 1e-9;//���������
int maxIterations = 1000;//���õ���������

void Initialize_Output()//�洢�ļ���ʼ��
{
	ofstream file_writer1("data_jacobi.txt", ios_base::out);
	ofstream file_writer2("data_GS.txt", ios_base::out);
	ofstream file_writer3("data_RE_jacobi.txt", ios_base::out);
	ofstream file_writer4("data_RE_GS.txt", ios_base::out);
}
void Output_J(double* x, int col)//�������jacobiÿ�������������Ž��ļ���
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
void Output_GS(double* x, int col)//�������GSÿ������������ļ���
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
void Output_RE_jacobi(double x, int n)//�������jacobi����ÿ���Ĳв�ļ���
{
	fstream f;
	f.open("data_RE_jacobi.txt", ios::out | ios::app);
	f << fixed << setprecision(10) << x << endl;
	f.close();
}
void Output_RE_GS(double x, int n)//�������GSÿ�������Ĳв�ļ���
{
	fstream f;
	f.open("data_RE_GS.txt", ios::out | ios::app);
	f << fixed << setprecision(10) << x << endl;
	f.close();
}
void Print(double* x, int col)//��������ӡ����Ļ
{
	for (int i = 0; i < col; i++)
	{
		cout << fixed << setprecision(10) << x[i] << ' ';
	}
	cout << endl;
}

double residualError(double** A, double* x, double* b, int row)//���زв�|b-Ax|
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

void jacobi(int n, double** A, double* b)//�ſɱȵ���
{
	int k = 0;
	double* x = new double[n];
	for (int i = 0; i < n; i++){
		x[i] = 0;
	}//��ʼ��x����Ϊ0
	double* x0 = new double[n];
	cout << "��ʼ�����ſɱȵ���" << endl;
	for (k; k < maxIterations; k++)
	{
		for (int i = 0; i < n; i++)
		{
			x0[i] = x[i];//�����ֵ
		}
		//�����ſɱȵ���
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
		if (error <= tolerance)//�����������
		{
			Print(x, n);
			Output_J(x, n);
			Output_RE_jacobi(residualError(A, x, b, n), n);
			cout << "������ϣ����������� " << k + 1 << endl;
			break;
		}
		else if (k == maxIterations - 1)
		{
			cout << "δ��ɵ����������������������ƣ����������ƣ� " << maxIterations << endl;
		}
	}
	residualError(A, x, b, n);
	delete[]x, x0;
}
void GS(int n, double** A, double* b)//��˹-���¶�����
{
	int k = 0;
	double* x = new double[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
	}//��ʼ��x����Ϊ0
	double* x0 = new double[n];
	cout << "��ʼ���и�˹�����¶�����" << endl;
	for (k; k < maxIterations; k++)
	{
		for (int i = 0; i < n; i++)
		{
			x0[i] = x[i];//�����ֵ
		}
		//����GS����
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
		if (error <= tolerance)//�����������
		{
			Print(x, n);
			Output_GS(x, n);
			Output_RE_GS(residualError(A, x, b, n), n);
			cout << "������ϣ����������� " << k + 1 << endl;
			break;
		}
		if (k == maxIterations - 1)
		{
			cout << "δ��ɵ����������������������ƣ����������ƣ� " << maxIterations << endl;
		}
	}
	delete[]x, x0;
}
bool testFile(string name)//���ڲ����ļ����Ƿ����
{
	ifstream f;
	f.open(name);
	if (f) { return 1; }
	else { return 0; }
}
bool getFileName(string& FileName_A,string& FileName_b, string m)//�����ȡ�ļ����Ϸ�����true������������Ҳ����ļ����ҷ���false
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
		cout << "��������A������ļ�����";
		cin >> FileName_A;
		cout << "��������b������ļ�����";
		cin >> FileName_b;
		bool A = testFile(FileName_A),
			b = testFile(FileName_b);
		if (!A) {
			cout << "�Ҳ����ļ�: \"" << FileName_A << "\"" << endl;
		}
		if (!b) {
			cout << "�Ҳ����ļ�: \"" << FileName_b << "\"" << endl;
		}
		if (!A || !b) {
			return 0;
		}else{
			return 1;
		}
	}
	else
	{
		cout << "���������룡" << endl;
		return 0;
	}
}
void getMatSize(int& n, int& num, string FileName)//��ȡA�����һ�У��洢������Ϣ
{
	ifstream f;
	f.open(FileName);
	if (!f) {
		cout << "�Ҳ����ļ�: \"" << FileName << "\"" << endl;
	}
	else {
		f >> n >> n >> num;
		f.close();
	}
}
void openFile_A(double**& A, int num, string FileName)//�򿪾���A��COOϡ�����
{
	int r, c;
	ifstream FileA;
	FileA.open(FileName);
	if (!FileA) {
		cout << "�Ҳ����ļ�: \"" << FileName << "\"" << endl;
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
void openFile_b(double*& b, int n, string FileName)//������b�����ܾ���
{
	ifstream Fileb;
	Fileb.open(FileName);
	if (!Fileb) {
		cout << "�Ҳ����ļ�: \"" << FileName << "\"" << endl;
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
	string menu_select();//����

	int n, num;//��������Сn������Ԫ�ظ���num
	string FileName_A, FileName_b;//������A��b������ļ���

	for (;;) {
		string m = menu_select();
		if (m != "0") {
			if (!getFileName(FileName_A, FileName_b, m))//��ȡ�ļ���)
			{
				system("pause");
				continue;
			}
			else
			{
				getMatSize(n, num, FileName_A);//��ȡ�����С

				double** A = new double* [n];
				for (int i = 0; i < n; i++) { A[i] = new double[n]; }//����nֵ���붯̬��ά����a
				for (int i = 0; i < n; i++) {//��ʼ������ֵΪ0
					for (int j = 0; j < n; j++) {
						A[i][j] = 0;
					}
				}

				double* b = new double[n];//���붯̬����b
				for (int i = 0; i < n; i++) {//��ʼ������Ϊ0
					b[i] = 0;
				}

				openFile_A(A, num, FileName_A);//��A��������A����
				openFile_b(b, n, FileName_b);//��b��������b����
				Initialize_Output();//Ϊ�������ļ�����ʼ��

				jacobi(n, A, b);//����Jacobi����
				GS(n, A, b);//����Guass-Seidel����

				cout << "�ݵ��������Jacobi���������װ뾶Ϊ��" << fixed << setprecision(10) << J_spectralRadius(A, n) << endl
					<< "�ݵ��������Gauss-Seidel���������װ뾶Ϊ��" << fixed << setprecision(10) << G_spectralRadius(A, n) << endl;

				for (int i = 0; i < n; i++) {
					delete[]A[i];
				}
				delete[]A, b;//�ͷ��ڴ�
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
string menu_select()//�˵�
{
	const char* m[5] =
	{
		"1.ֱ�Ӽ���Ԥ��3x3���󣨲��Ծ���",
		"2.ֱ�Ӽ���Ԥ��50x50����",
		"3.���������ļ���",
		"0.�˳�"
	};
	int i, choice;
	string ch;
	system("cls");	//����
	for (i = 0; m[i]; i++)
	{
			cout << m[i] << endl;
	}
	cout << "\n";
	cout << "������ѡ��";
	cin >> ch;
	return(ch);
}
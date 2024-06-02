import numpy as np

def read_coo_file(file_path):
    data = np.loadtxt(file_path)
    rows = data[:, 0].astype(int)
    cols = data[:, 1].astype(int)
    values = data[:, 2]
    return rows, cols, values

def read_dense_vector(file_path):
    return np.loadtxt(file_path)

def construct_dense_matrix(rows, cols, values, size):
    A = np.zeros((size, size))
    for r, c, v in zip(rows, cols, values):
        A[r, c] = v
    return A

def gauss_seidel(A, b, tol, max_iter=1000, omega=1.0):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iter_count = 0
    error = tol + 1

    while error > tol and iter_count < max_iter:
        error = 0
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i][i]

            error += abs(x_new[i] - x[i])

        x = x_new.copy()
        iter_count += 1
        print(x)

    if error <= tol:
        print(f"Converged in {iter_count} iterations.")
    else:
        print("Did not converge within the maximum number of iterations.")

    return x

def jacobi(A, b, tol, max_iter=1000):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iter_count = 0
    error = tol + 1

    while error > tol and iter_count < max_iter:
        error = 0
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum1) / A[i][i]

            error = max(error, abs(x_new[i] - x[i]))

        x = x_new.copy()
        iter_count += 1
        print(x)

    if error <= tol:
        print(f"Converged in {iter_count} iterations.")
    else:
        print("Did not converge within the maximum number of iterations.")

    return x

if __name__ == "__main__":
    # 从COO格式文件读取数据
    coo_file_path = 'mat_idx_50x50.dat'
    rows, cols, values = read_coo_file(coo_file_path)

    # 从稠密矩阵文件读取向量b
    b_file_path = 'rhs_50.dat'
    b = read_dense_vector(b_file_path)

    # 构建稠密矩阵
    matrix_size = max(max(rows), max(cols)) + 1   # 计算矩阵大小
    A = construct_dense_matrix(rows, cols, values, matrix_size)

    tol = 1e-9
    max_iter = 1000

    gs_solution = gauss_seidel(A, b, tol, max_iter)
    print("Gauss-Seidel Solution:", gs_solution)

    jacobi_solution = jacobi(A, b, tol, max_iter)
    print("Jacobi Solution:", jacobi_solution)
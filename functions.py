import pandas as pd
import numpy as np

def load_excel_to_P():
    # 读取Excel文件中的矩阵
    file_path = 'initialmatrix.xlsx'
    sheet_name = 'Sheet1'  # 可以根据实际情况更改工作表名称

    # 使用pandas读取Excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # 将DataFrame转换为矩阵
    matrix = df.values

    # print(matrix)
    return matrix

def load_excel_to_A():
    # 读取Excel文件中的矩阵
    file_path = 'matrix2.xlsx'
    sheet_name = 'Sheet1'  # 可以根据实际情况更改工作表名称

    # 使用pandas读取Excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # 将DataFrame转换为矩阵
    matrix = df.values

    # print(matrix)
    return matrix


def get_min_Standard_Deviation(matrix):
    # 对每个列向量进行单位化
    normalized_matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)

    # 计算每列的标准偏差
    std_devs = np.std(normalized_matrix, axis=0)
    # 找到标准偏差最小的列索引
    min_std_index = np.argmin(std_devs)
    # 返回最小标准偏差
    return std_devs[min_std_index]


def normalized_vector(matrix):
    # 这里沿用get_min_Standard_Deviation()的代码，可以单位化矩阵。但实际上是单位化向量
    normalized_matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
    return normalized_matrix


def get_T_init(P,E_ac,f_i):
    # 确保输入的矩阵和向量是numpy数组
    P = np.array(P)
    E_ac = np.array(E_ac)
    f_i = np.array(f_i)

    set=[]

    max_T = -np.inf
    max_vector = None


    # 遍历矩阵P的每一个列向量
    for j in range(P.shape[1]):
        # 跳过f_i列向量
        if np.array_equal(P[:, j], f_i):
            continue

        # 当前列向量
        f_j = P[:, j]

        # 计算到E_ac的垂直分量：c=1时直接减不用求和
        e=normalized_vector(f_j)
        perpendicular_component = normalized_vector(f_j) - normalized_vector(f_j).T @ normalized_vector(f_i) * normalized_vector(f_i)

        # 取模
        T_j = np.linalg.norm(perpendicular_component)
        set.append(T_j)
        # 更新最大值和对应列向量
        if T_j > max_T:
            max_T = T_j
            max_vector = f_j

    return max_T, max_vector


def make_new_space_in_cycle1(f_i,E_ac):
    E_ac=np.c_[E_ac,normalized_vector(f_i)]
    return E_ac


def get_T(P,E_ac,f_selected):
    # 确保输入的矩阵和向量是numpy数组
    P = np.array(P)
    E_ac = np.array(E_ac)
    f_selected = np.array(f_selected)

    set = []
    # 计算E_ac的伪逆
    E_ac_pseudo_inv = np.linalg.pinv(E_ac)

    max_T = -np.inf
    max_vector = None

    # 遍历矩阵P的每一个列向量
    for j in range(P.shape[1]):
        # 当前列向量
        f_j = P[:, j]

        # 跳过f_i列向量
        if any(np.array_equal(f_j, f) for f in f_selected):
            continue

        # 计算到E_ac的垂直分量
        perpendicular_component = normalized_vector(f_j) - E_ac @ (E_ac_pseudo_inv @ normalized_vector(f_j))

        # 取模
        T_j = np.linalg.norm(perpendicular_component)
        set.append(T_j)
        # 更新最大值和对应列向量
        if T_j > max_T:
            max_T = T_j
            max_vector = f_j


    # print(set)
    return max_T, max_vector


def get_index_set(f_selected,P):
    indices = []

    # 确保输入的矩阵和向量是numpy数组
    P = np.array(P)
    f_selected = [np.array(f) for f in f_selected]

    # 遍历向量数组中的每个向量
    for f in f_selected:
        for j in range(P.shape[1]):
            if np.array_equal(P[:, j], f):
                indices.append(j)
                break

    return indices


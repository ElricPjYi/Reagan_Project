# Powered by ElricPj 2024/6/1
import numpy as np

import functions


# 读取Excel表数据，作为矩阵P_{a*b}.P={f_1,f_2,...,f_b}
P=functions.load_excel_to_P()
a=P.shape[0] # 这里要求P是一个numpy二维数组
b=P.shape[1]

# 计算P的每一个列向量的最小偏差α
alpha=functions.get_min_Standard_Deviation(P)

# 初始化第一个循环所需要的向量空间E_{a*c}，由于刚开始c=1，所以该空间即为f_1的单位矢量e_1
c=1
f_1=P[:,-1]
E_ac=functions.normalized_vector(f_1)

# 初始化结束后，计算T以及对应的f_i
T,f_i=functions.get_T_init(P,E_ac,f_1) # get_T()的第三个变量为此次要从P的候选向量当中删去的那个变量。

# 此时筛选出了一个f_i。加入到筛选结束后的数组中。
f_selected=[]
f_selected.append(f_1)
f_selected.append(f_i)

# 第一个循环
while T>0.1*abs(alpha):
    c=c+1
    E_ac=functions.make_new_space_in_cycle1(f_i,E_ac) # 考虑到c在不断+1，所以应该是这个空间在不断变大，而不是始终跟f_1组合成c=2的空间
    T,f_i=functions.get_T(P,E_ac,f_selected) # 筛选出新的f_i，并且需要避开已经加入空间的f
    if T>0.1*abs(alpha):
        f_selected.append(f_i)







# 第一个循环结束，接下来输出为YES
# epsilon的集合，一样去读excel文件
A=functions.load_excel_to_A()
f_selected_np=np.array(f_selected) # 保证为np数组
n=f_selected_np.shape[1] # 选中f的数量

# 计算P的每一个列向量的最小偏差α_dash
alpha_dash=functions.get_min_Standard_Deviation(A)

# 初始化结束后，先根据f_selected去找出P里的对应索引，再根据索引提取A中的对应列向量
index_set=functions.get_index_set(f_selected,P)
epsilon=[]
for index in index_set:
    epsilon.append(A[:,index])

epsilon=np.array(epsilon).T
# 初始化E_dn空间
E_dn=functions.normalized_vector(epsilon) # 重命名，逻辑更通顺

# 求第一个T'和第一个挑选出的向量的值
T_dash,epsilon_i=functions.get_T(A,E_dn,epsilon) # get_T()的第三个变量为此次要从P的候选向量当中删去的那个变量。

# 这是最终结果
result=epsilon



# 此时筛选出了一个ε_i。加入到筛选结束后的数组中。

result=np.c_[result,epsilon_i.T]

# 第二个循环
while T_dash>0.1*abs(alpha_dash):
    n=n+1
    E_dn=functions.make_new_space_in_cycle1(epsilon_i,E_dn) # 考虑到c在不断+1，所以应该是这个空间在不断变大，而不是始终跟f_1组合成c=2的空间
    T_dash,epsilon_i=functions.get_T(A,E_dn,epsilon_i) # 筛选出新的f_i
    if T_dash>0.1*abs(alpha_dash):
        result=np.c_[result,epsilon_i.T]


# 循环结束输出结果
print("您所需要的ε_i是：\n")
print(f"{result}")


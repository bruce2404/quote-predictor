import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

print("所有库已成功导入！")

# 加载数据
import pandas as pd

# 读取Excel文件
excel_file = pd.ExcelFile('4.0版本七个维度.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')

# 查看数据的基本信息
print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))

# 确定特征变量和目标变量
X = df[['交通方式','年龄分段', '性别', '城市人均购物金额', '小组人数','小组性质', '导游人均购物金额']]
y = df['消费金额']

# 对分类变量进行独热编码，不删除第一个类别
X = pd.get_dummies(X, columns=['交通方式','年龄分段', '性别', '小组性质'], drop_first=False)

print("编码后的特征变量：")
print(X.head())

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集样本数：{len(X_train)}")
print(f"测试集样本数：{len(X_test)}")

# 创建线性回归模型
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# 在训练集上训练模型
model.fit(X_train, y_train)

print("模型训练完成！")

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² 分数：{r2:.4f}')  # 保留4位小数
print(f'均方误差：{mse:.2f}')

# 设置 Pandas 显示选项
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽
pd.set_option('display.max_colwidth', None)  # 显示完整的列内容

# 查看模型系数
print("\n模型系数：")
coef_df = pd.DataFrame({
    '特征': X.columns,
    '系数': model.coef_
})
print(coef_df)
print(f"截距项：{model.intercept_:.2f}")

import joblib
joblib.dump(model, 'model.pkl')


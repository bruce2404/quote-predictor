import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入平均绝对误差函数

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

# 填充特殊值 -1
df['时间间隔（月份）'] = df.apply(lambda row: row['时间间隔（月份）'] if row['是否二次入境'] == '是' else -1, axis=1)

# 确定特征变量和目标变量
X = df[['导管','交通方式','是否二次入境','时间间隔（月份）','客户等级','年龄分段', '性别', '城市人均购物金额', '小组人数','小组性质', '导游人均购物金额']]
y = df['消费金额']

# 对分类变量进行独热编码，不删除第一个类别
X = pd.get_dummies(X, columns=['导管','交通方式','是否二次入境','客户等级','年龄分段', '性别', '小组性质'], drop_first=False)

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

print("线性回归模型训练完成！")

# 在测试集上进行线性回归预测
y_pred_lr = model.predict(X_test)

# 评估线性回归模型
r2_lr = r2_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)  # 计算线性回归平均绝对误差

print(f'线性回归 R² 分数：{r2_lr:.4f}')  # 保留4位小数
print(f'线性回归 均方误差：{mse_lr:.2f}')
print(f'线性回归 平均绝对误差：{mae_lr:.2f}')  # 输出线性回归平均绝对误差

# 创建随机森林回归模型
rf_model = RandomForestRegressor(random_state=42)

# 在训练集上训练随机森林模型
rf_model.fit(X_train, y_train)

print("随机森林模型训练完成！")

# 在测试集上进行随机森林预测
y_pred_rf = rf_model.predict(X_test)

# 评估随机森林模型
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)  # 计算随机森林平均绝对误差

print(f'随机森林 R² 分数：{r2_rf:.4f}')  # 保留4位小数
print(f'随机森林 均方误差：{mse_rf:.2f}')
print(f'随机森林 平均绝对误差：{mae_rf:.2f}')  # 输出随机森林平均绝对误差

# 创建梯度提升树回归模型
gb_model = GradientBoostingRegressor(random_state=42)

# 在训练集上训练梯度提升树模型
gb_model.fit(X_train, y_train)

print("梯度提升树模型训练完成！")

# 在测试集上进行梯度提升树预测
y_pred_gb = gb_model.predict(X_test)

# 评估梯度提升树模型
r2_gb = r2_score(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)  # 计算梯度提升树平均绝对误差

print(f'梯度提升树 R² 分数：{r2_gb:.4f}')  # 保留4位小数
print(f'梯度提升树 均方误差：{mse_gb:.2f}')
print(f'梯度提升树 平均绝对误差：{mae_gb:.2f}')  # 输出梯度提升树平均绝对误差

# 设置 Pandas 显示选项
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽
pd.set_option('display.max_colwidth', None)  # 显示完整的列内容

# 查看线性回归模型系数
print("\n线性回归模型系数：")
coef_df = pd.DataFrame({
    '特征': X.columns,
    '系数': model.coef_
})
print(coef_df)
print(f"线性回归截距项：{model.intercept_:.2f}")

# 查看随机森林模型特征重要性
print("\n随机森林模型特征重要性：")
feature_importance_df = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
})
print(feature_importance_df)

# 查看梯度提升树模型特征重要性
print("\n梯度提升树模型特征重要性：")
gb_feature_importance_df = pd.DataFrame({
    '特征': X.columns,
    '重要性': gb_model.feature_importances_
})
print(gb_feature_importance_df)

from sklearn.svm import SVR


# 在文件最后添加以下代码保存模型
import joblib
joblib.dump(rf_model, 'model.pkl')
print("随机森林模型已保存为model.pkl")
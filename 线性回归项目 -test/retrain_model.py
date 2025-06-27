import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

print("开始重新训练模型...")

# 读取Excel文件
excel_file = pd.ExcelFile('4.0版本七个维度.xlsx', engine='openpyxl')
df = excel_file.parse('Sheet1')

print(f"数据集大小：{df.shape}")

# 填充特殊值 -1
df['时间间隔（月份）'] = df.apply(lambda row: row['时间间隔（月份）'] if row['是否二次入境'] == '是' else -1, axis=1)

# 确定特征变量和目标变量
X = df[['导管','交通方式','是否二次入境','时间间隔（月份）','客户等级','年龄分段', '性别', '城市人均购物金额', '小组人数','小组性质', '导游人均购物金额']]
y = df['消费金额']

# 在这里删除包含NaN的行
print(f"特征选择后 - 删除前数据量：{len(X)}")
print(f"X中NaN数量：{X.isnull().sum().sum()}, y中NaN数量：{y.isnull().sum()}")

# 找出完整的行（X和y都没有NaN的行）
valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[valid_mask].reset_index(drop=True)
y = y[valid_mask].reset_index(drop=True)

print(f"删除后数据量：{len(X)}")
print(f"删除后 - X中NaN数量：{X.isnull().sum().sum()}, y中NaN数量：{y.isnull().sum()}")

# 对分类变量进行独热编码，不删除第一个类别
X = pd.get_dummies(X, columns=['导管','交通方式','是否二次入境','客户等级','年龄分段', '性别', '小组性质'], drop_first=False)

print(f"编码后特征数量：{X.shape[1]}")

# 编码后再次检查和清理数据
print(f"编码后 - X中NaN数量：{X.isnull().sum().sum()}, y中NaN数量：{y.isnull().sum()}")
print(f"编码后 - X中无穷大数量：{np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")

# 检查是否有无穷大值或NaN值
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("发现NaN值，进行最终清理...")
    valid_mask_final = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask_final].reset_index(drop=True)
    y = y[valid_mask_final].reset_index(drop=True)
    print(f"最终清理后数据量：{len(X)}")

# 检查无穷大值
numeric_cols = X.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    inf_mask = np.isinf(X[numeric_cols]).any(axis=1)
    if inf_mask.sum() > 0:
        print(f"发现{inf_mask.sum()}行包含无穷大值，进行清理...")
        X = X[~inf_mask].reset_index(drop=True)
        y = y[~inf_mask].reset_index(drop=True)
        print(f"清理无穷大值后数据量：{len(X)}")

# 最终数据验证
print(f"最终数据量：{len(X)}")
print(f"最终 - X中NaN数量：{X.isnull().sum().sum()}, y中NaN数量：{y.isnull().sum()}")
print(f"最终 - X中无穷大数量：{np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集样本数：{len(X_train)}")
print(f"测试集样本数：{len(X_test)}")

# 创建随机森林回归模型
rf_model = RandomForestRegressor(random_state=42)

# 在训练集上训练随机森林模型
rf_model.fit(X_train, y_train)

print("随机森林模型训练完成！")

# 在测试集上进行预测
y_pred_rf = rf_model.predict(X_test)

# 评估模型
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f'随机森林 R² 分数：{r2_rf:.4f}')
print(f'随机森林 均方误差：{mse_rf:.2f}')
print(f'随机森林 平均绝对误差：{mae_rf:.2f}')

# 保存兼容的模型
joblib.dump(rf_model, 'model.pkl', protocol=4)  # 使用协议4，兼容Python 3.6
print("\n兼容的随机森林模型已保存为 model.pkl")

# 保存特征名称，供app.py使用
feature_names = list(X.columns)
with open('feature_names.txt', 'w', encoding='utf-8') as f:
    for name in feature_names:
        f.write(name + '\n')
print("特征名称已保存为 feature_names.txt")

print("\n模型重新训练完成！现在可以运行 app.py 了。")
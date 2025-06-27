import pandas as pd
import joblib

# 加载模型和数据
model = joblib.load('model.pkl')
df = pd.read_excel('4.0版本七个维度.xlsx')

# 数据预处理
df['时间间隔（月份）'] = df.apply(lambda row: row['时间间隔（月份）'] if row['是否二次入境'] == '是' else -1, axis=1)

# 特征工程
X = pd.get_dummies(df, columns=['导管','交通方式','是否二次入境','客户等级','年龄分段', '性别', '小组性质'], drop_first=False)

# 确保包含所有特征
X['城市人均购物金额'] = df['城市人均购物金额']
X['导游人均购物金额'] = df['导游人均购物金额']
X['小组人数'] = df['小组人数']
X['时间间隔（月份）'] = df['时间间隔（月份）']

# 检查并添加缺失的特征列
missing_columns = set(model.feature_names_in_) - set(X.columns)
for col in missing_columns:
    X[col] = 0

# 确保特征顺序一致
X = X[model.feature_names_in_]

# 进行预测
df['预测消费金额'] = model.predict(X)

# 保存结果
df.to_excel('4.0版本七个维度_带预测结果.xlsx', index=False)
print("预测完成，结果已保存为'4.0版本七个维度_带预测结果.xlsx'")
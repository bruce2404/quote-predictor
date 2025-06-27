# 游客消费预测系统

## 项目简介

这是一个基于机器学习的游客消费预测系统，使用随机森林回归模型来预测游客的消费金额。系统提供了Web界面供用户输入游客信息并获得消费预测结果。

## 功能特点

- 🎯 **智能预测**：基于多维度特征预测游客消费金额
- 🌐 **Web界面**：友好的用户界面，支持移动端访问
- 🔄 **模型重训练**：支持使用新数据重新训练模型
- 📊 **批量预测**：支持对现有数据进行批量预测
- 🐳 **Docker支持**：提供Docker容器化部署
- 📱 **身份证解析**：自动解析身份证信息获取年龄、性别等特征

## 项目结构

```
线性回归项目-test/
├── app.py                          # Flask Web应用主文件
├── model.pkl                       # 训练好的随机森林模型
├── retrain_model.py               # 模型重训练脚本
├── predict_existing_data.py       # 批量预测脚本
├── requirements.txt               # Python依赖包列表
├── Dockerfile                     # Docker容器配置文件
├── feature_names.txt              # 模型特征名称列表
├── model_coefficients.csv         # 模型系数信息
├── 4.0版本七个维度.xlsx            # 训练数据集
├── 4.0版本七个维度_带预测结果.xlsx  # 预测结果文件
├── 城市数据.txt                   # 城市信息数据
├── 客户等级.txt                   # 客户等级数据
├── templates/                     # HTML模板文件夹
│   ├── form.html                  # 输入表单页面
│   └── result.html                # 结果显示页面
└── README.md                      # 项目说明文档
```

## 技术栈

- **后端框架**: Flask 3.1.0
- **机器学习**: scikit-learn 1.6.1, RandomForestRegressor
- **数据处理**: pandas 2.2.3, numpy 2.2.5
- **模型持久化**: joblib 1.5.0
- **前端**: HTML5, CSS3, JavaScript
- **容器化**: Docker
- **Python版本**: 3.11+

## 模型特征

系统使用以下特征进行预测：

### 基础特征
- **时间间隔（月份）**: 二次入境的时间间隔
- **城市人均购物金额**: 游客所在城市的人均购物消费水平
- **小组人数**: 旅游团队规模
- **导游人均购物金额**: 导游的历史购物引导能力

### 分类特征
- **导管**: 负责导游
- **交通方式**: 到达方式（动车、火车、自驾、飞机）
- **是否二次入境**: 是否为回头客（是/否）
- **客户等级**: 客户分级（S级、A级、B级、C级、D级）
- **年龄分段**: 年龄区间（0-20、20-30、30-45、45-65、>65）
- **性别**: 男/女
- **小组性质**: 团队类型

## 安装说明

### 方式一：本地安装

1. **克隆项目**
```bash
git clone <repository-url>
cd 线性回归项目-test
```

2. **创建虚拟环境**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **运行应用**
```bash
python app.py
```

5. **访问应用**
打开浏览器访问：http://localhost:5000

### 方式二：Docker部署

1. **构建镜像**
```bash
docker build -t tourist-prediction .
```

2. **运行容器**
```bash
docker run -p 5000:5000 tourist-prediction
```

3. **访问应用**
打开浏览器访问：http://localhost:5000

## 使用说明

### Web界面使用

1. **访问首页**: 打开浏览器访问系统地址
2. **填写信息**: 在表单中输入游客相关信息
   - 身份证号码（系统会自动解析年龄、性别、城市信息）
   - 导管选择
   - 交通方式
   - 是否二次入境
   - 客户等级
   - 小组信息等
3. **获取预测**: 点击预测按钮获取消费金额预测结果
4. **查看详情**: 系统会显示详细的预测信息和置信度

### 模型重训练

当有新的训练数据时，可以重新训练模型：

```bash
python retrain_model.py
```

**注意事项**：
- 确保新数据格式与原始数据一致
- 训练数据应保存为 `4.0版本七个维度.xlsx`
- 重训练会覆盖现有的 `model.pkl` 文件

### 批量预测

对现有数据进行批量预测：

```bash
python predict_existing_data.py
```

结果将保存为 `4.0版本七个维度_带预测结果.xlsx`

## API文档

### 主要路由

#### GET/POST /
- **描述**: 主页面，显示预测表单和处理预测请求
- **方法**: GET（显示表单）, POST（处理预测）
- **参数**: 
  - `id_card`: 身份证号码
  - `guide`: 导管
  - `transport`: 交通方式
  - `is_return`: 是否二次入境
  - `time_interval`: 时间间隔（仅二次入境时需要）
  - `customer_level`: 客户等级
  - `group_size`: 小组人数
  - `group_type`: 小组性质
  - `guide_avg_shopping`: 导游人均购物金额

### 核心函数

#### `parse_id_card(id_card)`
解析身份证信息获取年龄、性别、城市等特征

**参数**:
- `id_card` (str): 18位身份证号码

**返回**:
```python
{
    'age': int,           # 年龄
    'age_group': str,     # 年龄分段
    'gender': str,        # 性别
    'city_info': dict     # 城市信息
}
```

## 数据说明

### 训练数据格式

训练数据应包含以下列：
- 导管
- 交通方式
- 是否二次入境
- 时间间隔（月份）
- 客户等级
- 年龄分段
- 性别
- 城市人均购物金额
- 小组人数
- 小组性质
- 导游人均购物金额
- 消费金额（目标变量）

### 城市数据格式

`城市数据.txt` 文件格式：
```
行政代码    省份    城市    城市人均购物金额    最高报价
1101       北京    北京市   2500              5000
...
```

## 模型性能

当前模型使用随机森林回归算法，具有以下特点：
- **算法**: RandomForestRegressor
- **特征工程**: 独热编码处理分类变量
- **数据预处理**: 自动处理缺失值和异常值
- **评估指标**: R²、MSE、MAE

## 开发说明

### 项目依赖

主要依赖包：
- Flask: Web框架
- pandas: 数据处理
- scikit-learn: 机器学习
- joblib: 模型序列化
- numpy: 数值计算

### 代码结构

- `app.py`: Flask应用主文件，包含路由和预测逻辑
- `retrain_model.py`: 模型训练脚本
- `predict_existing_data.py`: 批量预测脚本
- `templates/`: HTML模板文件

### 扩展开发

1. **添加新特征**: 在模型训练脚本中添加新的特征列
2. **优化算法**: 可尝试其他回归算法（XGBoost、LightGBM等）
3. **增强UI**: 改进前端界面和用户体验
4. **API扩展**: 添加RESTful API接口

## 故障排除

### 常见问题

1. **模型文件缺失**
   - 确保 `model.pkl` 文件存在
   - 如不存在，运行 `retrain_model.py` 重新训练

2. **依赖包版本冲突**
   - 使用虚拟环境隔离依赖
   - 严格按照 `requirements.txt` 安装

3. **数据格式错误**
   - 检查输入数据格式是否与训练数据一致
   - 确保身份证号码为18位有效格式

4. **端口占用**
   - 修改 `app.py` 中的端口号
   - 或终止占用端口的进程

### 日志调试

应用运行时会输出详细日志，包括：
- 模型加载状态
- 预测请求信息
- 错误信息和堆栈跟踪

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至：[zhangshuigao310@gmail.com]

---

**注意**: 本系统仅供学习和研究使用，实际应用时请根据具体业务需求进行调整和优化。
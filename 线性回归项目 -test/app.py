from flask import Flask, request, render_template
import pandas as pd
import joblib
import re
# 原导入语句
# import datetime
# 修改为导入 datetime 类
from datetime import datetime

app = Flask(__name__)

# 加载模型
model = joblib.load('model.pkl')

# 加载城市数据
city_data = pd.read_csv('城市数据.txt', sep='\t')
# 去除包含任何 NaN 值的行
city_data = city_data.dropna()
city_dict = dict(zip(city_data['行政代码'].astype(str), city_data[['省份', '城市', '城市人均购物金额', '最高报价']].to_dict('records')))

def parse_id_card(id_card):
    """解析身份证信息"""
    # 提取出生日期
    birth_date = datetime.strptime(id_card[6:14], '%Y%m%d')
    age = (datetime.now() - birth_date).days // 365
    
    # 年龄分段
    if age <= 20: age_group = '0-20'
    elif 20 < age <= 30: age_group = '20-30'
    elif 30 < age <= 45: age_group = '30-45'
    elif 45 < age <= 65: age_group = '45-65'
    else: age_group = '>65'
    
    # 性别
    gender = '男' if int(id_card[16]) % 2 else '女'
    
    # 城市信息
    city_code = id_card[:4]
    city_info = city_dict.get(city_code, {'省份':'未知','城市':'未知','城市人均购物金额':1000, '最高报价': 0})
    
    return {
        'age': age,
        'age_group': age_group,
        'gender': gender,
        'city_info': city_info
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    # 读取客户等级文件
    companies = []
    with open('客户等级.txt', 'r', encoding='utf-8') as file:
        next(file)  # 跳过标题行
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                companies.append(parts)

    if request.method == 'POST':
        # 获取表单数据，修改为获取客户公司名和客户等级
        customer_company = request.form.get('客户公司名', '')  # 允许为空
        customer_level = request.form.get('客户等级', 'A')  # 默认为A级
        
        # 如果客户公司名为空或客户等级为空，设置默认客户等级为A
        if not customer_company.strip() or not customer_level.strip():
            customer_level = 'A'
        
        # 只检查客户等级是否存在，不再检查客户公司名
        if not customer_level:
            return "缺少客户等级字段", 400
        
        # 获取身份证列表和个人特征
        id_cards = request.form.getlist('idCards[]')
        is_returns = request.form.getlist('is_return[]')
        time_intervals = request.form.getlist('time_interval[]')
        customer_levels = request.form.getlist('customer_level[]')
        
        # 过滤掉空的身份证输入框
        filtered_data = []
        for i, card in enumerate(id_cards):
            if card and card.strip():  # 只处理非空的身份证
                filtered_data.append({
                    'card': card.strip(),
                    'is_return': is_returns[i] if i < len(is_returns) else '否',
                    'time_interval': time_intervals[i] if i < len(time_intervals) else ''
                })
        
        # 检查是否至少有一个有效身份证
        if not filtered_data:
            return "请至少输入一个有效的身份证号码", 400
        
        # 解析每个身份证信息并获取年龄列表
        ages = []
        male_count = 0
        female_count = 0
        predictions = []
        for data in filtered_data:
            member = parse_id_card(data['card'])
            ages.append(member['age'])
            if member['gender'] == '男':
                male_count += 1
            else:
                female_count += 1
            member.update({
                'id_card': data['card'],
                'is_return': data['is_return'],
                'time_interval': int(data['time_interval']) if data['is_return'] == "是" and data['time_interval'] else -1,
            })
            predictions.append(member)
        
        group_size = len(filtered_data)  # 使用过滤后的数据长度
        age_diff = max(ages) - min(ages)
        
        # 根据公式判断小组性质
        if group_size == 1:
            group_nature = "单人"
        elif group_size == 2:
            if male_count == 1 and female_count == 1 and age_diff <= 20:
                group_nature = "夫妻"
            elif female_count == 2 and age_diff <= 20:
                group_nature = "两女"
            elif age_diff > 20:
                group_nature = "亲子"
            else:
                group_nature = ""
        elif 3 <= group_size <= 6:
            if age_diff > 20:
                group_nature = "家庭"
            else:
                group_nature = "同龄朋友"
        elif group_size >= 7:
            group_nature = "多人团"
        else:
            group_nature = ""
        
        # 从身份证提取框输入的信息中识别交通方式
        id_card_inputs = ' '.join(id_cards)  # 假设所有输入信息拼接在一起
        recognized_transportation = "飞机"
        if "动车" in id_card_inputs:
            recognized_transportation = "动车"
        elif "火车" in id_card_inputs:
            recognized_transportation = "火车"
        elif "自驾" in id_card_inputs:
            recognized_transportation = "自驾"

        # 解析每个身份证信息并进行预测
        for i, member in enumerate(predictions):
            # 判断是否为非正常人群（年龄≤25或≥69）
            is_abnormal_age = member['age'] <= 25 or member['age'] >= 69
            member['is_abnormal_age'] = is_abnormal_age
            
            # 如果是非正常人群，跳过预测计算
            if is_abnormal_age:
                member['final_quote'] = 0  # 设置为0，前端将显示"非正常人群"
                continue
                
            # 首先定义features字典（仅对正常年龄人群）
            features = {
                '时间间隔（月份）': member['time_interval'],
                '城市人均购物金额': member['city_info']['城市人均购物金额'],
                '导游人均购物金额': 2500,
                '小组人数': group_size,
                f'是否二次入境_{member["is_return"]}': 1,
                f'客户等级_{customer_level}': 1,
                f'交通方式_{recognized_transportation}': 1,
                f'导管_刘波': 1,
                f'年龄分段_{member["age_group"]}': 1,
                f'性别_{member["gender"]}': 1,
                f'小组性质_{group_nature}': 1
            }
            
            # 读取特征名称 - 添加encoding参数
            with open('feature_names.txt', 'r', encoding='utf-8') as f:
                feature_names_list = [line.strip() for line in f]
            
            # 创建特征DataFrame
            input_df = pd.DataFrame(columns=feature_names_list)
            input_df.loc[0] = 0
            
            # 设置特征值
            for feature in feature_names_list:
                if feature in features:
                    input_df[feature] = features[feature]
                else:
                    input_df[feature] = 0
            
            # 确保特征顺序与模型一致
            for feature in model.feature_names_in_:
                if feature in features:
                    input_df[feature] = features[feature]
                else:
                    input_df[feature] = 0
            
            # 进行预测
            member['prediction'] = model.predict(input_df)[0]

            # 计算预测报价
            prediction_quote = member['prediction'] * 0.68 - 200

            # 获取城市信息中的最高报价
            city_code = member['id_card'][:4]
            city_info = city_dict.get(city_code, {'省份':'未知','城市':'未知','城市人均购物金额':1000, '最高报价': 0})
            max_quote = city_info.get('最高报价', 0)

            # 比较预测报价和最高报价
            if prediction_quote > max_quote:
                prediction_quote = max_quote

            # 确保 season 变量在使用前被正确赋值
            # 获取当前日期
            today = datetime.today().date()
            month = today.month
            day = today.day
            
            # 判断淡旺季
            season = '平季'
            if (
                (month == 3 and 6 <= day <= 31) or
                (month == 5 and 20 <= day <= 31) or
                (month == 6 and 1 <= day <= 28) or
                (month == 8 and 20 <= day <= 31) or
                (month == 9 and 1 <= day <= 10) or
                (month == 10 and 9 <= day <= 15) or
                (month == 11 and 20 <= day <= 30) or
                (month == 12) or
                (month == 1) or
                (month == 2 and 1 <= day <= 14)
            ):
                season = '淡季'
            elif (
                (month == 4 and 20 <= day <= 30) or
                (month == 5 and 1 <= day <= 19) or
                (month == 6 and 29 <= day <= 30) or
                (month == 7) or
                (month == 8 and 1 <= day <= 19) or
                (month == 10 and 1 <= day <= 8) or
                (month == 2 and 15 <= day <= 28)
            ):
                season = '旺季'

            # 根据淡旺季调整报价，删除重复的调整逻辑
            if season == '淡季':
                prediction_quote += 300
            elif season == '旺季':
                prediction_quote -= 200

            member['final_quote'] = prediction_quote
        
        return render_template('result.html', 
                            predictions=predictions,
                            members=predictions,
                            group_data=request.form,
                            season=season)
    
    return render_template('form.html', companies=companies)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

# 获取训练时的特征名称（假设X_train是原始训练数据）
feature_names = X_train.columns.tolist()

# 保存特征名称到文件（在retrain_model.py中添加）
with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))
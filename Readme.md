# Enefit

## 环境准备
* 添加本地python环境到jupyter： python -m ipykernel install --user --name enefit --display-name "enefit"
* git提交前提：git config --global http.sslVerify "false"

## 开发说明
* 关于天气数据，仅仅是对列值取均值是不够的，高温，潮湿，温差过大都会降低发电效率
* TODO： （当前温度-露点温度）应该作为一个特征，因为当温度低于露点时将意味着发电能力下降
* temperature:温度，dewpoint:露点温度，rain:毫米降雨量 snowfall:厘米降雪量 surface_pressure：大气压
* cloudcover_:中高低空云层遮盖率，windspeed_10m：10米高空风速
* 更多电池板发电影响因素的信息：https://www.75xn.com/26488.html
* 关于时间信息，建议将上一个周期的数据作为特征传入当前预测，例如上周今日，昨天此时的值作为参考值送入模型作为特征
* electricity_price数据中仅一个euros_per_mwh字段是有用的，建议将上一个小时或者过去三小时或者昨天此时的电价分别作为一个特征用于预测
* gas_price同样的处理逻辑，对于price_per_mwh，需要考虑上一个小时或者过去三小时或者昨天此时的燃气价格

## 2023年12月19日
* 复现了notebook中的代码，实现了实验的正常运行，并为程序以及xgboost增加了输出到文件的日志记录能力，Epoch: 700, validation_1: mae: 73.80073650056661
* 特征重要度: 'county' 'highest_price_per_mwh_gas' 'day_of_month'
 'mean_price_per_mwh_gas' 'temperature_h_mean' 'prediction_unit_id' 'year'
 'target_11_days_ago' 'cloudcover_low_h_mean'
 '10_metre_u_wind_component_f_mean' 'cloudcover_total_f_mean'
 'dewpoint_h_mean' 'target_10_days_ago' 'euros_per_mwh_electricity'
 '10_metre_v_wind_component_f_mean' 'surface_pressure_h_mean'
 'cloudcover_mid_f_mean' 'cloudcover_total_h_mean'
 'winddirection_10m_h_mean' 'cloudcover_mid_h_mean' 'rain_h_mean'
 'cloudcover_high_f_mean' 'windspeed_10m_h_mean' 'cloudcover_high_h_mean'
 'snowfall_h_mean'

## 2023年12月3日
* 完成git本地配置以及代码初始化
# -*- coding:utf-8 -*-
"""
-----------------------------------
    Author      :    Conley.Kong
    Create Date :    2023/12/17
    Description :    文件描述信息
--------------------------------------------
    Change Activity: 
        2023/12/17 :    创建并初始化本文件        
"""
import os
import sys

if "win" in sys.platform:
    USING_LINUX = False
    basedir = os.path.abspath(os.path.dirname(__file__) + "/").replace('\\', '/')
else:
    USING_LINUX = True
    basedir = os.path.abspath(os.getcwd() + "/")
if basedir not in sys.path:
    sys.path.append(basedir)
    print(f">>>> {os.path.basename(__file__)} appended {basedir} into system path")
import numpy as np
import pandas as pd
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
# Geolocation
from geopy.geocoders import Nominatim
# Options
pd.set_option('display.max_columns', 100)
# Modeling
import xgboost as xgb
import torch


class FeatureProcessor():
    def __init__(self):
        # Columns to join on for the different datasets
        self.weather_join = ['datetime', 'county', 'data_block_id']
        self.gas_join = ['data_block_id']
        self.electricity_join = ['datetime', 'data_block_id']
        self.client_join = ['county', 'is_business', 'product_type', 'data_block_id']
        
        # Columns of latitude & longitude
        self.lat_lon_columns = ['latitude', 'longitude']
        
        # Aggregate stats
        self.agg_stats = ['mean']  # , 'min', 'max', 'std', 'median']
        
        # Categorical columns (specify for XGBoost)
        self.category_columns = ['county', 'is_business', 'product_type', 'is_consumption', 'data_block_id']
        
        # Location from https://www.kaggle.com/datasets/michaelo/fabiendaniels-mapping-locations-and-county-codes/data
        self.location = (pd.read_csv("/kaggle/input/fabiendaniels-mapping-locations-and-county-codes/county_lon_lats.csv")
                    .drop(columns=["Unnamed: 0"])
                    )
    
    def create_new_column_names(self, df, suffix, columns_no_change):
        '''Change column names by given suffix, keep columns_no_change, and return back the data'''
        df.columns = [col
                      if col in columns_no_change
                      else col + suffix
                      for col in df.columns
                      ]
        return df
    
    def flatten_multi_index_columns(self, df):
        df.columns = ['_'.join([col for col in multi_col if len(col) > 0])
                      for multi_col in df.columns]
        return df
    
    def create_data_features(self, data):
        '''📊Create features for main data (test or train) set📊'''
        # To datetime
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # todo: 当前周期对应上个周期（上周今日，昨天此时）的值作为一个特征（可以躲避对于日照时长的计算）
        
        # Time period features
        data['date'] = data['datetime'].dt.normalize()
        data['year'] = data['datetime'].dt.year
        data['quarter'] = data['datetime'].dt.quarter
        data['month'] = data['datetime'].dt.month
        data['week'] = data['datetime'].dt.isocalendar().week
        data['hour'] = data['datetime'].dt.hour
        
        # Day features
        data['day_of_year'] = data['datetime'].dt.day_of_year
        data['day_of_month'] = data['datetime'].dt.day
        data['day_of_week'] = data['datetime'].dt.day_of_week
        return data
    
    def create_client_features(self, client):
        '''💼 Create client features 💼'''
        # Modify column names - specify suffix
        client = self.create_new_column_names(client,
                                              suffix='_client',
                                              columns_no_change=self.client_join
                                              )
        return client
    
    def create_historical_weather_features(self, historical_weather):
        '''⌛🌤️ Create historical weather features 🌤️⌛'''
        
        # To datetime
        historical_weather['datetime'] = pd.to_datetime(historical_weather['datetime'])
        
        # Add county
        historical_weather[self.lat_lon_columns] = historical_weather[self.lat_lon_columns].astype(float).round(1)
        historical_weather = historical_weather.merge(self.location, how='left', on=self.lat_lon_columns)
        
        # Modify column names - specify suffix
        historical_weather = self.create_new_column_names(historical_weather,
                                                          suffix='_h',
                                                          columns_no_change=self.lat_lon_columns + self.weather_join
                                                          )
        
        # Group by & calculate aggregate stats
        agg_columns = [col for col in historical_weather.columns if col not in self.lat_lon_columns + self.weather_join]
        # TODO：仅仅是对列值取均值是不够的，高温，潮湿，温差过大都会降低发电效率
        #  TODO： （当前温度-露点温度）应该作为一个特征，因为当温度低于露点时将意味着发电能力下降
        # temperature:温度，dewpoint:露点温度，rain:毫米降雨量 snowfall:厘米降雪量 surface_pressure：大气压
        # cloudcover_:中高低空云层遮盖率，windspeed_10m：10米高空风速 shortwave_radiation：短波辐射量（wh/m2）
        # direct_solar_radiation: 直射辐射值，diffuse_radiation：散射辐射值，latitude/longitude经纬度，data_block_id：数据id
        # 更多电池板发电影响因素的信息：https://www.75xn.com/26488.html
        agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
        historical_weather = historical_weather.groupby(self.weather_join).agg(agg_dict).reset_index()
        
        # Flatten the multi column aggregates
        historical_weather = self.flatten_multi_index_columns(historical_weather)
        
        # Test set has 1 day offset for hour<11 and 2 day offset for hour>11
        historical_weather['hour_h'] = historical_weather['datetime'].dt.hour
        historical_weather['datetime'] = (historical_weather
                                          .apply(lambda x:
                                                 x['datetime'] + pd.DateOffset(1)
                                                 if x['hour_h'] < 11
                                                 else x['datetime'] + pd.DateOffset(2),
                                                 axis=1)
                                          )
        
        return historical_weather
    
    def create_forecast_weather_features(self, forecast_weather):
        '''🔮🌤️ Create forecast weather features 🌤️🔮'''
        
        # Rename column and drop
        forecast_weather = (forecast_weather
                            .rename(columns={'forecast_datetime': 'datetime'})
                            .drop(columns='origin_datetime')  # not needed
                            )
        
        # To datetime
        forecast_weather['datetime'] = (pd.to_datetime(forecast_weather['datetime'])
                                        .dt
                                        .tz_convert('Europe/Brussels')  # change to different time zone?
                                        .dt
                                        .tz_localize(None)
                                        )
        
        # Add county
        forecast_weather[self.lat_lon_columns] = forecast_weather[self.lat_lon_columns].astype(float).round(1)
        forecast_weather = forecast_weather.merge(self.location, how='left', on=self.lat_lon_columns)
        
        # Modify column names - specify suffix
        forecast_weather = self.create_new_column_names(forecast_weather,
                                                        suffix='_f',
                                                        columns_no_change=self.lat_lon_columns + self.weather_join
                                                        )
        
        # Group by & calculate aggregate stats
        agg_columns = [col for col in forecast_weather.columns if col not in self.lat_lon_columns + self.weather_join]
        agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
        forecast_weather = forecast_weather.groupby(self.weather_join).agg(agg_dict).reset_index()
        
        # Flatten the multi column aggregates
        forecast_weather = self.flatten_multi_index_columns(forecast_weather)
        return forecast_weather
    
    def create_electricity_features(self, electricity):
        '''⚡ Create electricity prices features ⚡'''
        # To datetime
        electricity['forecast_date'] = pd.to_datetime(electricity['forecast_date'])
        
        # Test set has 1 day offset
        electricity['datetime'] = electricity['forecast_date'] + pd.DateOffset(1)
        
        # 只有一个euros_per_mwh电价指标是有用的
        # todo 建议将上一个小时或者过去三小时或者昨天此时的电价分别作为一个特征用于预测
        # Modify column names - specify suffix
        electricity = self.create_new_column_names(electricity,
                                                   suffix='_electricity',
                                                   columns_no_change=self.electricity_join
                                                   )
        return electricity
    
    def create_gas_features(self, gas):
        '''⛽ Create gas prices features ⛽'''
        # Mean gas price
        gas['mean_price_per_mwh'] = (gas['lowest_price_per_mwh'] + gas['highest_price_per_mwh']) / 2
        #todo 需要考虑上一个小时或者过去三小时或者昨天此时的燃气价格
        # Modify column names - specify suffix
        gas = self.create_new_column_names(gas,
                                           suffix='_gas',
                                           columns_no_change=self.gas_join
                                           )
        return gas
    
    def __call__(self, data, client, historical_weather, forecast_weather, electricity, gas):
        '''Processing of features from all datasets, merge together and return features for dataframe df '''
        # Create features for relevant dataset
        data = self.create_data_features(data)
        client = self.create_client_features(client)
        historical_weather = self.create_historical_weather_features(historical_weather)
        forecast_weather = self.create_forecast_weather_features(forecast_weather)
        electricity = self.create_electricity_features(electricity)
        gas = self.create_gas_features(gas)
        
        # 🔗 Merge all datasets into one df 🔗
        df = data.merge(client, how='left', on=self.client_join)
        df = df.merge(historical_weather, how='left', on=self.weather_join)
        df = df.merge(forecast_weather, how='left', on=self.weather_join)
        df = df.merge(electricity, how='left', on=self.electricity_join)
        df = df.merge(gas, how='left', on=self.gas_join)
        
        # Change columns to categorical for XGBoost
        df[self.category_columns] = df[self.category_columns].astype('category')
        return df

class MyPredictor:
    
    def __init__(self):
        self.n_day_lags = 15  # Specify how many days we want to go back (at least 2)
        self.DEBUG = False  # False/True
        # GPU or CPU use for model
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.cls = None
        self.target = None
        self.features = None
    
    def preprocess(self,x):
        pass

    def create_revealed_targets_train(self, data, n_day_lags):
        '''🎯 Create past revealed_targets for train set based on number of day lags N_day_lags 🎯 '''
        original_datetime = data['datetime']
        revealed_targets = data[['datetime', 'prediction_unit_id', 'is_consumption', 'target']].copy()
    
        # Create revealed targets for all day lags
        for day_lag in range(2, n_day_lags + 1):
            revealed_targets['datetime'] = original_datetime + pd.DateOffset(day_lag)
            data = data.merge(revealed_targets,
                              how='left',
                              on=['datetime', 'prediction_unit_id', 'is_consumption'],
                              suffixes=('', f'_{day_lag}_days_ago')
                              )
        return data
    
    def train(self):
        DATA_DIR = "./data/predict-energy-behavior-of-prosumers/"
    
        # Read CSVs and parse relevant date columns
        train = pd.read_csv(DATA_DIR + "train.csv")
        client = pd.read_csv(DATA_DIR + "client.csv")
        historical_weather = pd.read_csv(DATA_DIR + "historical_weather.csv")
        forecast_weather = pd.read_csv(DATA_DIR + "forecast_weather.csv")
        electricity = pd.read_csv(DATA_DIR + "electricity_prices.csv")
        gas = pd.read_csv(DATA_DIR + "gas_prices.csv")
    
        # Create all features
        feature_processor = FeatureProcessor()
        data = feature_processor(data=train.copy(),
                                 client=client.copy(),
                                 historical_weather=historical_weather.copy(),
                                 forecast_weather=forecast_weather.copy(),
                                 electricity=electricity.copy(),
                                 gas=gas.copy(),
                                 )
    
        df = self.create_revealed_targets_train(data.copy(),
                                                n_day_lags=self.n_day_lags)
        # Remove empty target row
        # Remove columns for features
        self.target = 'target'
        df = df[df[self.target].notnull()].reset_index(drop=True)
        no_features = ['date',
                       'latitude',
                       'longitude',
                       'data_block_id',
                       'row_id',
                       'hours_ahead',
                       'hour_h',
                       ]
        remove_columns = [col for col in df.columns for no_feature in no_features if no_feature in col]
        remove_columns.append(self.target)
        self.features = [col for col in df.columns if col not in remove_columns]
        
        #### Create single fold split ######
        train_block_id = list(range(0, 600))
        tr = df[df['data_block_id'].isin(train_block_id)]  # first 600 data_block_ids used for training
        val = df[~df['data_block_id'].isin(train_block_id)]  # rest data_block_ids used for validation

        self.clf = xgb.XGBRegressor(
            device=self.device,
            enable_categorical=True,
            objective='reg:absoluteerror',
            n_estimators=2 if self.DEBUG else 1500,
            early_stopping_rounds=100
        )
        self.clf.fit(X=tr[self.features],
                y=tr[self.target],
                eval_set=[(tr[self.features], tr[self.target]), (val[self.features], val[self.target])],
                verbose=True  # False #True
                )
        import pickle
        pickle.dump(self.clf,open(f"./model.pickle",'wb+'))

        # Plot RMSE
        import matplotlib.pyplot as plt
        results = self.clf.evals_result()
        train_mae, val_mae = results["validation_0"]["mae"], results["validation_1"]["mae"]
        x_values = range(0, len(train_mae))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_values, train_mae, label="Train MAE")
        ax.plot(x_values, val_mae, label="Validation MAE")
        ax.legend()
        plt.ylabel("MAE Loss")
        plt.title("XGBoost MAE Loss")
        plt.savefig('./loss.jpg')
        plt.show()
        
        # 绘制特征重要度
        TOP = 20
        importance_data = pd.DataFrame({'name': self.clf.feature_names_in_,
                                        'importance': self.clf.feature_importances_})
        importance_data = importance_data.sort_values(by='importance', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=importance_data[:TOP],
                    x='importance',
                    y='name'
                    )
        patches = ax.patches
        count = 0
        for patch in patches:
            height = patch.get_height()
            width = patch.get_width()
            perc = 100 * importance_data['importance'].iloc[count]  # 100*width/len(importance_data)
            ax.text(width, patch.get_y() + height / 2, f'{perc:.1f}%')
            count += 1

        plt.title(f'The top {TOP} features sorted by importance')
        plt.savefig('./importance.jpg')
        plt.show()

        not_important_feats = importance_data[importance_data['importance'] < 0.0005].name.values
        print(not_important_feats)

    def create_revealed_targets_test(self, data, previous_revealed_targets, n_day_lags):
        '''🎯 Create new test data based on previous_revealed_targets and N_day_lags 🎯 '''
        for count, revealed_targets in enumerate(previous_revealed_targets):
            day_lag = count + 2
        
            # Get hour
            revealed_targets['hour'] = pd.to_datetime(revealed_targets['datetime']).dt.hour
        
            # Select columns and rename target
            revealed_targets = revealed_targets[['hour', 'prediction_unit_id', 'is_consumption', 'target']]
            revealed_targets = revealed_targets.rename(columns={"target": f"target_{day_lag}_days_ago"})
        
            # Add past revealed targets
            data = pd.merge(data,
                            revealed_targets,
                            how='left',
                            on=['hour', 'prediction_unit_id', 'is_consumption'],
                            )
    
        # If revealed_target_columns not available, replace by nan
        all_revealed_columns = [f"target_{day_lag}_days_ago" for day_lag in range(2, n_day_lags + 1)]
        missing_columns = list(set(all_revealed_columns) - set(data.columns))
        data[missing_columns] = np.nan
    
        return data

    def make_submit(self):
        import enefit
        env = enefit.make_env()
        iter_test = env.iter_test()
        # Reload enefit environment (only in debug mode, otherwise the submission will fail)
        if self.DEBUG:
            enefit.make_env.__called__ = False
            type(env)._state = type(type(env)._state).__dict__['INIT']
            iter_test = env.iter_test()
        # %%
        # List of target_revealed dataframes
        previous_revealed_targets = []
    
        for (test,
             revealed_targets,
             client_test,
             historical_weather_test,
             forecast_weather_test,
             electricity_test,
             gas_test,
             sample_prediction) in iter_test:
        
            # Rename test set to make consistent with train
            test = test.rename(columns={'prediction_datetime': 'datetime'})
        
            # Initiate column data_block_id with default value to join on
            id_column = 'data_block_id'
        
            test[id_column] = 0
            gas_test[id_column] = 0
            electricity_test[id_column] = 0
            historical_weather_test[id_column] = 0
            forecast_weather_test[id_column] = 0
            client_test[id_column] = 0
            revealed_targets[id_column] = 0
        
            data_test = FeatureProcessor(
                data=test,
                client=client_test,
                historical_weather=historical_weather_test,
                forecast_weather=forecast_weather_test,
                electricity=electricity_test,
                gas=gas_test
            )
        
            # Store revealed_targets
            previous_revealed_targets.insert(0, revealed_targets)
        
            if len(previous_revealed_targets) == self.n_day_lags:
                previous_revealed_targets.pop()
        
            # Add previous revealed targets
            df_test = self.create_revealed_targets_test(data=data_test.copy(),
                                                        previous_revealed_targets=previous_revealed_targets.copy(),
                                                        n_day_lags=self.n_day_lags
                                                        )
        
            # Make prediction
            X_test = df_test[self.features]
            sample_prediction['target'] = self.clf.predict(X_test)
            env.predict(sample_prediction)
    
    def predict(self):
        pass


def run():
    my_model = MyPredictor()
    my_model.train()


if __name__ == "__main__":
    run()

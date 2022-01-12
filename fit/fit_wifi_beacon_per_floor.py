import optuna.integration.lightgbm as op_lgb
import lightgbm as lgb
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os
from pathlib import Path
import re

def bname(object_name):
    return os.path.basename(object_name.rstrip(os.sep))

def save_model(facility_name,floor_name,model_name,model):
    DIR = 'models_wifi_per_floor'
    path = f'./{DIR}/{facility_name}'
    if not os.path.exists(path):
        os.mkdir(path)
    path = f'./{DIR}/{facility_name}/{floor_name}'
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}/{model_name}.pickle', mode='wb') as f:
        pickle.dump(model, f)
    print(f'{model_name}モデルを保存')

if __name__== "__main__":
    DIR = './got_data_per_floor/*'
    DIR = './got_data_per_floor_wifi_only/*'
    got_data_folders = glob.glob(DIR)
    skip = True
    # 各施設 を順に走査
    for i,facility_path in enumerate(got_data_folders):
        print(f'{i+1}施設目 {bname(facility_path)}')
        floor_csv_paths = glob.glob(f'{facility_path}/*')
        for floor_csv_path in floor_csv_paths:
            floor_num = Path(floor_csv_path).parts[-1][:-len('.csv')]
            print(f'{floor_num} を探索')
            if bname(facility_path) == '5da138b74db8ce0c98bd4774':
                skip = False
                continue
            if skip:
                continue
            facility_data = pd.read_csv(floor_csv_path).iloc[:,1:]
            # trainとtestで分ける
            train, test = train_test_split(facility_data, test_size=0.2,random_state=42)
            X_train_data = train.drop(['x','y','floor'],axis=1)
            X_test_data = test.drop(['x','y','floor'],axis=1)

            # 訓練用正解ラベル
            t_train_x = train['x']
            t_train_y = train['y']
            # 検証用正解ラベル
            t_test_x = test['x']
            t_test_y = test['y']

            # X座標
            train_data_x = lgb.Dataset(X_train_data, label=t_train_x)
            eval_data_x = lgb.Dataset(X_test_data, label=t_test_x, reference= train_data_x)

            # y座標
            train_data_y = lgb.Dataset(X_train_data, label=t_train_y)
            eval_data_y = lgb.Dataset(X_test_data, label=t_test_y, reference= train_data_y)

            params_xy = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'verbose': -1,
                'metric': 'rmse'
            }

            # 座標の訓練
            gbm_x = op_lgb.train(
            # gbm_x = lgb.train(
                params_xy,
                train_data_x,
                valid_sets=eval_data_x,
                num_boost_round=1000,
                verbose_eval=50,
                early_stopping_rounds=15,
                )
            save_model(bname(facility_path),floor_num,'x',gbm_x)

            gbm_y = op_lgb.train(
            # gbm_y = lgb.train(
                params_xy,
                train_data_y,
                valid_sets=eval_data_y,
                num_boost_round=1000,
                verbose_eval=50,
                early_stopping_rounds=15,
                )
            save_model(bname(facility_path),floor_num,'y',gbm_y)


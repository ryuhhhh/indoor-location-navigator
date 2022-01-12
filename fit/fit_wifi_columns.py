import optuna.integration.lightgbm as op_lgb
import lightgbm as lgb
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os
import csv

DATA_DIR_1000 = './got_data_wifi_1000columns/*'
MDOELS_DIR_1000 = './models_wifi_1000columns'
DATA_DIR_ALL = './got_data_wifi_all_columns/*'
MDOELS_DIR_ALL = './models_wifi_all_columns'
DATA_DIR_LAG = './got_data_lag_wifi_1000columns/*'
MDOELS_DIR_LAG = './models_wifi_lag_columns'

DATA_DIR = DATA_DIR_LAG
MDOELS_DIR = MDOELS_DIR_LAG

def bname(object_name):
    return os.path.basename(object_name.rstrip(os.sep))

def save_model(facility_path,model_name,model):
    path = f'{MDOELS_DIR}/{bname(facility_path)[:-4]}/'
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}/{model_name}.pickle', mode='wb') as f:
        pickle.dump(model, f)
    print(f'{model_name}モデルを保存')


def get_target_facilities():
    """
    submitに使用する施設を取得
    """
    with open('./test_facilities.csv','r',encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=",", doublequote=True):
            return row

if __name__== "__main__":

    got_data_folders = glob.glob(DATA_DIR)
    skip = True
    # 各施設 を順に走査
    for i,facility_csv_path in enumerate(got_data_folders):
        print(f'{i+1}施設目')
        faility_id = bname(facility_csv_path)[:-4]
        print(faility_id)
        target_facilities = get_target_facilities()
        if faility_id not in target_facilities:
            print(f'{faility_id}は載っていないため、スキップ')
            continue
        # 一時的
        # if bname(facility_csv_path) == '5cdac620e403deddaf467fdb.csv':
        #     skip = False
        # if skip:
        #     continue
        facility_data = pd.read_csv(facility_csv_path).iloc[:,1:]
        if facility_data.shape[0] <= 0 or len(facility_data['floor'].unique()) <= 1:
            continue
        # trainとtestで分ける
        train, test = train_test_split(facility_data, test_size=0.2)
        X_train = train.iloc[:,0:-3]
        X_test = test.iloc[:,0:-3]
        # 訓練用正解ラベル
        t_train_floor = train['floor']
        t_train_x = train['x']
        t_train_y = train['y']
        # 検証用正解ラベル
        t_test_floor = test['floor']
        t_test_x = test['x']
        t_test_y = test['y']

        min_floor = t_train_floor.min()

        if min_floor < 0:
            t_train_floor = t_train_floor + (-min_floor)
            t_test_floor = t_test_floor + (-min_floor)
            tmp_facility_floor_data = facility_data['floor']  + (-min_floor)
            num_class = tmp_facility_floor_data.max()+1
        else:
            num_class = facility_data['floor'].max()+1

        # フロア
        train_data_floor = lgb.Dataset(X_train, label=t_train_floor)
        eval_data_floor = lgb.Dataset(X_test, label=t_test_floor, reference= train_data_floor)

        # X座標
        train_data_x = lgb.Dataset(X_train, label=t_train_x)
        eval_data_x = lgb.Dataset(X_test, label=t_test_x, reference= train_data_x)

        # y座標
        train_data_y = lgb.Dataset(X_train, label=t_train_y)
        eval_data_y = lgb.Dataset(X_test, label=t_test_y, reference= train_data_y)

        params_floor = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': num_class,
            'verbose': -1,
        }

        params_xy = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'verbose': -1,
            'metric': 'rmse'
        }

        # フロアの訓練
        gbm_floor = lgb.train(
            params_floor,
            train_data_floor,
            valid_sets=eval_data_floor,
            num_boost_round=500,
            verbose_eval=10,
            early_stopping_rounds=10,
            )
        save_model(facility_csv_path,'floor',gbm_floor)

        # 座標の訓練
        gbm_x = op_lgb.train(
            params_xy,
            train_data_x,
            valid_sets=eval_data_x,
            num_boost_round=500,
            verbose_eval=10,
            early_stopping_rounds=10,
            )
        save_model(facility_csv_path,'x',gbm_x)

        gbm_y = op_lgb.train(
            params_xy,
            train_data_y,
            valid_sets=eval_data_y,
            num_boost_round=500,
            verbose_eval=10,
            early_stopping_rounds=10,
            )
        save_model(facility_csv_path,'y',gbm_y)

        with open(f'{MDOELS_DIR}/{bname(facility_csv_path)[:-4]}/meta.txt', 'w') as f:
            f.write(str(min_floor))
            print('メタデータ記入')

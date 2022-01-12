"""
testデータを使って、提出ファイルを作成
"""
import glob
import os
import pandas as pd
import csv
import pickle
import numpy as np

def bname(object_name):
    return os.path.basename(object_name.rstrip(os.sep))

def get_site_id(path):
    with open(path,'r',encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", doublequote=True):
            if 'SiteID' in row[1]:
                return row[1][len('SiteID:'):]

def get_wifi_data(path,bssid_list):
    # wifiデータをすべて取得
    wifi_list = []
    print(f'{path}のwifiデータを取得')
    with open(path,'r',encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", doublequote=True):
            if row[1] == 'TYPE_WIFI' and row[3] in bssid_list:
                # wifi: 0列目にtimestamp,1列目にTYPE_WIFI,3列目にbssid,4列目にRSSI
                wifi_list.append([int(row[0]),row[3],float(row[4])])
    return wifi_list

def load_model(path):
    with open(path, mode="rb") as f:
        model = pickle.load(f)
    return model

def get_nearest_data(timestamp,wifi_timestamp_list,wifi_list,bssid_list):
    """
    時刻が一番近い かつ 利用頻度ランキング上位の wifiとビーコンのデータを取得
    """
    # waypointのtimestampとの時間差分リストを取得
    wifi_timedelta = np.abs(wifi_timestamp_list - timestamp)
    # 時間差リストソートでwifi_listも一緒にソート
    c = zip(wifi_timedelta,wifi_list)
    c = sorted(c)
    wifi_timedelta,wifi_list = zip(*c)
    # 最大1000件(wifiの特徴量数)
    wifi_timedelta = wifi_timedelta[:1000]
    # 一番近いインデックスを全てを取得
    wifi_min_idxs = [i for i, time_delta in enumerate(wifi_timedelta) if time_delta == min(wifi_timedelta)]
    # 一番近いwifiデータを取得
    target_wifi_list = np.asarray(wifi_list)[wifi_min_idxs]
    # bssidリストに載っている奴のみ
    target_wifi_list = [wifi_data for wifi_data in target_wifi_list if wifi_data[1] in bssid_list]
    if len(target_wifi_list) == 0:
        return target_wifi_list
    # bassidとrssiを返す
    target_wifi_list = np.array(target_wifi_list)[:,[1,2]]
    return target_wifi_list

if __name__ == '__main__':
    """
    site_path_timestamp
    """
    TEST_DIR = './download_data/indoor-location-navigation/test'
    submission_df = pd.read_csv('./sample_submission.csv')
    pre_site_id = ''
    pre_path = ''
    rmse_sum = 0
    # skip = True
    for i,site_path_timestamp in enumerate(submission_df['site_path_timestamp']):
        # print(f'{i}番目')
        # if site_id == '5cdac620e403deddaf467fdb':
        #     skip = False
        # if skip:
        #     continue
        spt_list = site_path_timestamp.split('_')
        site_id = spt_list[0]
        path = spt_list[1]
        timestamp = spt_list[2]
        # site_idが1つ前と違うなら、モデル/メタ/特徴量を取得
        if site_id != pre_site_id:
            print(f'施設:{site_id}')
            # モデルを取得
            model_x = load_model(f'./models_wifi_1000columns/{site_id}/x.pickle')
            model_y = load_model(f'./models_wifi_1000columns/{site_id}/y.pickle')
            rmse_sum += (model_x.best_score['valid_0']['rmse']+model_y.best_score['valid_0']['rmse'])
            model_floor = load_model(f'./models_wifi_1000columns/{site_id}/floor.pickle')
            # 特徴量のカラムを取得
            df = pd.read_csv(f'./got_data_wifi_1000columns/{site_id}.csv')
            bssid_list = df.columns[:-3]
            # メタ取得
            with open(f'./models_wifi_1000columns/{site_id}/meta.txt') as f:
                # 最低階 (例)B2 = -2、フロアの予測にこの値を足す
                min_floor_num = int(f.read())
        # pathが1つ前と違うなら、wifiデータを取得
        if pre_path != path:
            wifi_list = get_wifi_data(f'{TEST_DIR}/{path}.txt',bssid_list)
            wifi_timestamp_list = np.asarray(wifi_list)[:,0].astype(np.int64)
        # timestampと一番近いwifi_dataを取得
        target_wifi_list = get_nearest_data(int(timestamp),wifi_timestamp_list,wifi_list,bssid_list)
        # 特徴量用意
        input_series = pd.Series(data=target_wifi_list[:,1],index=target_wifi_list[:,0])
        input_df = pd.DataFrame(columns=bssid_list,index=[site_path_timestamp]).drop('Unnamed: 0', axis=1)
        input_series = input_series[np.logical_not(input_series.index.duplicated())]
        input_df = input_df.append(input_series,ignore_index=True).drop(df.index[[0]]).astype('float64')
        # wifi_dataを使って予測
        predict_floor = model_floor.predict(input_df, num_iteration=model_floor.best_iteration)
        predict_floor = np.argmax(predict_floor)+min_floor_num
        predict_x = model_x.predict(input_df, num_iteration=model_x.best_iteration)[0]
        predict_y = model_y.predict(input_df, num_iteration=model_y.best_iteration)[0]
        # 結果をsubmission_dfに書き込む
        submission_df.loc[submission_df['site_path_timestamp']==site_path_timestamp,['floor','x','y']] = [predict_floor,predict_x,predict_y]
        print(predict_floor,predict_x,predict_y)        # 値の更新
        print(f'rmseの合計:{rmse_sum}')
        submission_df.to_csv('./submission.csv')
        pre_site_id = site_id
        pre_path = path

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

def get_data(path,id_list):
    # wifiデータをすべて取得
    wifi_list = []
    beacon_list = []
    print(f'{path}のwifi/beaconデータを取得')
    with open(path,'r',encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", doublequote=True):
            if row[1] == 'TYPE_WIFI' and row[3] in id_list:
                # wifi: 0列目にtimestamp,1列目にTYPE_WIFI,3列目にbssid,4列目にRSSI
                wifi_list.append([int(row[0]),row[3],float(row[4])])
            # 特徴量 beacon 取得
            elif row[1] == 'TYPE_BEACON' and row[8] in id_list:
                beacon_list.append([int(row[0]),row[7],row[8]])
    return wifi_list,beacon_list

def load_model(path):
    with open(path, mode="rb") as f:
        model = pickle.load(f)
    return model

def get_nearest_data(timestamp,id_list,wifi_list,wifi_timestamp_list,beacon_list,beacon_timestamp_list):
    """
    時刻が一番近い かつ 利用頻度ランキング上位の wifiとビーコンのデータを取得
    """
    # waypointのtimestampとの時間差分リストを取得
    wifi_timedelta = np.abs(wifi_timestamp_list - timestamp)
    # 時間差リストソートでwifi_listも一緒にソート
    c = zip(wifi_timedelta,wifi_list)
    c = sorted(c)
    wifi_timedelta,wifi_list = zip(*c)
    # 最大1500件(wifiの特徴量数)
    wifi_timedelta = wifi_timedelta[:1500]
    # 一番近いインデックスを全てを取得
    wifi_min_idxs = [i for i, time_delta in enumerate(wifi_timedelta) if time_delta == min(wifi_timedelta)]
    # 一番近いwifiデータを取得
    target_wifi_list = np.asarray(wifi_list)[wifi_min_idxs]
    # bssidリストに載っている奴のみ
    target_wifi_list = [wifi_data for wifi_data in target_wifi_list if wifi_data[1] in id_list]
    # if len(target_wifi_list) == 0:
    #     return target_wifi_list
    # bassidとrssiを返す
    target_wifi_list = np.array(target_wifi_list)[:,[1,2]]

    # 一番近いtimestampから±3000msのbeaconデータ取得
    beacon_min_timedelta = np.abs(beacon_timestamp_list - timestamp)
    closest_beacon_time = beacon_timestamp_list[np.argmin(beacon_min_timedelta)]
    # timestampと±3000以内のbeaconデータを取得
    beacon_min_idxs = [i for i, timestamp_ in enumerate(beacon_timestamp_list) if closest_beacon_time-3000 < timestamp_ and timestamp_ < closest_beacon_time+3000 ]
    target_beacon_list = np.asarray(beacon_list)[beacon_min_idxs]
    # mac_listに載っているやつのみ使用
    target_beacon_list = [beacon_data for beacon_data in target_beacon_list if beacon_data[2] in id_list]
    target_beacon_list = np.array(target_beacon_list)[:,[2,1]]
    return target_wifi_list,target_beacon_list

if __name__ == '__main__':
    """
    site_path_timestamp
    """
    TEST_DIR = './download_data/indoor-location-navigation/test'
    submission_df = pd.read_csv('./sample_submission.csv')
    pre_site_id = ''
    pre_path = ''
    rmse_sum = 0
    skip = True
    for i,site_path_timestamp in enumerate(submission_df['site_path_timestamp']):
        print(f'{i}番目')
        # if site_id == '5d27097f03f801723c320d97':
        #     skip = False
        #     continue
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
            model_x = load_model(f'./models_wifi_beacon_1500/{site_id}/x.pickle')
            model_y = load_model(f'./models_wifi_beacon_1500/{site_id}/y.pickle')
            model_floor = load_model(f'./models_wifi_beacon_1500/{site_id}/floor.pickle')
            rmse_sum += (model_x.best_score['valid_0']['rmse']+model_y.best_score['valid_0']['rmse'])
            print(model_x.best_score['valid_0']['rmse'],model_y.best_score['valid_0']['rmse'])
            # 特徴量のカラムを取得
            df = pd.read_csv(f'./got_data_wifi_beacon_1500columns/{site_id}.csv')
            id_list = df.columns[:-3]
            # メタ取得
            with open(f'./models_wifi_beacon_1500/{site_id}/meta.txt') as f:
                # 最低階 (例)B2 = -2、フロアの予測にこの値を足す
                min_floor_num = int(f.read())
        # pathが1つ前と違うなら、wifiデータを取得
        if pre_path != path:
            wifi_list,beacon_list = get_data(f'{TEST_DIR}/{path}.txt',id_list)
            wifi_timestamp_list = np.asarray(wifi_list)[:,0].astype(np.int64)
            beacon_timestamp_list = np.asarray(beacon_list)[:,0].astype(np.int64)
        # timestampと一番近いdataを取得
        target_wifi_list,target_beacon_list = get_nearest_data(int(timestamp),id_list,wifi_list,wifi_timestamp_list,beacon_list,beacon_timestamp_list)
        # 特徴量用意
        input_series = pd.Series(data=target_wifi_list[:,1],index=target_wifi_list[:,0])
        if len(target_beacon_list) != 0:
            input_series_beacon = pd.Series(data=target_beacon_list[:,1],index=target_beacon_list[:,0])
            input_series = pd.concat([input_series, input_series_beacon])

        input_df = pd.DataFrame(columns=id_list,index=[site_path_timestamp]).drop('Unnamed: 0', axis=1)
        input_series = input_series[np.logical_not(input_series.index.duplicated())]
        input_df = input_df.append(input_series,ignore_index=True).drop(df.index[[0]]).astype('float64')
        # wifi_dataを使って予測
        predict_floor = model_floor.predict(input_df, num_iteration=model_floor.best_iteration)
        predict_floor = np.argmax(predict_floor)+min_floor_num
        predict_x = model_x.predict(input_df, num_iteration=model_x.best_iteration)[0]
        predict_y = model_y.predict(input_df, num_iteration=model_y.best_iteration)[0]
        # 結果をsubmission_dfに書き込む
        submission_df.loc[submission_df['site_path_timestamp']==site_path_timestamp,['floor','x','y']] = [predict_floor,predict_x,predict_y]
        print(predict_floor,predict_x,predict_y)
        # 値の更新
        pre_site_id = site_id
        pre_path = path
        submission_df.to_csv('./submission.csv')
    print(rmse_sum)

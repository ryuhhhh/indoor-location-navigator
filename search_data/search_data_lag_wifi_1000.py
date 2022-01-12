"""
データ探索を行う
"""

import os
from glob import glob
import csv
import pandas as pd
import numpy as np
from collections import Counter
import re
from pathlib import Path

LAG_NUM = 5

def bname(object_name):
    return os.path.basename(object_name.rstrip(os.sep))

def make_data(path,bssid_list):
    # WAYPOINT: 1列目TYPE_WAYPOINT,2列名x,3列目y
    # wifi: 0列目にtimestamp,1列目にTYPE_WIFI,3列目にbssid,4列目にRSSI,5列目に周波数
    # beacon: 0列目にtimestamp,1列目にTYPE_BEACON,8列目にdistance,9列目にmac_address
    way_point_list = []
    wifi_list = []
    beacon_list = []
    # wifi特徴量,floor,x,y
    train_data = pd.DataFrame()
    with open(path,'r',encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", doublequote=True):
            # 教師データ取得
            if row[1] == 'TYPE_WAYPOINT':
                way_point_list.append([int(row[0]),row[2],row[3]])
            # 特徴量 wifi 取得
            elif row[1] == 'TYPE_WIFI':
                wifi_list.append([int(row[0]),row[3],row[4],row[5]])
    if len(wifi_list) == 0:
        return train_data
    # 施設の再頻出wifiデータのみ使用
    wifi_list = [wifi_data for wifi_data in wifi_list if wifi_data[1] in bssid_list]
    for way_point in way_point_list:
        # waypointと一番近い過去LAG_NUM地点までのwifiのデータ取得
        target_wifi_list =\
             get_lag_data(LAG_NUM,way_point,wifi_list)
        if len(target_wifi_list) == 0:
            continue
        index, value = zip(*target_wifi_list)
        target_wifi_series = pd.Series(value, index=index)
        target_wifi_series['x'] = way_point[1]
        target_wifi_series['y'] = way_point[2]
        # 重複したINDEXを削除
        target_wifi_series = target_wifi_series[np.logical_not(target_wifi_series.index.duplicated())]
        # bssid_listを列名,floor,x,y の訓練データ作成 wifi_listは開いていれる
        train_data = train_data.append(target_wifi_series,ignore_index=True)
    return train_data

def get_time_boundary(wifi_list,target_timestamp,is_future=False):
    """
    lag_num分のまとまったwifi_listを取得
    """
    wifi_timestamp_list = np.asarray(wifi_list)[:,0].astype(np.int64)
    lag_list = []
    pre_d = wifi_timestamp_list[0]
    pre_index = 0
    lag_count = 0
    for i,d in enumerate(wifi_timestamp_list):
        # 過去から取得の場合は、時刻が対象時刻を越えた時 または 指定回数ラグデータを取得した時
        if (not is_future) and (d > target_timestamp) or (lag_count >= LAG_NUM):
            break
        # 未来から取得の場合はlag_num分まで取得
        if is_future and (lag_count >= LAG_NUM):
            break
        # 未来から取得の場合は自身の時刻よりも未来でないといけない
        elif is_future and (target_timestamp > d):
            continue
        if d != pre_d:
            lag_count += 1
            lag_list.append(wifi_list[pre_index:i])
            pre_index = i
        pre_d = d
    # 未来の場合反転
    if is_future:
        return lag_list[::-1]
    return lag_list

def get_lag_data(lag_num,way_point,wifi_list):
    """
    時刻が近い順にラグデータを取得
    """
    # 過去のラグリストを取得
    lag_list = get_time_boundary(wifi_list,way_point[0])
    if len(lag_list) < lag_num:
        # 過去にラグリストが作れないなら、未来から作る
        lag_list = get_time_boundary(wifi_list,way_point[0],is_future=True)
    target_wifi_list = []
    for i,lags in enumerate(lag_list):
        for lag in lags:
            target_wifi_list.append([lag[1]+'_lag_'+str(len(lag_list)-i),float(lag[2])])
    return target_wifi_list

def get_bssid_list(floor_folders):
    """
    BSSIDのリスト作成
    """
    bssid_list = []
    # フロアのパスを取得
    for floor_folder in floor_folders:
        print(f'{bname(floor_folder)}のBSSID/MACを探索中')
        for survey_path in glob(f'{floor_folder}/*'):
            with open(survey_path,'r',encoding="utf-8") as f:
                for row in csv.reader(f, delimiter="\t", doublequote=True):
                    if row[1] == 'TYPE_WIFI':
                        bssid_list.append(row[3])
    bssid_tuple, _ = zip(*Counter(bssid_list).most_common(1500))
    return list(bssid_tuple)

def get_floor_num(floor):
    # 数字が含まれていない or 2文字以下の場合スキップ
    if len(floor) < 2 or (not bool(re.search(r'\d', floor))):
        return -99
    FLOOR = ['F','L']
    BASEMENT = ['B']
    floor_num = int(re.sub(r"\D", "", floor))
    floor_str = re.sub(r"\d", "", floor)
    if floor_str in FLOOR:
        return floor_num - 1
    elif floor_str in BASEMENT:
        return -floor_num
    else:
        return -99

def get_target_facilities():
    """
    submitに使用する施設を取得
    """
    with open('./test_facilities.csv','r',encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=",", doublequote=True):
            return row

def create_df_columns(bssid_list):
    columns = []
    for id in bssid_list:
        for n in range(LAG_NUM):
            columns.append(id+f'_lag_{str(n+1)}')
    return columns

if __name__== "__main__":
    DIR = './download_data/indoor-location-navigation'
    train_folders = glob(f'{DIR}/train/*')
    print(f'施設数: {len(train_folders)}')
    skip = True
    target_facilities = get_target_facilities()
    for facility_id in train_folders:
        if bname(facility_id) not in target_facilities:
            print(f'{bname(facility_id)}は載っていないため、スキップ')
            continue
        # if bname(facility_id) == '5cd56b99e2acfd2d33b5f491':
        #     skip = False
        print(f'施設ID {bname(facility_id)} を探索')
        # if skip:
        #     continue
        floor_folders = glob(f'{facility_id}/*')
        print(f'階層一覧: {[bname(p) for p in floor_folders]}')
        bssid_list = get_bssid_list(floor_folders)
        columns = create_df_columns(bssid_list)
        train_data_parent = pd.DataFrame(columns=[*columns,'floor','x','y'])
        # 各フロア の調査データから訓練データ作成
        for floor_paths in [(glob(f'{p}/*')) for p in floor_folders]:
            floor_num = get_floor_num(Path(floor_paths[0]).parts[-2])
            if floor_num == -99:
                continue
            # 各調査データから特徴量作成
            for survey in floor_paths:
                print(f'ターゲットファイル: {bname(survey)}')
                # if bname(survey) != '5d8f0954b6e29d0006fb8c0d.txt':
                #     continue
                train_data = make_data(survey,bssid_list)
                train_data['floor'] = floor_num
                train_data_parent = pd.concat([train_data_parent,train_data],axis=0)
                print(train_data_parent)
        train_data_parent.to_csv(f'./got_data_lag_wifi_1000columns/{bname(facility_id)}.csv')
    print('全て終了')
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

COLUMNS_NUM = 1500

def bname(object_name):
    return os.path.basename(object_name.rstrip(os.sep))

def make_data(path,floor_num):
    # WAYPOINT: 1列目TYPE_WAYPOINT,2列名x,3列目y
    # wifi: 0列目にtimestamp,1列目にTYPE_WIFI,3列目にbssid,4列目にRSSI,5列目に周波数
    # beacon: 0列目にtimestamp,1列目にTYPE_BEACON,7列目にdistance,8列目にmac_address
    way_point_list = []
    wifi_list = []
    beacon_list = []
    train_data = pd.DataFrame(columns=['floor','x','y'])
    with open(path,'r',encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", doublequote=True):
            # 教師データ取得
            if row[1] == 'TYPE_WAYPOINT':
                way_point_list.append([int(row[0]),row[2],row[3]])
            # 特徴量 wifi 取得
            elif row[1] == 'TYPE_WIFI':
                wifi_list.append([int(row[0]),row[3],row[4],row[5]])
            # 特徴量 beacon 取得
            elif row[1] == 'TYPE_BEACON':
                beacon_list.append([int(row[0]),row[7],row[8]])
    if len(wifi_list) == 0:
        return train_data
    for way_point in way_point_list:
        target_wifi_list,target_beacon_list =\
             get_nearest_data(way_point,wifi_list,beacon_list)
        # wifiの特徴量が取れないならスキップ
        if len(target_wifi_list) == 0:
            continue
        index, value = zip(*target_wifi_list)
        target_series = pd.Series(value, index=index)
        # beacon取らないときコメントアウト
        # if len(target_beacon_list) != 0:
        #     index, value = zip(*target_beacon_list)
        #     target_beacon_series = pd.Series(value, index=index)
        #     target_series = pd.concat([target_series, target_beacon_series])
        target_series['floor'] = floor_num
        target_series['x'] = way_point[1]
        target_series['y'] = way_point[2]
        # 重複したINDEXを削除
        target_series = target_series[np.logical_not(target_series.index.duplicated())]
        train_data = train_data.append(target_series,ignore_index=True)
    return train_data

def get_time_boundary(timedelta_list):
    v = timedelta_list[0]
    for i,d in enumerate(timedelta_list):
        if v != d:
            break
    return i

def get_nearest_data(way_point,wifi_list,beacon_list):
    """
    時刻が一番近い かつ 利用頻度ランキング上位の wifiとビーコンのデータを取得
    """
    # timestampだけをリスト化
    wifi_timestamp_list = np.asarray(wifi_list)[:,0].astype(np.int64)
    # waypointのtimestampとの時間差分リストを取得
    wifi_timedelta = np.abs(wifi_timestamp_list - way_point[0])
    # 時間差リストソートでwifi_listも一緒にソート
    c = zip(wifi_timedelta,wifi_list)
    c = sorted(c)
    wifi_timedelta,wifi_list = zip(*c)
    # 一番近いwifiデータを取得
    target_wifi_list = np.asarray(wifi_list)[:get_time_boundary(wifi_timedelta)]
    if len(target_wifi_list) != 0:
        target_wifi_list = np.array(target_wifi_list)[:,[1,2]]
    else:
        target_wifi_list = []

    if len(beacon_list) == 0:
        return target_wifi_list,[]
    # beaconは一番近いデータの±3000msを取得
    way_point_timestamp = way_point[0]
    beacon_timestamp_list = np.asarray(beacon_list)[:,0].astype(np.int64)
    beacon_min_timedelta = np.abs(beacon_timestamp_list - way_point_timestamp)
    # 一番近いtimestampを取得
    closest_beacon_time = beacon_timestamp_list[np.argmin(beacon_min_timedelta)]
    # 一番近いtimestamの±5000msの範囲のbeaconデータを取得
    beacon_min_idxs = [i for i, timestamp in enumerate(beacon_timestamp_list) if closest_beacon_time-5000 < timestamp and timestamp < closest_beacon_time+5000 ]
    target_beacon_list = np.asarray(beacon_list)[beacon_min_idxs]
    if len(target_beacon_list) != 0:
        target_beacon_list = np.array(target_beacon_list)[:,[2,1]]
    else:
        target_beacon_list = []
    return target_wifi_list,target_beacon_list

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

def save_file(df,path,floor_name):
    if not os.path.exists(path):
        os.mkdir(path)
    df.to_csv(f'{path}/{floor_name}.csv')
    print(f'データを保存')

if __name__== "__main__":
    DIR = './download_data/indoor-location-navigation'
    SAVE_DIR = 'got_data_per_floor_wifi_only'
    # SAVE_DIR = 'got_data_per_floor'
    train_folders = glob(f'{DIR}/train/*')
    print(f'施設数: {len(train_folders)}')
    skip = True
    target_facilities = get_target_facilities()
    # 各施設 を順に走査
    for i,facility_id in enumerate(train_folders):
        print(f'{i}番目')
        if bname(facility_id) not in target_facilities:
            print(f'{bname(facility_id)}は載っていないため、スキップ')
            continue
        # if bname(facility_id) == '5d27099f03f801723c32511d':
        # if i == 120:
        #     skip = False
        #     continue
        # if skip:
        #     continue
        floor_folders = glob(f'{facility_id}/*')
        print(f'階層一覧: {[bname(p) for p in floor_folders]}')
        # 各フロア の調査データから訓練データ作成
        for floor_paths in [(glob(f'{p}/*')) for p in floor_folders]:
            train_data_parent = pd.DataFrame()
            floor_num = get_floor_num(Path(floor_paths[0]).parts[-2])
            if floor_num == -99:
                continue
            # 各調査データから特徴量作成
            for survey in floor_paths:
                print(f'{i}施設目: {bname(facility_id)} の {floor_num} を探索')
                print(f'ターゲットファイル: {bname(survey)}')
                # if bname(survey) != '5d0b164bea52920008961c9d.txt':
                #     continue
                train_data = make_data(survey,floor_num)
                train_data_parent = pd.concat([train_data_parent,train_data],axis=0)
                print(train_data_parent)
            save_file(train_data_parent,f'./{SAVE_DIR}/{bname(facility_id)}',floor_num)
    print('全て終了')
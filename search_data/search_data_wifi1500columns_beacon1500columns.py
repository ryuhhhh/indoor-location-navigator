"""
データ探索を行う
★ 施設名.csv 特徴量ファイル作成 列はwifiのbssid,floor,x,y
★ 上記のcsvに代入していく
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

def make_data(path,bssid_list,mac_list):
    # WAYPOINT: 1列目TYPE_WAYPOINT,2列名x,3列目y
    # wifi: 0列目にtimestamp,1列目にTYPE_WIFI,3列目にbssid,4列目にRSSI,5列目に周波数
    # beacon: 0列目にtimestamp,1列目にTYPE_BEACON,7列目にdistance,8列目にmac_address
    way_point_list = []
    wifi_list = []
    beacon_list = []
    train_data = pd.DataFrame(columns=[*bssid_list,*mac_list,'floor','x','y'])
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
             get_nearest_data(way_point,wifi_list,beacon_list,bssid_list,mac_list)
        # wifiの特徴量が取れないならスキップ
        if len(target_wifi_list) == 0:
            continue
        index, value = zip(*target_wifi_list)
        target_series = pd.Series(value, index=index)
        if len(target_beacon_list) != 0:
            index, value = zip(*target_beacon_list)
            target_beacon_series = pd.Series(value, index=index)
            target_series = pd.concat([target_series, target_beacon_series])
        target_series['x'] = way_point[1]
        target_series['y'] = way_point[2]
        # 重複したINDEXを削除
        target_series = target_series[np.logical_not(target_series.index.duplicated())]
        train_data = train_data.append(target_series,ignore_index=True)
    return train_data

def get_nearest_data(way_point,wifi_list,beacon_list,bssid_list,mac_list):
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
    # 全特徴量数の80%までを見る
    wifi_timedelta = wifi_timedelta[:int(len(bssid_list)*0.8)]
    # 一番近いインデックスを全てを取得
    wifi_min_idxs = [i for i, time_delta in enumerate(wifi_timedelta) if time_delta == min(wifi_timedelta)]
    # 一番近いwifiデータを取得
    target_wifi_list = np.asarray(wifi_list)[wifi_min_idxs]
    # bssid_listに載っているwifiデータだけ使用する
    target_wifi_list = [wifi_data for wifi_data in target_wifi_list if wifi_data[1] in bssid_list]
    if len(target_wifi_list) != 0:
        target_wifi_list = np.array(target_wifi_list)[:,[1,2]]
    else:
        target_wifi_list = []

    if len(mac_list) == 0 or len(beacon_list) == 0:
        return target_wifi_list,[]
    # beaconは一番近いデータの±3000msを取得
    way_point_timestamp = way_point[0]
    beacon_timestamp_list = np.asarray(beacon_list)[:,0].astype(np.int64)
    beacon_min_timedelta = np.abs(beacon_timestamp_list - way_point_timestamp)
    # 一番近いtimestampを取得
    closest_beacon_time = beacon_timestamp_list[np.argmin(beacon_min_timedelta)]
    # 一番近いtimestamの±3000msの範囲のbeaconデータを取得
    beacon_min_idxs = [i for i, timestamp in enumerate(beacon_timestamp_list) if closest_beacon_time-3000 < timestamp and timestamp < closest_beacon_time+3000 ]
    target_beacon_list = np.asarray(beacon_list)[beacon_min_idxs]
    # mac_listに載っているやつのみ使用
    target_beacon_list = [beacon_data for beacon_data in target_beacon_list if beacon_data[2] in mac_list]
    if len(target_beacon_list) != 0:
        target_beacon_list = np.array(target_beacon_list)[:,[2,1]]
    else:
        target_beacon_list = []
    return target_wifi_list,target_beacon_list

def get_bssid_mac_list(floor_folders):
    """
    BSSIDの多い順リスト作成
    """
    bssid_list = []
    mac_list = []
    # フロアのパスを取得
    for floor_folder in floor_folders:
        print(f'{bname(floor_folder)}のBSSID/MACを探索中')
        for survey_path in glob(f'{floor_folder}/*'):
            with open(survey_path,'r',encoding="utf-8") as f:
                for row in csv.reader(f, delimiter="\t", doublequote=True):
                    if row[1] == 'TYPE_WIFI':
                        bssid_list.append(row[3])
                    elif row[1] == 'TYPE_BEACON':
                        mac_list.append(row[8])
    bssid_tuple, _ = zip(*Counter(bssid_list).most_common(COLUMNS_NUM))
    if len(mac_list) !=0:
        mac_tuple, _ = zip(*Counter(mac_list).most_common(COLUMNS_NUM))
        return list(bssid_tuple),list(mac_tuple)
    return list(bssid_tuple),[]

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

if __name__== "__main__":
    DIR = './download_data/indoor-location-navigation'
    train_folders = glob(f'{DIR}/train/*')
    print(f'施設数: {len(train_folders)}')
    # skip = True
    target_facilities = get_target_facilities()
    # 各施設 を順に走査
    for i,facility_id in enumerate(train_folders):
        print(f'{i}番目')
        if bname(facility_id) not in target_facilities:
            print(f'{bname(facility_id)}は載っていないため、スキップ')
            continue
        # if bname(facility_id) == '5cd56b99e2acfd2d33b5f491':
        # if i == 120:
        #     skip = False
        #     continue
        # if skip:
        #     continue
        train_data_parent = pd.DataFrame()
        floor_folders = glob(f'{facility_id}/*')
        print(f'階層一覧: {[bname(p) for p in floor_folders]}')
        # 全フロアのBSSID多い順リストを取得
        bssid_list,mac_list = get_bssid_mac_list(floor_folders)
        # 各フロア の調査データから訓練データ作成
        for floor_paths in [(glob(f'{p}/*')) for p in floor_folders]:
            floor_num = get_floor_num(Path(floor_paths[0]).parts[-2])
            if floor_num == -99:
                continue
            # 各調査データから特徴量作成
            for survey in floor_paths:
                print(f'{i}施設目: {bname(facility_id)} を探索')
                print(f'ターゲットファイル: {bname(survey)}')
                # if bname(survey) != '5d0b164bea52920008961c9d.txt':
                #     continue
                train_data = make_data(survey,bssid_list,mac_list)
                train_data['floor'] = floor_num
                train_data_parent = pd.concat([train_data_parent,train_data],axis=0)
                print(train_data_parent)
        train_data_parent.to_csv(f'./got_data_wifi_beacon_{COLUMNS_NUM}columns/{bname(facility_id)}.csv')
    print('全て終了')
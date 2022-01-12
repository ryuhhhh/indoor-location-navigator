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


def bname(object_name):
    return os.path.basename(object_name.rstrip(os.sep))

# def make_data(path,bssid_list,mac_list):
def make_data(path,bssid_list):
    # WAYPOINT: 1列目TYPE_WAYPOINT,2列名x,3列目y
    # wifi: 0列目にtimestamp,1列目にTYPE_WIFI,3列目にbssid,4列目にRSSI,5列目に周波数
    # beacon: 0列目にtimestamp,1列目にTYPE_BEACON,8列目にdistance,9列目にmac_address
    way_point_list = []
    wifi_list = []
    beacon_list = []
    # wifi特徴量,floor,x,y
    train_data = pd.DataFrame(columns=[*bssid_list,'floor','x','y'])
    with open(path,'r',encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", doublequote=True):
            # 教師データ取得
            if row[1] == 'TYPE_WAYPOINT':
                way_point_list.append([int(row[0]),row[2],row[3]])
            # 特徴量 wifi 取得
            elif row[1] == 'TYPE_WIFI':
                wifi_list.append([int(row[0]),row[3],row[4],row[5]])
            # ✅beaconも使うとき 特徴量 beacon 取得
            # elif row[1] == 'TYPE_BEACON':
            #     beacon_list.append([int(row[0]),row[7],row[8]])
    if len(wifi_list) == 0:
        return train_data
    for way_point in way_point_list:
        # waypointと一番近い時刻のwifiとbeaconのデータ取得 => ベースラインはwifiのみ
        # target_wifi_list は bssidとrsssi の2次元配列となる
        target_wifi_list,target_beacon_list =\
             get_nearest_data(way_point,wifi_list,beacon_list,bssid_list)
            #  ✅get_nearest_data(way_point,wifi_list,beacon_list,bssid_list,mac_list)
        # 特徴量が取れないなら無視
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

# ✅def get_nearest_data(way_point,wifi_list,beacon_list,bssid_list,mac_list):
def get_nearest_data(way_point,wifi_list,beacon_list,bssid_list):
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
    # 最大1000件(wifiの特徴量数)
    wifi_timedelta = wifi_timedelta[:1000]
    # 一番近いインデックスを全てを取得
    wifi_min_idxs = [i for i, time_delta in enumerate(wifi_timedelta) if time_delta == min(wifi_timedelta)]
    # 一番近いwifiデータを取得
    target_wifi_list = np.asarray(wifi_list)[wifi_min_idxs]
    # bssid_listに載っているwifiデータだけ使用する
    target_wifi_list = [wifi_data for wifi_data in target_wifi_list if wifi_data[1] in bssid_list]
    if len(target_wifi_list) == 0:
        return target_wifi_list,None
    # ✅wifiデータはbssid(1列目)とRSSI(2列目)のみ取得
    target_wifi_list = np.array(target_wifi_list)[:,[1,2]]
    return target_wifi_list,None

    # beaconは一番近いデータの±500msを取得 => 後回し
    if beacon_list:
        beacon_timestamp_list = np.asarray(beacon_list)[:,0].astype(np.int64)
        beacon_min_timedelta = np.abs(beacon_timestamp_list - way_point[0])
        beacon_min_idxs = [i for i, time_delta in enumerate(beacon_min_timedelta) if time_delta == min(beacon_min_timedelta)]
        target_beacon_list = np.asarray(beacon_list)[beacon_min_idxs]
        target_beacon_list = [beacon_data for beacon_data in target_beacon_list if beacon_data[2] in mac_list]
    else:
        target_beacon_list = []
    # wifiデータはbssid(1列目)とRSSI(2列目)のみ取得
    target_wifi_list = np.array(target_wifi_list)[:,[1,2]]
    return target_wifi_list,target_beacon_list

def get_bssid_list(floor_folders):
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
    bssid_tuple, _ = zip(*Counter(bssid_list).most_common(1000))
    # ✅一旦wifiのみ特徴量にする
    return list(bssid_tuple)
    mac_tuple, _ = zip(*Counter(mac_list).most_common(1000))
    return list(bssid_tuple),list(mac_tuple)

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

if __name__== "__main__":
    DIR = './download_data/indoor-location-navigation'
    train_folders = glob(f'{DIR}/train/*')
    print(f'施設数: {len(train_folders)}')
    skip = True
    # 各施設 を順に走査
    for facility_id in train_folders:
        if bname(facility_id) == '5cd56b99e2acfd2d33b5f491':
            skip = False
        train_data_parent = pd.DataFrame()
        print(f'施設ID {bname(facility_id)} を探索')
        if skip:
            continue
        floor_folders = glob(f'{facility_id}/*')
        print(f'階層一覧: {[bname(p) for p in floor_folders]}')
        # 全フロアのBSSID多い順リストを取得
        # bssid_list,mac_list = get_bssid_list(floor_folders)
        bssid_list = get_bssid_list(floor_folders)
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
                # ✅train_data = make_data(survey,bssid_list,mac_list)
                train_data = make_data(survey,bssid_list)
                train_data['floor'] = floor_num
                train_data_parent = pd.concat([train_data_parent,train_data],axis=0)
                print(train_data_parent)
        train_data_parent.to_csv(f'./got_data_wifi_1000columns/{bname(facility_id)}.csv')
    print('全て終了')
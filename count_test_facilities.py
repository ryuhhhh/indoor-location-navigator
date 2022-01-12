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

if __name__ == '__main__':
    """
    site_path_timestamp
    """
    submission_df = pd.read_csv('./sample_submission.csv')
    pre_site_id = ''
    site_list = []
    for i,site_path_timestamp in enumerate(submission_df['site_path_timestamp']):
        spt_list = site_path_timestamp.split('_')
        site_id = spt_list[0]
        if site_id != pre_site_id:
            print(f'施設:{site_id}')
            site_list.append(site_id)
        pre_site_id = site_id
    print(f'テストファイルの施設数：{len(site_list)}')
    print(site_list)
    np.savetxt("./test_facilities.csv", [site_list], delimiter =",",fmt ='% s')



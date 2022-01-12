# 概要
* 初めて参加したkaggleコンペティション「indoor-location-navigation」で実装したコードです。テーブルデータを扱います。
  * URL: https://www.kaggle.com/c/indoor-location-navigation/leaderboard
* コンペティションの概要は、ショッピングモール建物内の人の位置をスマホのセンサやwifiの強度のデータから予測するものです。
  * 訓練データ: wifiのセンサデータなど
  * 教師データ: 人の実際の位置(x,y座標)および何階にいるか
* (下位30%と残念な結果に終わりましたが、初めてのコンペティションでデータを加工し予測するのは楽しかったです。実際の人の位置と予測した位置の誤差について、1位の方は約1mの誤差で私は10mほどの誤差がでました。)

# 予測方法
* 各ショッピングモールごとに予測モデルを作成しました(ショッピングモールは全部で数百あります。)
* パラメータはwifiのRSSI(Received signal strength indication)とBeaconとの距離データを使用しました。
* モデルは勾配ブースティング木のLightGBMを使用しました。
* 訓練データは最大1500種のwifiのセンサデータやそこにbeaconのデータも加えたデータを使用しました。最終的にはそれらをアンサンブルにしたモデルを作成したりしました。

# フォルダ構成
## /search_data
* データをcsvから取得します
* 1000や1500種のwifiデータを取得します
* beaconとスマホとの距離のデータも取得します

## /fit
* モデルを学習します。
* モデルは勾配ブースティング木のLightGBMを使用しました。

## /predict
* 学習したモデルで予測を行います。
* 出力値は各入力行に対して、人の位置(x,y座標)と階数です。
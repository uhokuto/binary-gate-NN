# binary gate ニューラルネットワーク

## 1. binary gateにもとづくニューラルネットワーク[L0__sparse.py](L0__sparse.py)
epoch毎に変数zにbinary gateの重み（重要でない特徴量は0になっている）が入る。これを以下の通り
epoch毎のzを履歴としてdataframeに書き出す。また、最後の行にはこの履歴で各特徴量（列）別に0となった回数を書き出す。この回数が確率的に大きければ，この特徴量は重要ではない。
```python
L0_weight_df = pd.DataFrame(L0_weight,columns = feature_name).iloc[-100:,:]
zero_counts = (L0_weight_df == 0).sum()
L0_weight_df = pd.concat([L0_weight_df, pd.DataFrame([zero_counts], columns=df.columns)], ignore_index=True)
with open("L0_weight.csv", "w", encoding="cp932", errors="ignore", newline="") as f:
        L0_weight_df.to_csv(f, index=False)
```
詳細は、スライドを参照
## 2. 比較のためにbinary gateを介さない同じ構成のNNを実装した[neural.py](neural.py)
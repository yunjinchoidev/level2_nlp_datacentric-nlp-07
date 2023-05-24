import pandas as pd

train = pd.read_csv("../data/train.csv")

# keep=False -> 중복 싹 다 제거임 (2개가 중복되면 2개 모두 제거임)
df_drop = train.drop_duplicates(subset=["url"], keep=False)

df_drop.to_csv("drop_duplicates.csv")

import pandas as pd
from hanspell import spell_checker

train = pd.read_csv("../data/train.csv")

# input_text 에 대해서 맞춤법 검사를 적용해주기.
train["input_text"] = train["input_text"].apply(
    lambda x: spell_checker.check(x).checked
)

train.to_csv("spell_check.csv", index="first")

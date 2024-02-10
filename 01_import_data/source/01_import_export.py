import pandas as pd

train = pd.read_csv("01_import_data/input/train.csv")
test =  pd.read_csv("01_import_data/input/test.csv")





train.to_csv("01_import_data/output/train.csv", index=False)

test.to_csv("01_import_data/output/test.csv", index=False)












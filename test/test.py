import pandas as pd

df_train = pd.read_csv("../dataset/train_val_annotations/train.csv")
classes = len(set(df_train.category_id))
print(classes)
# result : 1394

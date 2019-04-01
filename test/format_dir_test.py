import pandas as pd

name_csv = "../dataset/train_val_annotations/val.csv"
# format dir into keras on-the-fly image generator format
# train and val data frames
df = pd.read_csv(name_csv)
print(df.iterrows)

# loop over image
for ix, (filename, image_id, category_id) in df.iterrows():
    # get names
    tv_dir, cat_dir, image_name = filename.split("/")
    print(image_name)

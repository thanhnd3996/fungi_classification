import os
import pandas as pd
import shutil


def format_dir(train_val_dir, target_dir, name_csv):
    # format dir into keras on-the-fly image generator format

    # train and val data frames
    df = pd.read_csv(name_csv)

    # loop over image
    for ix, (image_path, image_id, category_id) in df.iterrows():
        # get names
        tv_dir, super_cat_dir, cat_dir, image_name = image_path.split("/")
        original_img_path = "%s%s" % (train_val_dir, image_path)
        new_img_dir = "%s%d/" % (target_dir, category_id)
        new_img_path = "%s%d_%s" % (new_img_dir, image_id, image_name)
        # create category dir
        if not os.path.exists(new_img_path):
            os.makedirs(new_img_dir)
        # copy image
        shutil.copy(original_img_path, new_img_path)


if __name__ == '__main__':
    format_dir(train_val_dir="../dataset/images",
               target_dir="../dataset/train_images",
               name_csv="../dataset/train_val_annotations/train.csv")
    format_dir(train_val_dir="../dataset/images",
               target_dir="../dataset/val_images",
               name_csv="../dataset/train_val_annotations/val.csv")

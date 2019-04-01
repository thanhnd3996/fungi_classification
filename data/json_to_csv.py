import sys
import json

import pandas as pd


def json_to_csv(json_name):
    # read json
    with open(json_name, "r") as f:
        json_dict = json.loads(f.read())

    # get data frame
    df_annotations = pd.DataFrame(json_dict["annotations"])
    df_images = pd.DataFrame(json_dict["images"])
    data_frame = df_images.merge(df_annotations, how="inner", left_on="id", right_on="image_id")[[
        "file_name", "image_id", "category_id"]]
    return data_frame


if __name__ == '__main__':
    js_name = str(sys.argv[1])
    df = json_to_csv(js_name)
    csv_name = '.'.join(js_name.split('.')[:-1]) + '.csv'
    df.to_csv(csv_name, index=False, header=True)

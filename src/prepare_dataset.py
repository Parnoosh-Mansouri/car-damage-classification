import json
import pandas as pd
def prepare_dataset(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_label = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        
        if image_id not in images_label:
            images_label[image_id] = set()
        
        images_label[image_id].add(category_id)

    image_file={}
    for image in data["images"]:
        image_file[image["id"]] = image["file_name"]

    category_name = {}
    for category in data["categories"]:
        category_name[category["id"]] = category["name"]

    record = []
    for image_id,category_ids in images_label.items():
        file_name = image_file[image_id]
        labels = [category_name[category_id] for category_id in category_ids]
        label_vector = [0] * 6
        for category_id in category_ids:
            label_vector[category_id - 1] = 1
        record.append({
            "file_name":file_name,
            "labels":labels,
            "label_vector":label_vector
        })
 
    return record

train_path = "dataset/CarDD_COCO/annotations/instances_train2017.json"
val_path = "dataset/CarDD_COCO/annotations/instances_val2017.json"
test_path = "dataset/CarDD_COCO/annotations/instances_test2017.json"

train_records = prepare_dataset(train_path)
val_records = prepare_dataset(val_path)
test_records = prepare_dataset(test_path)

columns = [
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "lamp_broken",
    "tire_flat"
]

def record_to_dataframe(record):
    df = pd.DataFrame(record)
    df[columns]= pd.DataFrame(df["label_vector"].tolist(),index = df.index)
    df= df.drop(columns=["labels","label_vector"])
    return df

train_df = record_to_dataframe(train_records)
val_df = record_to_dataframe(val_records)
test_df = record_to_dataframe(test_records)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
print("CSV files saved successfully!")


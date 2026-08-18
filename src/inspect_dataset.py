import json
from collections import Counter

# Load annotation file
path = "dataset/CarDD_COCO/annotations/instances_train2017.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)


# Basic dataset information
print("Number of images:", len(data["images"]))
print("Number of annotations:", len(data["annotations"]))


# Damage categories
print("\nCategories:")
for category in data["categories"]:
    print(category["id"], "->", category["name"])


# Number of annotations per category
category_counts = Counter(
    annotation["category_id"]
    for annotation in data["annotations"]
)

print("Annotations per category:")
print(category_counts)


# Number of annotations per image
image_annotation_counts = Counter(
    annotation["image_id"]
    for annotation in data["annotations"]
)

single_damage = sum(
    1
    for count in image_annotation_counts.values()
    if count == 1
)

multiple_damage = sum(
    1
    for count in image_annotation_counts.values()
    if count > 1
)

print("Images with one annotation:", single_damage)
print("Images with multiple annotations:", multiple_damage)


# Damage types per image
image_categories = {}

for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    if image_id not in image_categories:
        image_categories[image_id] = set()

    image_categories[image_id].add(category_id)


multi_label_images = sum(
    1
    for categories in image_categories.values()
    if len(categories) > 1
)

print("Images with multiple damage types:", multi_label_images)


# Number of different damage types per image
damage_type_counts = Counter(
    len(categories)
    for categories in image_categories.values()
)

print("Damage types per image:", damage_type_counts)
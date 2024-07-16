#
# create_test_dataset.py: utility script to create test dataset
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#


def do():
    import fiftyone
    import degirum as dg
    import os, json, shutil

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    in_root = cur_dir + "/sample_dataset_in/"
    out_root = cur_dir + "/sample_dataset/"

    dataset = fiftyone.zoo.load_zoo_dataset(
        "coco-2017",
        label_types=["detections", "segmentations"],
        split="validation",
        max_samples=200,
    )
    dataset.export(
        export_dir=in_root,
        dataset_type=fiftyone.types.COCODetectionDataset,
        label_field="segmentations",
        export_media=True,
    )

    model_name = "mobilenet_v2_ssd_coco--300x300_quant_n2x_cpu_1"
    model_path = f"{cur_dir}/model-zoo/{model_name}/{model_name}.json"
    zoo = dg.connect(dg.LOCAL, model_path)
    model = zoo.load_model(model_name)
    dg_id_to_label_map = model.label_dictionary
    dg_label_to_id_map = {v: k for k, v in dg_id_to_label_map.items()}

    dataset_annotations = json.load(open(in_root + "labels.json"))

    # create dataset to model class id mapping
    class_map = {}
    not_found_labels = []
    for c in dataset_annotations["categories"]:
        lbl = c["name"]
        if lbl not in dg_label_to_id_map:
            not_found_labels.append(lbl)
        class_map[c["id"]] = dg_label_to_id_map[lbl]

    if not_found_labels:
        raise Exception(
            f"The following labels are not found in model label dictionary:/n{not_found_labels}"
        )

    #
    # patch dataset class ids to match model class ids
    #

    # patch annotations
    for a in dataset_annotations["annotations"]:
        a["category_id"] = class_map[a["category_id"]]

    # patch categories
    for c in dataset_annotations["categories"]:
        id = class_map[c["id"]]
        c["id"] = id
        c["name"] = dg_id_to_label_map[id]

    # patch images and copy them to the output subdirs
    for im in dataset_annotations["images"][:]:
        fn = im["file_name"]

        image_annotations = [
            a for a in dataset_annotations["annotations"] if a["image_id"] == im["id"]
        ]
        if not image_annotations:
            print("No annotations for", fn)
            dataset_annotations["images"].remove(im)
            continue

        # we select the object with the largest area and use it as main label
        max_area_object = max(image_annotations, key=lambda x: x["area"])

        label = dg_id_to_label_map[max_area_object["category_id"]]
        dir = out_root + label + "/"

        if not os.path.exists(dir):
            os.makedirs(dir)

        im["file_name"] = f"{label}/{fn}"
        shutil.copy(in_root + "data/" + fn, dir)

    dataset_annotations["info"] = {}
    json.dump(dataset_annotations, open(out_root + "labels.json", "w"), indent=4)

    shutil.rmtree(in_root)


if __name__ == "__main__":
    do()

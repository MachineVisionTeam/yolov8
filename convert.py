import json
import os
import shutil

def convert_to_yolo(input_images_path, input_json_path, output_images_path, output_labels_path):
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create directories for output images and labels
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # List to store filenames
    file_names = []
    for image_info in coco_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        image_path = os.path.join(input_images_path, file_name)
        shutil.copy(image_path, os.path.join(output_images_path, file_name))
        file_names.append(file_name)

        label_path = os.path.join(output_labels_path, file_name.replace('.png', '.txt'))

        with open(label_path, 'w') as label_file:
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    category_id = annotation['category_id']
                    x, y, w, h = annotation['bbox']

                    # Normalize coordinates
                    x_center = x / width
                    y_center = y / height
                    width_normalized = w / width
                    height_normalized = h / height

                    label_file.write(f"{category_id} {x_center} {y_center} {width_normalized} {height_normalized}\n")


if __name__ == "__main__":
    base_input_path = r"C:\Users\ADMIN\Desktop\Detectron\nucleus"
    base_output_path = r"C:\Users\ADMIN\Desktop\yolov8\nucleus"

    # Processing validation dataset (if needed)
    convert_to_yolo(
        input_images_path=os.path.join(base_input_path, "val/images"),
        input_json_path=os.path.join(base_input_path, "val/nuc_coco_val.json"),
        output_images_path=os.path.join(base_output_path, "val/images"),
        output_labels_path=os.path.join(base_output_path, "val/labels")
    )

    # Processing training dataset 
    convert_to_yolo(
        input_images_path=os.path.join(base_input_path, "train/images"),
        input_json_path=os.path.join(base_input_path, "train/nuc_coco_train.json"),
        output_images_path=os.path.join(base_output_path, "train/images"),
        output_labels_path=os.path.join(base_output_path, "train/labels")
    )

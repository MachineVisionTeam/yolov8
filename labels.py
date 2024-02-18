import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def display_images_with_annotations(image_paths, annotation_paths):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for ax, img_path, ann_path in zip(axs.ravel(), image_paths, annotation_paths):
        # Load image using OpenCV and convert it from BGR to RGB color space
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape

        ax.imshow(image)
        ax.axis('off')  # Turn off the axes

        # Open the annotation file and process each line
        with open(ann_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Convert normalized coordinates back to image coordinates
                x = int((x_center - width / 2) * img_w)
                y = int((y_center - height / 2) * img_h)
                w = int(width * img_w)
                h = int(height * img_h)

                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

# Example usage
image_dir = r"C:\Users\ADMIN\Desktop\yolov8\nucleus\train\images"
annotation_dir = r"C:\Users\ADMIN\Desktop\yolov8\nucleus\train\labels"
all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
random_image_files = random.sample(all_image_files, 4)

# Get corresponding annotation files
image_paths = [os.path.join(image_dir, f) for f in random_image_files]
annotation_paths = [os.path.join(annotation_dir, f.replace('.png', '.txt')) for f in random_image_files]

display_images_with_annotations(image_paths, annotation_paths)

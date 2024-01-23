import os
import cv2
import numpy as np
import pandas as pd

# Path to the original dataset folder containing fire images
dataset_path = 'data/fire'

# Path to the folder where the new data with bounding boxes will be saved
output_path = 'data/bou'

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load the bounding box annotations (assuming a CSV file with columns: image_path, x_min, y_min, x_max, y_max, label)
annotations_path = 'data/annotations.csv'
annotations_df = pd.read_csv(annotations_path)

# Loop through each annotation
for index, row in annotations_df.iterrows():
    image_path = os.path.join(dataset_path, row['image_path'])
    x_min = row['x_min']
    y_min = row['y_min']
    x_max = row['x_max']
    y_max = row['y_max']
    label = row['label']
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Draw the bounding box on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Save the image with the bounding box
    output_image_path = os.path.join(output_path, f'bbox_{index}.jpg')
    cv2.imwrite(output_image_path, image)
    
    # Save the bounding box information to a new annotation file (optional)
    output_annotation_path = os.path.join(output_path, 'annotations.csv')
    output_row = [output_image_path, x_min, y_min, x_max, y_max, label]
    with open(output_annotation_path, 'a') as f:
        f.write(','.join(map(str, output_row)) + '\n')

print('Bounding boxes generated and saved successfully.')

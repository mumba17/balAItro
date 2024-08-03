import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define the paths
image_folder_path = 'templates/'
csv_file_path = 'template_labels.csv'

# Load the labeled data from the CSV file
labels_df = pd.read_csv(csv_file_path)

# Function to find the best matching template for a given image
def find_best_match(image_path, template_paths):
    # Read the main image
    main_image = cv2.imread(image_path)
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    
    best_match_value = -1
    best_match_label = None
    
    # Loop over all template images
    for template_path in template_paths:
        # Read the template image
        template_image = cv2.imread(template_path)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Get the best match value
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Update the best match if the current one is better
        if max_val > best_match_value:
            best_match_value = max_val
            best_match_label = labels_df[labels_df['Image ID'] == int(os.path.splitext(os.path.basename(template_path))[0])]['Label'].values[0]
    
    return best_match_label, best_match_value

# List of non-labeled image paths
non_labeled_images = [os.path.join(image_folder_path, f'{i}.png') for i in range(10, 55)]

# List of labeled image paths
labeled_images = [os.path.join(image_folder_path, f'{i}.png') for i in range(55, 98)]

# DataFrame to store the new labels
new_labels = []

# Loop over all non-labeled images and find the best matching template
for image_path in non_labeled_images:
    best_match_label, best_match_value = find_best_match(image_path, labeled_images)
    image_id = int(os.path.splitext(os.path.basename(image_path))[0])
    
    if best_match_value >= 0.7:  # Threshold for a match
        new_labels.append([image_id, best_match_label])
    else:
        print(f'No good match for image {image_id}. Please assign manually.')

# Append the new labels to the CSV file
new_labels_df = pd.DataFrame(new_labels, columns=['Image ID', 'Label'])
new_labels_df.to_csv('updated_labels.csv', index=False)

print('Labeling completed and saved to updated_labels.csv')

import cv2
import numpy as np
from PIL import ImageGrab
import os
from pynput import keyboard
import glob
import re
import json
import csv

# Directory to save the screenshots
screenshot_dir = "screenshots"

# File to store image counts
count_file = "image_counts.json"

# CSV file to store image information
csv_file = "image_labels.csv"

# Define the base template names and their meanings
base_templates = {
    "blind_play_template": "Blind Play Screen",
    "blind_reward_template": "Blind Reward Screen",
    "blind_select_template": "Blind Selection Screen",
    "pack_large_template": "Large Pack Screen",
    "pack_small_template": "Small Pack Screen",
    "shop_template": "Shop Screen"
}

def load_image_counts():
    if os.path.exists(count_file):
        with open(count_file, 'r') as f:
            return json.load(f)
    return {base: 0 for base in base_templates}

def save_image_counts():
    with open(count_file, 'w') as f:
        json.dump(image_counts, f)

def get_next_image_id():
    return max([int(f.split('.')[0]) for f in os.listdir(screenshot_dir) if f.split('.')[0].isdigit()], default=0) + 1

def save_image_info(image_id, label):
    filepath = os.path.join(screenshot_dir, f"{image_id}.png")
    label_meaning = base_templates[label]
    
    # Append to CSV file
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_id, label, label_meaning])

# Initialize image_counts from file
image_counts = load_image_counts()

# Initialize CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image ID', 'Label', 'Label Meaning'])

# Initialize an empty list to hold template file paths
templates = []

# Search for files matching the pattern and append them to the templates list
for base_template in base_templates:
    template_files = glob.glob(f"templates/{base_template}_*.png")
    templates.extend(template_files)

print("Templates found:", templates)

# Initialize dictionaries to store average images and counts
average_images = {base: None for base in base_templates}

def update_average(base_template, new_image):
    if average_images[base_template] is None:
        average_images[base_template] = new_image.astype(float)
    else:
        average_images[base_template] = (average_images[base_template] * image_counts[base_template] + new_image) / (image_counts[base_template] + 1)
    image_counts[base_template] += 1
    save_image_counts()
    
    # Save the updated average image
    avg_image = average_images[base_template].astype(np.uint8)
    cv2.imwrite(f"averages/{base_template}_average.png", avg_image)

def process_screenshot():
    screenshot = ImageGrab.grab()
    image_id = get_next_image_id()
    screenshot_path = os.path.join(screenshot_dir, f"{image_id}.png")
    screenshot.save(screenshot_path)

    main_image = cv2.imread(screenshot_path)
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_val = -1
    best_template = None

    for template in templates:
        template_image = cv2.imread(template)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val = max_val
            best_match = max_loc
            best_template = template

    if best_val >= 0.7:  # Adjust threshold as needed
        print(f"Best match found for template: {best_template} with value: {best_val}")

        base_template = next(base for base in base_templates if base in best_template)
        update_average(base_template, main_image)
        save_image_info(image_id, base_template)
    else:
        print("No matches found.")
        print("1: blind_play_template")
        print("2: blind_reward_template")
        print("3: blind_select_template")
        print("4: pack_large_template")
        print("5: pack_small_template")
        print("6: shop_template")
        
        user_input = input("Please assign a base template number (e.g., 1 for blind_play_template): ")
        if user_input in [str(i) for i in range(1, 7)]:
            base_template = list(base_templates.keys())[int(user_input) - 1]
            update_average(base_template, main_image)
            save_image_info(image_id, base_template)
        else:
            print("Invalid input. Screenshot not saved.")
            os.remove(screenshot_path)

def on_press(key):
    try:
        if key.char == '`':
            process_screenshot()
    except AttributeError:
        pass

# Create 'averages' directory if it doesn't exist
if not os.path.exists('averages'):
    os.makedirs('averages')

# Set up the keyboard listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
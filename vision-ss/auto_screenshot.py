import cv2
import numpy as np
from PIL import ImageGrab
import os
from pynput import keyboard
import json
import csv

# Directories
screenshot_dir = "screenshots"
template_dir = "templates"

# File to store image counts
count_file = "image_counts.json"

# CSV files to store image information
csv_file_template = "template_labels.csv"
csv_file_screenshots = "screenshot_labels.csv"

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
    existing_ids = [int(f.split('.')[0]) for f in os.listdir(screenshot_dir) if f.split('.')[0].isdigit()]
    template_ids = []
    screenshot_ids = []
    
    with open(csv_file_template, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        template_ids = [int(row[0]) for row in reader]
    
    with open(csv_file_screenshots, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        screenshot_ids = [int(row[0]) for row in reader]
    
    all_ids = existing_ids + template_ids + screenshot_ids
    return max(all_ids, default=0) + 1

def save_image_info(image_id, label):
    filepath = os.path.join(screenshot_dir, f"{image_id}.png")
    
    # Append to CSV file
    with open(csv_file_screenshots, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_id, label])

def load_templates():
    templates = []
    with open(csv_file_template, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            image_id, label = row
            template_path = os.path.join(template_dir, f"{image_id}.png")
            if os.path.exists(template_path):
                templates.append((label, template_path))
    return templates

# Initialize image_counts from file
image_counts = load_image_counts()

# Initialize CSV file if it doesn't exist
if not os.path.exists(csv_file_screenshots):
    with open(csv_file_screenshots, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image ID', 'Label'])

# Load templates from CSV
templates = load_templates()
print("Templates loaded:", templates)

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

    for label, template_path in templates:
        template_image = cv2.imread(template_path)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val = max_val
            best_match = max_loc
            best_template = label
            best_path = template_path

    if best_val >= 0.7:  # Adjust threshold as needed
        print(f"Best match found for template: {best_template} with value: {best_val}, file: {best_path}")

        update_average(best_template, main_image)
        save_image_info(image_id, best_template)
    else:
        print("No matches found.")
        for i, (label, _) in enumerate(templates, 1):
            print(f"{i}: {label}")
        
        user_input = input("Please assign a template number: ")
        if user_input.isdigit() and 1 <= int(user_input) <= len(templates):
            base_template = templates[int(user_input) - 1][0]
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
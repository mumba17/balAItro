import os
import csv
import shutil
import re
import glob

# Directories
screenshot_dir = "screenshots"
templates_dir = "templates"

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

def get_label_from_filename(filename):
    for base in base_templates.keys():
        if base.replace("_template", "") in filename:
            return base
    return None

def get_next_id(csv_file):
    if not os.path.exists(csv_file):
        return 1
    with open(csv_file, 'r') as f:
        return sum(1 for line in f) # This counts the header too, so it's perfect for next ID

def convert_and_add_images(source_dir, start_id, is_template=False):
    image_files = glob.glob(os.path.join(source_dir, "*.png"))
    image_files.sort()

    for i, old_path in enumerate(image_files, start=start_id):
        old_filename = os.path.basename(old_path)
        new_filename = f"{i}.png"
        new_path = os.path.join(screenshot_dir, new_filename)

        # Copy for templates, move for screenshots
        if is_template:
            shutil.copy2(old_path, new_path)
        else:
            shutil.move(old_path, new_path)

        # Get the label from the old filename
        label = get_label_from_filename(old_filename)

        if label:
            label_meaning = base_templates[label]

            # Append to CSV file
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i, label, label_meaning])

            print(f"{'Added' if is_template else 'Converted'} {old_filename} to {new_filename} - Label: {label}")
        else:
            print(f"{'Added' if is_template else 'Converted'} {old_filename} to {new_filename} - No label found")

    return i + 1  # Return the next available ID

def convert_existing_images():
    # Create a new CSV file or clear existing one
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image ID', 'Label', 'Label Meaning'])

    # Convert screenshots
    next_id = convert_and_add_images(screenshot_dir, 1)

    # Add templates
    convert_and_add_images(templates_dir, next_id, is_template=True)

    print(f"Conversion complete. CSV file updated: {csv_file}")

if __name__ == "__main__":
    convert_existing_images()
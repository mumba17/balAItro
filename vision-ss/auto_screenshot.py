import cv2
import numpy as np
from PIL import ImageGrab
import os
from pynput import mouse

# Directory to save the screenshots
screenshot_dir = "screenshots"

def get_next_number(base_name):
    files = os.listdir(screenshot_dir)
    numbers = [int(f.split('_')[-1].split('.')[0]) for f in files if f.startswith(base_name) and f.split('_')[-1].split('.')[0].isdigit()]
    return max(numbers) + 1 if numbers else 1

templates = ["templates/blind_play_template.png", "templates/blind_reward_template.png", 
             "templates/blind_select_template.png", "templates/pack_large_template.png", 
             "templates/pack_small_template.png", "templates/shop_template.png"]

def process_screenshot():
    screenshot = ImageGrab.grab()
    screenshot_path = os.path.join(screenshot_dir, "temp.png")
    screenshot.save(screenshot_path)

    for template in templates:
        # Read the images
        main_image = cv2.imread(screenshot_path)
        template_image = cv2.imread(template)

        # Convert images to grayscale
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Get the dimensions of the template
        w, h = template_gray.shape[::-1]

        # Perform template matching
        res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Set a threshold to detect the match
        threshold = 0.7
        loc = np.where(res >= threshold)
        
        # Check if a match is found
        if np.where(res >= threshold):
            print(f"Match found for template: {template}")
            
            # Determine the base name for the template
            base_name = os.path.splitext(os.path.basename(template))[0]
            
            # Get the next number for the screenshot file
            next_number = get_next_number(base_name)
            
            # Rename the temp screenshot to the new name
            new_filename = f"{base_name}_{next_number}.png"
            new_filepath = os.path.join(screenshot_dir, new_filename)
            os.rename(screenshot_path, new_filepath)
            
            # Save the screenshot with the new name
            screenshot.save(new_filepath)
            
            # Exit after the first match (remove this if you want to check for multiple templates)
            break
    else:
        print("No matches found.")

def on_click(x, y, button, pressed):
    if pressed:
        process_screenshot()

# Set up the mouse listener
with mouse.Listener(on_click=on_click) as listener:
    listener.join()

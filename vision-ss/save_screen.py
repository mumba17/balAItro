import os
import pyautogui
from pynput import keyboard
from PIL import ImageGrab

# Directory to save screenshots
screenshot_dir = 'screenshots'

# Create the directory if it doesn't exist
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# Function to get the next available number for screenshots
def get_next_number(base_name):
    files = os.listdir(screenshot_dir)
    numbers = [int(f.split('_')[-1].split('.')[0]) for f in files if f.startswith(base_name) and f.split('_')[-1].split('.')[0].isdigit()]
    return max(numbers) + 1 if numbers else 1

# Function to take a screenshot with the given base name
def take_screenshot(base_name):
    next_number = get_next_number(base_name)
    screenshot_path = os.path.join(screenshot_dir, f"{base_name}_{next_number}.png")
    screenshot = ImageGrab.grab()
    screenshot.save(screenshot_path)
    print(f"Screenshot saved as {screenshot_path}")

# Define the key press handler
def on_press(key):
    try:
        if key.char == '1':
            take_screenshot('shop')
        elif key.char == '2':
            take_screenshot('blind_select')
        elif key.char == '3':
            take_screenshot('blind_reward')
        elif key.char == '4':
            take_screenshot('blind_play')
        elif key.char == '5':
            take_screenshot('pack_small')
        elif key.char == '6':
            take_screenshot('pack_large')
        elif key.char == 'q':
            #Close the program
            print("Exiting...")
            os._exit(0)
    except AttributeError:
        pass

# Create a keyboard listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

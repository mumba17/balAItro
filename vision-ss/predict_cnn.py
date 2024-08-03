import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageGrab
from cnn_classifier import CNN
from pynput import keyboard

def load_model(model_path, num_classes, device):
    model = CNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_single_image(model, device, image):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((160, 90)),
        transforms.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        print(output)
        pred = output.argmax(dim=1, keepdim=True).item()
    return pred

def main():
    global screenshot, predicted_class
    # Example image (replace this with your PIL image object extracted from the screen)
    screenshot = ImageGrab.grab()  # Take a screenshot
    screenshot.save("temp.png")
    image = Image.open("temp.png").convert('L')  # Make sure to convert to grayscale

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    class_labels = ['blind_reward_template', 'blind_play_template', 'blind_select_template', 'pack_large_template', 'pack_small_template', 'shop_template', 'other']  # Replace with your actual class labels
    num_classes = len(class_labels)
    model = load_model('models/cnn_classifier.pth', num_classes, device)

    # Predict on the single image
    predicted_class_idx = predict_single_image(model, device, image)
    predicted_class = class_labels[predicted_class_idx]

    print(f"Predicted class: {predicted_class}, index: {predicted_class_idx}")
    print("Press 'A' to accept and add to CSV, or 'D' to manually classify.")
    
def add_to_csv(template_label):
    # Load the CSV file
    if os.path.exists('template_labels.csv'):
        df = pd.read_csv('template_labels.csv')
        
        # Find the last valid index, or start from 0 if all are NaN
        last_valid_index = df['Image ID'].last_valid_index()
        if last_valid_index is not None:
            new_index = int(df.loc[last_valid_index, 'Image ID']) + 1
        else:
            new_index = 1
    else:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=['Image ID', 'Label'])
        new_index = 1
    
    # Save the screenshot with the new index
    screenshot.save(f"templates/{new_index}.png")
    
    # Add new row to the dataframe
    new_row = pd.DataFrame({'Image ID': [new_index], 'Label': [template_label]})
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated dataframe back to CSV
    df.to_csv('template_labels.csv', index=False)
    
    print(f"Added image {new_index} with label '{template_label}' to CSV.")

def on_press(key):
    global predicted_class
    try:
        if key.char == '`':
            main()
        elif key.char == 'A':
            print("ADDING TO CSV")
            add_to_csv(predicted_class)
        elif key.char == 'D':
            print("Manual classification mode. Press 1-6 to classify:")
            print("1: shop, 2: blind_select, 3: blind_reward, 4: blind_play, 5: pack_small, 6: pack_large, 7: other")
        elif key.char in ['1', '2', '3', '4', '5', '6', '7']:
            labels = ['shop_template', 'blind_select_template', 'blind_reward_template', 'blind_play_template', 'pack_small_template', 'pack_large_template', 'other']
            add_to_csv(labels[int(key.char) - 1])
        elif key.char == 'q':
            # Stop the listener
            return False
    except AttributeError:
        pass

if __name__ == '__main__':
    with keyboard.Listener(on_press=on_press) as listener:
        print("Press '`' to take a screenshot and classify.")
        print("Press 'q' to quit.")
        listener.join()
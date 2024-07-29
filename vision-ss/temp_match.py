import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the main image and the template image
main_image_path = 'screenshots/blind_play_5.png'  # Path to the main image
template_image_path = 'templates/blind_play_template.png'   # Path to the template image

# Read the images
main_image = cv2.imread(main_image_path)
template_image = cv2.imread(template_image_path)

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

# Draw a rectangle around the matched region
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w-10, pt[1] + h), (0, 255, 0), 10)

# Display the result
plt.imshow(cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Template')
plt.show()

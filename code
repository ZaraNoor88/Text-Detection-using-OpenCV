import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

image_path = '/content/test1.png'
img = cv2.imread(image_path)
reader = easyocr.Reader(['en'], gpu=False)
text_ = reader.readtext(img)

threshold = 0.25

for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        # Convert float coordinates to integers before passing to cv2 functions
        pt1 = (int(bbox[0][0]), int(bbox[0][1]))
        pt2 = (int(bbox[2][0]), int(bbox[2][1]))
        cv2.rectangle(img, pt1, pt2, (0, 225, 0), 2)
        cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)

original_img = cv2.imread(image_path)
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Create a figure with two subplots
plt.figure(figsize=(15, 8))

# Subplot 1: Original Image
plt.subplot(1, 2, 1)
plt.imshow(original_img_rgb)
plt.title('Original Image')
plt.axis('off')

# Subplot 2: Text Detected Image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Text Detected Image')
plt.axis('off')

plt.show()

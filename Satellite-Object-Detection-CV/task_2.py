import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

image = cv2.imread("kpi_campus.png")
output_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)

morph_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

morph_closed = cv2.morphologyEx(morph_opened, cv2.MORPH_CLOSE, kernel, iterations=1)

contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

buildings = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    if area > 300 and 0.2 < aspect_ratio < 5.0:
        buildings.append((x, y, w, h))
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Threshold (Original)")
plt.imshow(thresh, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Morphological Opening (Noise removal)")
plt.imshow(morph_opened, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Morphological Closing (Hole filling)")
plt.imshow(morph_closed, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title(f"Виявлені будівлі: {len(buildings)}")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
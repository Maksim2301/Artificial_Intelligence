import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Завантаження зображення
image = cv2.imread("kpi_campus.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Бінаризація (Отсу) ---
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Пошук контурів ---
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Фільтрація контурів ---
buildings = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    if area > 300 and 0.2 < aspect_ratio < 5.0:
        buildings.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Gray")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Threshold (Otsu)")
plt.imshow(thresh, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Detected Buildings: {len(buildings)}")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()

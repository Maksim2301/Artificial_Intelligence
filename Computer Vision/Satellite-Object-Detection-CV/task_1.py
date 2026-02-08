import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

img1 = cv2.imread('img1_1.png')
img2 = cv2.imread('img2_2.png')

img1 = cv2.resize(img1, (600, 400))
img2 = cv2.resize(img2, (600, 400))

# Попередня обробка 
# Використовуємо GaussianBlur + unsharp mask для покращення різкості
img1_blur = cv2.GaussianBlur(img1, (5,5), 0)
img2_blur = cv2.GaussianBlur(img2, (5,5), 0)

# Посилення контрасту
sharp_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img1_sharp = cv2.filter2D(img1_blur, -1, sharp_kernel)
img2_sharp = cv2.filter2D(img2_blur, -1, sharp_kernel)

# вирівнювання яскравості
img1_yuv = cv2.cvtColor(img1_blur, cv2.COLOR_BGR2YUV)
img1_yuv[:,:,0] = cv2.equalizeHist(img1_yuv[:,:,0])
img1_corr = cv2.cvtColor(img1_yuv, cv2.COLOR_YUV2BGR)

img2_yuv = cv2.cvtColor(img2_blur, cv2.COLOR_BGR2YUV)
img2_yuv[:,:,0] = cv2.equalizeHist(img2_yuv[:,:,0])
img2_corr = cv2.cvtColor(img2_yuv, cv2.COLOR_YUV2BGR)

plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img1_corr, cv2.COLOR_BGR2RGB)), plt.title('Оброблене зображення 1')
plt.subplot(122), plt.imshow(cv2.cvtColor(img2_corr, cv2.COLOR_BGR2RGB)), plt.title('Оброблене зображення 2')
plt.show()

# Виділення ознак ORB (feature detection) 
orb = cv2.ORB_create(nfeatures=1000)

# Знаходження ключових точок та дескрипторів
kp1, des1 = orb.detectAndCompute(img1_corr, None)
kp2, des2 = orb.detectAndCompute(img2_corr, None)

# Візуалізація знайдених точок
img_kp1 = cv2.drawKeypoints(img1_corr, kp1, None, color=(0,255,0))
img_kp2 = cv2.drawKeypoints(img2_corr, kp2, None, color=(0,255,0))

plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img_kp1, cv2.COLOR_BGR2RGB)), plt.title('Ознаки зображення 1')
plt.subplot(122), plt.imshow(cv2.cvtColor(img_kp2, cv2.COLOR_BGR2RGB)), plt.title('Ознаки зображення 2')
plt.show()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1_corr, kp1, img2_corr, kp2, matches[:50], None, flags=2)

plt.figure(figsize=(15,7))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Порівняння ознак між зображеннями')
plt.show()




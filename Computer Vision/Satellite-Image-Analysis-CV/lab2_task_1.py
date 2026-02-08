import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ------------------- Корекція кольору -------------------
def color_corrections(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    negative = cv2.bitwise_not(img)

    # ефект сепії
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255)

    return gray, negative, sepia


# ------------------- Векторизація -------------------
def vectorization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 10, 50)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    gabor_accumulator = np.zeros_like(gray, dtype=np.float64)

    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 градусів

    for theta in orientations:
        g_kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)

        filtered_img = cv2.filter2D(gray, cv2.CV_32F, g_kernel)

        np.add(gabor_accumulator, abs(filtered_img), out=gabor_accumulator)

    gabor_accumulator = gabor_accumulator / len(orientations)

    gabor = cv2.normalize(gabor_accumulator, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return canny, contour_img, gabor


# ------------------- Порівняння ознак -------------------
def compare_features(img1, img2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)
    print("Кількість збігів:", len(matches))

    return result

if __name__ == "__main__":
    img = cv2.imread("img1.png")

    # --- Корекція кольору ---
    gray, negative, sepia = color_corrections(img)

    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)), plt.title("Grayscale")
    plt.subplot(132), plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)), plt.title("Negative")
    plt.subplot(133), plt.imshow(cv2.cvtColor(sepia.astype('uint8'), cv2.COLOR_BGR2RGB)), plt.title("Sepia")
    plt.show()

    # --- Векторизація ---
    canny, contour_img, gabor = vectorization(img)

    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(canny, cmap="gray"), plt.title("Canny Edges")
    plt.subplot(132), plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)), plt.title("Contours")
    plt.subplot(133), plt.imshow(gabor, cmap="gray"), plt.title("Gabor Filter")
    plt.show()

    # --- Порівняння з іншим зображенням ---
    img2 = cv2.imread("img2.png")
    comparison = compare_features(gray, cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title("Feature Matching (ORB)")
    plt.show()


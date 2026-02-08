# Task 1
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

digits = load_digits()

figure1, axes1 = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
for ax, image, label in zip(axes1.ravel(), digits.images[:24], digits.target[:24]):
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label)
plt.tight_layout()
plt.show()

figure2, axes2 = plt.subplots(nrows=6, ncols=6, figsize=(8, 8))
for ax, image, label in zip(axes2.ravel(), digits.images[:36], digits.target[:36]):
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label)
plt.tight_layout()
plt.show()

# Task 2
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=11, test_size=0.20)

print("Розмір навчальних даних:", X_train.shape)
print("Розмір тестових даних:", X_test.shape)

# Task 3
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

# Task 4
predicted = knn.predict(X=X_test)
expected = y_test

# Task 5
print("Прогнозовані значення:", predicted[:20])
print("Очікувані значення:", expected[:20])

# Task 6.1
print(f'Точність моделі: {knn.score(X_test, y_test):.2%}')
# Task 6.2
confusion = confusion_matrix(y_true=y_test, y_pred=predicted)
print("Матриця невідповідностей:")
print(confusion)

# Task 7
names = [str(digit) for digit in digits.target_names]
print(classification_report(expected, predicted, target_names=names))

# Task 8
models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(),
    "GaussianNB": GaussianNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: Точність {score:.2%}")

# Task 9
print("\nПідбір найкращого K:")

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print(f"K = {k}: Точність = {score:.2%}")

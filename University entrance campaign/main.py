import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import BinaryCrossentropy

# 1. Генерація та підготовка даних
np.random.seed(42)
num_applicants = 1500

applicants = pd.DataFrame({
    'Math': np.random.randint(100, 201, num_applicants),
    'English': np.random.randint(100, 201, num_applicants),
    'Ukrainian': np.random.randint(100, 201, num_applicants),
    'Benefit': np.random.choice([0, 1], num_applicants, p=[0.9, 0.1])
})

applicants['Rating'] = (
    0.4 * applicants['Math'] +
    0.3 * applicants['English'] +
    0.3 * applicants['Ukrainian']
)

applicants['Admitted'] = applicants.apply(
    lambda row: int(
        (row['Math'] >= 120 and row['English'] >= 120 and row['Ukrainian'] >= 120 and row['Rating'] >= 144)
        if row['Benefit']
        else (row['Math'] >= 140 and row['Rating'] >= 160)
    ), axis=1
)

print("Баланс класів у 'Admitted':")
print(applicants['Admitted'].value_counts())

# 2. Масштабування та розбиття
X = applicants[['Math', 'English', 'Ukrainian', 'Benefit']]
y = applicants['Admitted']
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# 3. Навчання моделей з різними архітектурами
architectures = [(4,), (8,), (16,), (8, 4), (16, 8), (32, 16)]
results = []

for arch in architectures:
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    for units in arch:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adadelta(1.0), loss=BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=0)
    y_pred = (model.predict(X_test).ravel() >= 0.5).astype(int)
    results.append({
        'architecture': arch,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'model': model
    })

# 4. Вивід результатів
print("\nРезультати по архітектурах:")
for r in results:
    if r['accuracy'] >= 0.85:  # Наприклад, така ціль
        print(f"Задовільна модель: {r['architecture']} з accuracy {r['accuracy']:.4f}")
# 5. Вибір найкращої моделі
best = max(results, key=lambda x: x['accuracy'])

# 6. Прогнозування
applicants['Predicted'] = (best['model'].predict(X_scaled).ravel() >= 0.5).astype(int)
print("\nРозподіл передбачених класів:")
print(applicants['Predicted'].value_counts())

# 7. Збереження
sorted_applicants = applicants.sort_values(by='Rating', ascending=False)
benefit = sorted_applicants[sorted_applicants['Benefit'] == 1]
non_benefit = sorted_applicants[sorted_applicants['Benefit'] == 0]
max_benefit = min(35, len(benefit))
top_benefit = benefit.head(max_benefit)
top_non_benefit = non_benefit.head(350 - len(top_benefit))

final_selected = pd.concat([top_benefit, top_non_benefit]).sort_values(by='Rating', ascending=False)

final_selected.to_excel('admitted_students.xlsx', index=False)
print("Результати збережено у файлі 'admitted_students.xlsx'")


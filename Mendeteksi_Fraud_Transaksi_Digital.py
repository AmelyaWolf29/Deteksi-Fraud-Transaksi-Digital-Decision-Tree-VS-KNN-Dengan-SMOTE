import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, confusion_matrix, f1_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.tree import plot_tree

# 1. Load Dataset
df = pd.read_csv('creditcard.csv')

# 2. Cek distribusi kelas
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title('Distribusi Kelas (0 = Normal, 1 = Fraud)')
plt.show()

# 3. Preprocessing
x = df.drop('Class', axis=1)
y = df['Class']

# 4. Standarisasi fitur 'Amount' dan 'Time'
scaler = StandardScaler()
x[['Time', 'Amount']] = scaler.fit_transform(x[['Time', 'Amount']])

# 5. Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

# 6. Model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)

# 7. Model K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)

# 8. Evaluasi
print("==== Decision Tree ====")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("F1-Score:", f1_score(y_test, y_pred_dt))
ap_dt = average_precision_score(y_test, y_pred_dt)
print("Average Precision Score:", ap_dt)

print("\n==== K-Nearest Neighbors ====")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("F1-Score:", f1_score(y_test, y_pred_knn))
ap_knn = average_precision_score(y_test, y_pred_knn)
print("Average Precision Score:", ap_knn)

# 9. Precision dan Recall (Decision Tree)
precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_pred_dt)

plt.figure(figsize=(12,10))
plt.plot(recall_dt, precision_dt, label=f'Decision Tree (AP = {ap_dt:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Decision Tree)')
plt.legend()
plt.grid(True)
plt.show()

# 10. Precision dan Recall (K-NearestNeighbors)
precision_knn, recall_knn, _ = precision_recall_curve(y_test, y_pred_knn)

plt.figure(figsize=(12,10))
plt.plot(recall_knn, precision_knn, label=f'K-Nearest Neighbors (AP = {ap_knn:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (K-Nearest Neighbors)')
plt.legend()
plt.grid(True)
plt.show()

# 11. Visualisasi Pohon Keputusan
plt.figure(figsize=(20, 10))
plot_tree(dt_model, max_depth=2, feature_names=x.columns, class_names=["Normal", "Fraud"], filled=True)
plt.title("Visualisasi Decision Tree (Max Depth=2)")
plt.show()

# 12. Menangani Imbalances Dataset (dengan SMOTE) (Percobaan)
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

print("Setelah SMOTE:")
print(pd.Series(y_train_res).value_counts())

# 13. Model Decision Tree (Setelah SMOTE)
dt_model_res = DecisionTreeClassifier(random_state=42)
dt_model_res.fit(x_train_res, y_train_res)
y_pred_dt_res = dt_model_res.predict(x_test)

# 14. Model K-Nearest Neighbors (Setelah SMOTE)
knn_model_res = KNeighborsClassifier(n_neighbors=5)
knn_model_res.fit(x_train_res, y_train_res)
y_pred_knn_res = knn_model_res.predict(x_test)

# 15. Evaluasi(Setelah SMOTE)
print("==== Decision Tree (Setelah SMOTE) ====")
print(confusion_matrix(y_test, y_pred_dt_res))
print(classification_report(y_test, y_pred_dt_res))
print("Accuracy:", accuracy_score(y_test, y_pred_dt_res))
print("F1-Score:", f1_score(y_test, y_pred_dt_res))
ap_dt_res = average_precision_score(y_test, y_pred_dt_res)
print("Average Precision Score:", ap_dt_res)

print("\n==== K-Nearest Neighbors )(Setelah SMOTE) ====")
print(confusion_matrix(y_test, y_pred_knn_res))
print(classification_report(y_test, y_pred_knn_res))
print("Accuracy:", accuracy_score(y_test, y_pred_knn_res))
print("F1-Score:", f1_score(y_test, y_pred_knn_res))
ap_knn_res = average_precision_score(y_test, y_pred_knn_res)
print("Average Precision Score:", ap_knn_res)


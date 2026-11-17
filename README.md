# Deteksi Fraud Transaksi Digital: Decision Tree vs KNN dengan SMOTE
## Tujuan 
Membandingkan Decision Tree dan KNN untuk mendeteksi fraud pada 284.807 transaksi (492 fraud) dan menguji SMOTE untuk mengatasi ketidakseimbangan kelas. Setelah SMOTE, data latih berisi 227.451 transaksi per kelas.

## Dataset 
Credit Card (Kaggle): Time, V1-V28 adalah fitur anonim hasil transformasi PCA, Amount berisi nominal (jumlah), Class (0 = normal (284.315 transaksi), 1 = fraud (492 transaksi)).
Sumber: https://www.kaggle.com/datasets/jacklizhi/creditcard

## Metode
- Mengambil data dari dataset dengan variabel df menggunakan pandas
- Cek distribusi Kelas untuk menghitung berapa kelas 0 dan kelas 1.
- Preprocessing data dan melakukan standardisasi pada fitur Time dan Amount
- Split data 80:20 antara data train dan data test
- Model algoritma Decision Tree dan K-Nearest Neighbors dengan data train
- Evaluasi model (Confusion matrix, Accuracy, F1-Score, Average Precision)
- Visualisasi Precision dan Recall pada algoritma Decision Tree dan KNN
- Visualisasi Decision Tree dengan max_depth 2 
- Menangani Imbalance Dataset dengan SMOTE (SMOTE hanya dilakukan pada data latih), serta mencoba melakukan evaluasi setelah melakukan model SMOTE pada algoritma Decision Tree dan KNN

## Hasil
|        Model          | Precision | Recall |  F1  |  AP  | Accuracy |
|-----------------------|-----------|--------|------|------|----------|
|      Decision Tree    |    0,75   |  0,74  | 0,75 | 0,56 |  99,91%  |
|          KNN          |	 0,92   |  0,81  | 0,86 | 0,74 |  99,95%  |
| Decision Tree + SMOTE |    0,34   |  0,78  | 0,47 | 0,26 |  99,70%  |
|      KNN + SMOTE      |    0,46   |  0,88  | 0,61 | 0,41 |  99,81%  |

Data uji memakai 56.962 transaksi dengan 98 fraud

Visualisasi Decision Tree:
![Visualisasi DT](<Visualisasi_Decision_Tree.png>)

## Cara Menjalankan
Unduh creditcard.csv dari Kaggle dan letakkan di folder yang sama
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
python Mendeteksi_Fraud_Transaksi_Digital.py
```

## Catatan
- KNN terbaik (F1 0,86; AP 0,74 vs Decision Tree F1 0,75; AP 0,56). SMOTE menaikkan recall (KNN 0,81 -> 0,88) tetapi menurunkan precision (0,92 -> 0,46), sehingga F1 turun ke 0,61.
- AP dihitung dari prediksi kelas (bukan probabilitas), Scaler dilatih sebelum melakukan split data, KNN memakai k=5 (bawaan scikit-learn) dan tidak dituning, dan max_depth 2 hanya untuk visualisasi decision tree (pohon yang dilatih tidak dibatasi kedalamannya).
- Waktu berjalan KNN setelah SMOTE sekitar 3 menit (tergantung perangkat/ komputer).
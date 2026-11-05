# PEMS — Estimasi Emisi CO dan NOx Turbin Gas

**Predictive Emission Monitoring System (PEMS):** model yang memperkirakan emisi **NOx** (target utama) dan **CO** dari sensor operasi turbin dan kondisi lingkungan. PEMS dipakai sebagai cadangan ketika alat ukur kontinu (CEMS) bermasalah.

Mini Project 1 · Deep Learning for Electrical Engineering · Studi Kasus 4 (Energi)

**Dataset:** [Gas Turbine CO and NOx Emission Data Set (UCI #551)](https://archive.ics.uci.edu/dataset/551). Berisi 36.733 data rata-rata per jam (2011–2015) dengan 9 fitur dan 2 target (mg/m³).

---

## Struktur repositori

```
PEMS-GasTurbine-CO-NOx/
├── data/
│   ├── raw/                      # 5 file asli: gt_2011.csv … gt_2015.csv
│   └── processed/gt_clean.csv    # hasil Notebook 01 (digabung + dideduplikasi)
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb   # Tahap 1 — Preprocessing & EDA
│   ├── 02_model_ml_regresi.ipynb    # Tahap 2 — 6 model regresi × 3 split × NOx/CO
│   └── 03_mlp_ann.ipynb             # Tahap 3 — MLP/ANN + perbandingan dengan ML
├── src/pems_utils.py             # konstanta, pemuatan data, skema split, metrik, plot (dipakai bersama)
├── results/
│   ├── tables/                   # semua tabel hasil (CSV) untuk laporan
│   ├── figures/                  # semua grafik (PNG)
│   ├── preprocessing_config.json # keputusan Notebook 01
│   └── ml_decisions.json         # keputusan Notebook 02 (dipakai Notebook 03)
└── requirements.txt
```

Ketiga notebook sudah dijalankan dari awal sampai akhir, dan semua output tersimpan di dalam notebook.

## Cara menjalankan

**Lokal**
```bash
git clone https://github.com/<username>/PEMS-GasTurbine-CO-NOx.git
cd PEMS-GasTurbine-CO-NOx
pip install -r requirements.txt
jupyter notebook notebooks/
```

**Google Colab:** buka notebook dari GitHub di Colab, lalu jalankan sel berikut terlebih dahulu:
```python
!git clone https://github.com/<username>/PEMS-GasTurbine-CO-NOx.git
%cd PEMS-GasTurbine-CO-NOx/notebooks
```

Jalankan notebook **berurutan: 01 → 02 → 03**. Perkiraan waktu dengan CPU 2 core: 01 ± 1 menit, 02 ± 65–75 menit (SVR paling lama), 03 ± 10 menit.
`random_state = 42` dipakai di semua split dan model.

---

## Metodologi

| Tahap | Isi |
|---|---|
| **1. Preprocessing & EDA** | penggabungan 5 file + cek konsistensi · missing value & nilai invalid · duplikat · distribusi · *imbalance* target · outlier & rezim operasi · korelasi · analisis per tahun · perbandingan 5 cara penanganan nilai invalid · VIF (full vs tereduksi) · scaling (fit hanya di data latih) |
| **2. Model ML** | Linear · Polinomial + Ridge · SVR (RBF) · Decision Tree · Random Forest · Gradient Boosting (HistGradientBoosting). Untuk tiap (target, split): tuning `RandomizedSearchCV` → **5-fold CV (mean ± std)** → evaluasi sekali di data uji. Ditambah uji log target CO, full vs fitur tereduksi, diagnosa under/overfitting, dan permutation importance |
| **3. MLP / ANN** | 3 arsitektur (32 · 64-32 · 128-64-32) × 2 learning rate (0,001 · 0,01), optimizer **Adam**, **early stopping + L2**, kurva loss train vs validation, uji variasi L2, permutation importance, perbandingan dengan ML terbaik |
| **Split** | Acak 80:20 · Acak 70:30 · Per tahun (latih 2011–2013, uji 2014–2015) |
| **Metrik** | MAE, RMSE (mg/m³), R² · plot prediksi vs aktual & residual |

**Aturan anti-kebocoran data:**
- Semua keputusan berbasis data (penanganan invalid, VIF, tuning) hanya memakai data latih.
- Scaler dan imputer berada di dalam `Pipeline`.
- Model terbaik dipilih berdasarkan CV atau validation loss, bukan skor uji.

## Keputusan preprocessing (Notebook 01)

| Aspek | Temuan | Keputusan |
|---|---|---|
| Missing value | tidak ada | – |
| Duplikat | 7 baris identik di semua sensor (logger membeku) | dihapus → 36.726 baris |
| AT < 0 °C (62 baris) | valid (musim dingin) | dipertahankan |
| AH > 100% (478 baris) | invalid secara fisik (maks. 100,2%) | 5 cara dibandingkan (biarkan, clip, imputasi median, imputasi mean, drop) → **clip ke 100%** (kinerja setara, sesuai batas fisik) |
| Outlier IQR (TAT 13%, CO 7%) | merupakan rezim operasi (beban tinggi/parsial) | dipertahankan |
| *Imbalance* target | CO sangat menceng (skew 4,8); CO > 10 mg/m³ hanya 1,4% | uji `log(1 + CO)`; **tanpa SMOTE/SMOGN** (sampel sintetis tidak fisis & bocor di data runtut waktu) |
| Encoding | semua fitur numerik | tidak diperlukan |
| Multikolinearitas | VIF TEY 256, CDP 251 | VIF iteratif ≤ 10 → fitur tereduksi: AT, AP, AH, AFDP, TIT, TAT |
| Scaling | – | `StandardScaler` di dalam Pipeline (fit di data latih) |

## Hasil utama

### Model terbaik per target dan skema split (data uji)

| Target | Split | ML terbaik | MAE (mg/m³) | RMSE (mg/m³) | R² | MLP terbaik | RMSE (mg/m³) | R² |
|---|---|---|---|---|---|---|---|---|
| **NOx** | Acak 80:20 | Gradient Boosting | 2,69 | 4,06 | 0,875 | 128-64-32, lr 0,01 | **3,87** | **0,887** |
| **NOx** | Acak 70:30 | Gradient Boosting | 2,70 | 4,08 | 0,874 | 128-64-32, lr 0,01 | **3,98** | **0,879** |
| **NOx** | Per tahun | Gradient Boosting | 11,06 | 12,53 | −0,40 | 64-32, lr 0,01 | 14,47 | −0,87 |
| **CO** | Acak 80:20 | Random Forest | 0,48 | **1,05** | **0,788** | 64-32, lr 0,01 | 1,11 | 0,760 |
| **CO** | Acak 70:30 | Random Forest | 0,49 | **1,09** | **0,770** | 128-64-32, lr 0,001 | 1,15 | 0,746 |
| **CO** | Per tahun | Random Forest | 0,95 | 1,59 | 0,472 | 128-64-32, lr 0,001 | 1,59 | 0,476 |

Tabel lengkap untuk keenam model ada di `results/tables/02_hasil_semua_model_ml.csv`, dan untuk ke-36 konfigurasi MLP di `results/tables/03_hasil_semua_mlp.csv`.

### Ringkasan NOx (split acak 80:20, 5-fold CV mean ± std)

| Model | CV RMSE (mg/m³) | CV R² | Train R² | Test RMSE (mg/m³) | Test R² | Diagnosa |
|---|---|---|---|---|---|---|
| Linear Regression | 8,123 ± 0,061 | 0,520 ± 0,009 | 0,520 | 8,07 | 0,507 | underfitting |
| Polinomial + Ridge | 4,946 ± 0,083 | 0,822 ± 0,006 | 0,827 | 4,91 | 0,818 | fit baik |
| SVR (RBF) | 4,753 ± 0,065 | 0,836 ± 0,004 | 0,855 | 4,55 | 0,843 | fit baik |
| Decision Tree | 5,521 ± 0,076 | 0,778 ± 0,004 | 0,883 | 5,35 | 0,784 | overfitting |
| Random Forest | 4,202 ± 0,111 | 0,871 ± 0,005 | 0,982 | 4,07 | 0,875 | varians tinggi, tetap terbaik ke-2 |
| **Gradient Boosting** | **4,161 ± 0,084** | **0,874 ± 0,005** | 0,946 | **4,06** | **0,875** | terbaik |

![NOx terbaik](results/figures/02_NOX_80-20_terbaik.png)
![CO terbaik](results/figures/02_CO_80-20_terbaik.png)

### Temuan penting

1. **Split acak terlalu optimis.** Pada split per tahun, semua model NOx gagal (R² negatif) karena *concept drift*: model memprediksi NOx 2014–2015 rata-rata **± 10 mg/m³ terlalu tinggi** (−8 mg/m³ pada 2014, −13 mg/m³ pada 2015). Pergeseran ini sudah terdeteksi di EDA (SMD NOx ≈ −0,8), sedangkan fitur hampir tidak bergeser. Sebaliknya, CO masih dapat diprediksi (R² ± 0,5).
   ![split per tahun](results/figures/02_NOX_Tahun_terbaik.png)
2. **Transformasi log CO** menurunkan MAE di sebagian besar model, tetapi tidak menurunkan RMSE model terbaik. Pada split per tahun, transformasi ini bahkan membuat model yang berekstrapolasi (polinomial, SVR) meledak. Karena itu **target CO asli** dipilih.
3. **Full fitur > fitur tereduksi** untuk semua model. Fitur beban yang saling berkorelasi tetap membawa informasi non-linear.
4. **MLP** dengan 3 hidden layer sedikit mengungguli Gradient Boosting untuk NOx di split acak, tetapi Random Forest tetap terbaik untuk CO. Arsitektur 1 layer *underfit*, early stopping + L2 10⁻⁴ efektif, dan L2 10⁻² terlalu kuat sehingga *underfit*.
   ![kurva loss](results/figures/03_NOX_80-20_kurva_loss.png)
5. **Variabel proses paling berpengaruh** (permutation importance):
   - **NOx:** suhu ambien **AT** dan suhu masuk turbin **TIT**, sesuai mekanisme NOx termal.
   - **CO:** **TIT, TAT,** dan beban **TEY**, sesuai mekanisme pembakaran tidak sempurna pada beban rendah.
   ![importance](results/figures/03_permutation_importance_MLP_vs_ML.png)

### Rekomendasi untuk PEMS PLTGU
- **NOx:** Gradient Boosting (ringan & cepat) atau MLP 128-64-32 (sedikit lebih akurat). **CO:** Random Forest.
- **Kalibrasi ulang berkala** dengan data CEMS terbaru, minimal koreksi offset, karena hubungan proses → NOx bergeser antar periode.
- Prioritaskan keandalan sensor **TIT, TAT, dan AT**, karena ketiganya paling menentukan estimasi.
- Untuk pemantauan baku mutu CO, perhatikan bahwa model cenderung **meremehkan puncak CO > 15 mg/m³** karena sampel di rentang itu sangat sedikit.

---

## Referensi
- Kaya, H., Tüfekci, P., & Uzun, E. (2019). Predicting CO and NOx emissions from gas turbines: novel data and a benchmark PEMS. *Turkish Journal of Electrical Engineering & Computer Sciences*, 27, 4783–4796.
- Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). *Applied Linear Statistical Models* (5th ed.). McGraw-Hill. (ambang VIF 10)
- Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2010). *Multivariate Data Analysis* (7th ed.). Pearson.
- Branco, P., Torgo, L., & Ribeiro, R. P. (2016). A survey of predictive modeling on imbalanced domains. *ACM Computing Surveys*, 49(2), 1–50.
- Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., & Zeileis, A. (2008). Conditional variable importance for random forests. *BMC Bioinformatics*, 9, 307.
- Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). christophm.github.io/interpretable-ml-book
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

## Catatan
- MLP diimplementasikan dengan `MLPRegressor` (scikit-learn) yang dilatih per epoch (`partial_fit`), sehingga kurva loss dan early stopping bisa dikendalikan tanpa dependensi TensorFlow.
- Sebagian kode dan narasi disusun dengan bantuan AI generatif (Claude). Cantumkan hal ini di laporan sesuai ketentuan RPS.

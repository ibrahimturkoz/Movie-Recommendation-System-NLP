# 🎬 Movie-Recommendation-System-NLP


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-En%20Güncel-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bu proje, **Doğal Dil İşleme (NLP)** tekniklerini kullanarak film özetleri üzerinden birbirine en benzer filmleri bulan bir **İçerik Tabanlı Öneri Sistemi**dir. TMDB 5000 veri seti kullanılarak geliştirilen bu sistem, filmlerin hikaye benzerliklerini matematiksel vektörlere dönüştürerek analiz eder.

---

## 🚀 Proje Özeti ve Akışı

Sistem temel olarak şu adımları izler:
1.  **TF-IDF Vektörleştirme:** Film özetlerindeki kelimelerin önem sırasını belirler ve metinleri sayısal verilere dönüştürür.
2.  **Kosinüs Benzerliği:** Sayısallaşan film verileri arasındaki "açıyı" hesaplayarak içeriklerin birbirine ne kadar yakın olduğunu ölçer.
3.  **Sıralama:** Belirlenen filme en yakın puanı alan 5 farklı film kullanıcıya sunulur.

---

## 🛠️ Kullanılan Teknolojiler
* **Python 3.x**
* **Pandas:** Veri manipülasyonu ve CSV yönetimi.
* **Scikit-Learn:** Makine öğrenmesi algoritmaları (TF-IDF ve Cosine Similarity).

---

## 📊 Veri Seti Bilgisi
Projelerimizde Kaggle'ın popüler [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) veri seti kullanılmıştır. 
* **Toplam Veri:** 5,000 Film
* **Kritik Sütunlar:** `title` (Film Adı), `overview` (Film Özeti).

---

## 📉 Model Performans Tablosu

| Bileşen | Yöntem | Amaç |
| :--- | :--- | :--- |
| **Özellik Çıkarımı** | `TF-IDF Vectorizer` | Özet metinlerini sayısal ağırlıklara çevirme. |
| **Benzerlik Ölçümü** | `Cosine Similarity` | Vektörler arasındaki benzerlik skorunu hesaplama. |
| **Filtreleme Türü** | `Content-Based` | Filmin konusuna göre öneri yapma. |

---

## ✅ Özellikler

1-Kelime anlamlarına dayalı isabetli film önerileri.

2-Optimize edilmiş hızlı benzerlik matrisi hesaplaması.

3-Boş veri (NaN) kontrolü ile hatasız çalışma.

---

## 💻 Örnek Kullanım (Kod Bloğu)

Eğitilen modeli kullanarak herhangi bir film için benzer önerileri şu şekilde alabilirsiniz:

```python
# Fonksiyonu kullanarak önerileri alalım
oneriler = film_oner('The Dark Knight Rises')

print("Önerilen Filmler:")
print(oneriler)

"""
Beklenen Çıktı:
--- 'The Dark Knight Rises' İçin Öneriler ---
65      Batman Begins
299     Batman Forever
428     Batman Returns
1359    Batman
3854    Batman: The Dark Knight Returns, Part 2
Name: title, dtype: object
"""
```
---

## Geliştirici: İbrahim Türköz

import pandas as pd # Veri setini yüklemek ve tablolarla çalışmak için pandas kütüphanesini içe aktarır.
from sklearn.feature_extraction.text import TfidfVectorizer # Metinlerdeki kelimelerin önemini hesaplayan TF-IDF aracını yükler.
from sklearn.metrics.pairwise import cosine_similarity # İki film arasındaki benzerlik puanını (0-1 arası) hesaplamak için kullanılır.

# 1. VERİYİ YÜKLEME
df = pd.read_csv('tmdb_5000_movies.csv') # CSV dosyasını okuyarak bir DataFrame yapısına aktarır.

# 2. VERİ ÖN İŞLEME
df['overview'] = df['overview'].fillna('') # Film açıklaması boş olan satırları, hata almamak için boş bir metinle doldurur.

# 3. METİN ANALİZİ (TF-IDF VEKTÖRLEŞTİRME)
# stop_words='english' ile 'the', 'is', 'an' gibi etkisiz kelimeleri analizden çıkartırız.
tfidf = TfidfVectorizer(stop_words='english') # TF-IDF modelini oluşturur.
tfidf_matrix = tfidf.fit_transform(df['overview']) # Tüm film özetlerini sayısal bir kelime-ağırlık matrisine dönüştürür.

# 4. BENZERLİK MATRİSİ HESAPLAMA
# Her filmin diğer tüm filmlerle olan benzerlik skorunu hesaplayan dev bir tablo oluşturur.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) # Kosinüs benzerliği yöntemini kullanarak matrisi hesaplar.

# 5. İNDEKS VE FİLM ADI EŞLEŞTİRMESİ
# Film adına göre hızlıca satır numarası (index) bulmak için bir yardımcı liste oluşturur.
indices = pd.Series(df.index, index=df['title']).drop_duplicates() # Film başlıklarını anahtar, indeksleri değer olarak tutar.

# 6. ÖNERİ FONKSİYONU
def film_oner(baslik, cosine_sim=cosine_sim): # Öneri yapılacak film adını alan bir fonksiyon tanımlar.
    # Girilen film isminin veri setindeki konumunu (indeksini) bulur.
    idx = indices[baslik] 
    
    # Seçilen filmin diğer tüm filmlerle olan benzerlik puanlarını bir liste haline getirir.
    benzerlik_puanlari = list(enumerate(cosine_sim[idx])) 
    
    # Puanları en yüksekten (en benzerden) en düşüğe doğru sıralar.
    benzerlik_puanlari = sorted(benzerlik_puanlari, key=lambda x: x[1], reverse=True) 
    
    # Listenin ilk sırasındaki film, filmin kendisidir; bu yüzden 1'den 6'ya kadar olanları (toplam 5 adet) alır.
    benzerlik_puanlari = benzerlik_puanlari[1:6] 
    
    # En benzer 5 filmin satır numaralarını (indekslerini) bir listede toplar.
    film_indeksleri = [i[0] for i in benzerlik_puanlari] 
    
    # Satır numaralarını kullanarak film isimlerini ana veri setinden çeker ve döndürür.
    return df['title'].iloc[film_indeksleri] 

# 7. TEST ETME
print("--- 'The Dark Knight Rises' İçin Öneriler ---") # Başlık yazdırır.
print(film_oner('The Dark Knight Rises')) # Kara Şövalye Yükseliyor filmine benzer 5 filmi ekrana basar.

print("\n--- 'Interstellar' İçin Öneriler ---") # Başlık yazdırır.
print(film_oner('Interstellar')) # Yıldızlararası filmine benzer 5 filmi ekrana basar.
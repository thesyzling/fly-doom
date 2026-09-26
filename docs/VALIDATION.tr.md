# İlk altyapı doğrulaması — 26 Eylül 2026

- Ortam: Windows, Python 3.13.11, ViZDoom 1.3.1; tam paket sürümleri `requirements.lock.txt` içinde.
- `python -m pytest -q`: 9 test geçti. Bağlantı yönü, bölge satırlarının toplanması, izole nöronlar, öz bağlantılar, büyük tam sayı kimlikleri, geçersiz girdiler ve bozuk checksum ele alındı.
- `python -m flydoom.doom_smoke --episodes 3`: üç rastgele politika bölümü tamamlandı; 240 × 320 RGB görüntüler alındı. JSON raporu `runs/doom-smoke.json` içinde.
- Mutlak yoldaki Türkçe karakterler ViZDoom yerel katmanında hata oluşturdu. Proje içindeki sanal ortam dosyalarına göreli yollar kullanılması bu makinede sorunu çözdü.

İlk altyapı kontrolü sırasında gerçek veri henüz indirilmemişti. Aşağıdaki takip çalışması tam veri içe aktarmayı ve açıklama eşlemesini tamamladı.

## Gerçek veri doğrulaması — 26 Eylül 2026

`download`, `prepare` ve `annotate` komutları başarıyla tamamlandı.

| Ölçüm | Sonuç |
|---|---:|
| Benzersiz nöron | 139.255 |
| Kaynak bağlantı tablosu satırı | 16.847.997 |
| Bölgeler birleştirildikten sonra yönlü nöron çifti | 15.091.983 |
| Toplam sinaps | 54.492.922 |
| Açıklamalarla eşleşen nöron | 139.255 |
| Graf dışında kalan açıklama satırı | 0 |
| Seyrek matrisin veri/indeks dizileri, RAM | 173,25 MiB |
| Dolu matris hücreleri | %0,077826 |
| Kaynakta öz bağlantısı bulunan nöron | 0 |

İki Zenodo dosyası yayınlanan MD5 değerleriyle doğrulandı. Kaynak ve çıktı SHA-256 değerleri `data/processed/fafb783/manifest.json` içinde. Hazırlanan dosya diskten yeniden yüklendi; matris boyutu, pozitif sinaps sayıları ve ham dosyayla toplam sinaps eşitliği kontrol edildi.

Gerçek veride kök kimlik listesi `uint64`, bağlantı dosyasındaki kimlikler `int64` çıktı. NumPy `searchsorted` karışık türlerle kullanıldığında büyük kimliklerde hassasiyet kaybı oluşturdu. Kimlikleri karşılaştırmadan önce kayıpsız `uint64` dönüşümü uygulandı. Bitişik büyük kimliklerle regresyon testi eklendi.

Açıklama kaynağı: [flywire_annotations v2.1.0](https://github.com/flyconnectome/flywire_annotations/tree/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0), commit `ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0`. İndirilen kaynağın SHA-256 değeri `30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6`. Açıklamalar kimlik üzerinden eşlendi ve çıktı satır sırasının graf kimlikleriyle birebir aynı olduğu diskten tekrar okunarak doğrulandı.

Tam eşleşme, bütün alanların bilindiği anlamına gelmez: 28.165 nöronun `cell_type`, 601 nöronun `top_nt` alanı boş. Kaynak sütunları korunur; eksik değerler tahminle doldurulmadı. `top_nt` etiketleri tahmindir. Bu aşamada bağlantılara uyarıcı/baskılayıcı işaret atanmadı.

`python -m pytest -q`: **13 test geçti.** Kimlik türü, negatif kimlik ve açıklama eşlemesi kontrolleri eklendi. Bu sonuçlar veri hazırlama doğruluğuna ilişkindir; nöron dinamiği, görsel sinyal aktarımı, eğitim ve biyolojik geçerlilik henüz doğrulanmadı.

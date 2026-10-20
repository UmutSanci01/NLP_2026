# 🚀 Bölüm 4: Modellerin ve Tokenizer'ların Paylaşılması — Uygulama Laboratuvarı

**Takım:** NLP04 (2026 Güz)  
**Ders:** Doğal Dil İşleme (NLP)  
**Resmi Kaynak:** [Hugging Face LLM Course — Chapter 4](https://huggingface.co/learn/llm-course/chapter4/1)  
**Konu:** Hugging Face Hub, Model Paylaşımı, `push_to_hub` API, `huggingface_hub` ve Model Kartı  

---

## 📌 Not Defterleri (Notebooks)

Klasör içerisinde her iki derin öğrenme çatısı için resmi kurs kodlarının **birebir 1:1 sırasıyla ve eksiksiz** yer aldığı iki ana not defteri bulunmaktadır:

| Dosya / Dizin | Çatı / Tür | Açıklama |
|---|---|---|
| 🔵 **[`main_pytorch.ipynb`](./main_pytorch.ipynb)** | PyTorch | Hugging Face Bölüm 4'ün tüm PyTorch kodlarını (22 kod hücresi) resmi sırasıyla içerir. |
| 🟠 **[`main_tensorflow.ipynb`](./main_tensorflow.ipynb)** | TensorFlow | Hugging Face Bölüm 4'ün tüm TensorFlow kodlarını (`TFCamembert*`, `PushToHubCallback`) resmi sırasıyla içerir. |
| 📦 **[`requirements.txt`](./requirements.txt)** | Bağımlılıklar | Gerekli Python kütüphaneleri (`torch`, `transformers`, `sentencepiece`, `huggingface-hub`, vb.). |
| 📖 **[`README.md`](./README.md)** | Dokümantasyon | Uygulama laboratuvarı rehberi ve Colab bağlantıları. |

> 💡 **Önemli Not:** Not defterleri çalıştırıldığında yerel olarak üretilen `dummy_*` klasörleri (model ağırlıkları ~442 MB) GitHub'ın 100 MB dosya limitini aşmamak ve depoyu temiz tutmak amacıyla repoya dahil edilmez. Kodlar çalıştırıldığında bu klasörler çalışma anında otomatik oluşturulur.

---

## 🧪 Not Defterlerinin Birebir Ders Akışı

Her iki not defteri de resmi Hugging Face müfredatını birebir takip eder:

1. **Ortam Kurulumu:** Gerekli kütüphanelerin (`transformers`, `huggingface_hub`, `datasets`, `evaluate`) kurulumu.
2. **Önceden Eğitilmiş Modellerin Kullanımı (Section 2):**
   - **Kod 1:** `pipeline("fill-mask", model="camembert-base")` ile maske doldurma.
   - **Kod 2:** Modele özel mimari sınıfları (`CamembertTokenizer`, `CamembertForMaskedLM` / `TFCamembertForMaskedLM`).
   - **Kod 3:** Mimariden bağımsız dinamik sınıflar (`AutoTokenizer`, `AutoModelForMaskedLM` / `TFAutoModelForMaskedLM`).
3. **Önceden Eğitilmiş Modelleri Paylaşma (Section 3):**
   - **Kod 4:** Git-LFS ve Git kimlik yapılandırması (`git config --global`).
   - **Kod 5:** Hugging Face Hub kimlik doğrulama (`notebook_login()`).
   - **Kod 6:** Eğitim sırasında otomatik yükleme (`TrainingArguments(push_to_hub=True)` veya `PushToHubCallback`).
   - **Kod 7 - 11:** Doğrudan nesne üzerinden `model.push_to_hub()` ve `tokenizer.push_to_hub()` (varsayılan, `organization="huggingface"` ve `use_auth_token` parametreleriyle).
   - **Kod 12 - 15:** `huggingface_hub` Python kütüphanesi fonksiyonları (`create_repo`, `upload_file`).
   - **Kod 16 - 20:** `Repository` sınıfı ile Git benzeri iş akışı (`git_pull`, `model.save_pretrained`, `git_add`, `git_commit`, `git_push`).
   - **Kod 21:** Bölüm 3 kapanışı — model ve tokenizer'ın diske kaydedilmesi.
4. **Model Kartı Oluşturma (Section 4):**
   - **Kod 22:** Resmi kurs YAML Frontmatter başlığı (`language: fr`, `license: mit`, `datasets: - oscar`) içeren Model Kartı (`README.md`).

---

## 💻 Nasıl Çalıştırılır?

### Yöntem 1: Google Colab (Doğrudan Tek Tıkla)
Notebook'ları Google Colab'a yükleyip ilk hücredeki `!pip install` komutunu çalıştırarak sıfır kurulumla hemen çalıştırabilirsiniz:
- [Google Colab (PyTorch)](https://colab.research.google.com/github/emirhanalemdaroglu/NLP_2026/blob/main/bolum-04-modellerin-ve-tokenizerlarin-paylasilmasi/04-modellerin-ve-tokenizerlarin-paylasilmasi/uygulama/v1-2026-nlp04/main_pytorch.ipynb)
- [Google Colab (TensorFlow)](https://colab.research.google.com/github/emirhanalemdaroglu/NLP_2026/blob/main/bolum-04-modellerin-ve-tokenizerlarin-paylasilmasi/04-modellerin-ve-tokenizerlarin-paylasilmasi/uygulama/v1-2026-nlp04/main_tensorflow.ipynb)

### Yöntem 2: Yerel Bilgisayarda (Local)
```bash
# 1. Sanal ortamı aktif edin ve bağımlılıkları kurun:
pip install -r requirements.txt

# 2. Not defterini başlatın:
jupyter notebook main_pytorch.ipynb
# veya
jupyter notebook main_tensorflow.ipynb
```

> **🔑 Not:** Model yükleme (`push_to_hub`) dışındaki tüm hücreler (model indirme, pipeline, tokenizer, mimari sınıfları, yerel kaydetme ve model kartı oluşturma) Hugging Face hesabı olmadan da doğrudan ve eksiksiz çalışır.

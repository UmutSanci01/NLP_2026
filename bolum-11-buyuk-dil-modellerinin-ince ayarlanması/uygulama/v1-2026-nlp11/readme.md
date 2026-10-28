# Supervised Fine-Tuning (SFT) ve LoRA — Hugging Face Transformers

Bu depo, Hugging Face `transformers`, `peft`, `trl` ve `datasets` kütüphaneleriyle **Supervised Fine-Tuning (SFT)** sürecini uygulamalı olarak gösteren tek hücreli bir Jupyter Notebook (`sft_lora_pipeline.ipynb`) içerir. Notebook dört ana konuyu ele alır: **chat templates**, **SFT eğitimi**, **LoRA adaptörleri** ve **değerlendirme**.

## İçerik

Notebook aşağıdaki bölümlerden oluşur:

| Bölüm | Başlık | Açıklama |
|-------|--------|----------|
| 1 | Ortam ve Kurulum | Gerekli paketlerin kurulumu, cihaz (CPU/GPU) tespiti ve model/tokenizer yükleme |
| 2 | Chat Templates | `apply_chat_template` ile sistem promptu ve rol tabanlı mesaj yapılandırması |
| 3 | Supervised Fine-Tuning (SFT) | `SFTTrainer` ile etiketli veri üzerinde denetimli ince ayar |
| 4 | Low Rank Adaptation (LoRA) | `LoraConfig` ve PEFT ile bellek-verimli ince ayar |
| 5 | Değerlendirme | `evaluate` kütüphanesi ile metrik hesaplama ve model performans ölçümü |

## Kullanılan Teknolojiler

- **Python**
- **PyTorch** — derin öğrenme altyapısı
- **Hugging Face Transformers** — önceden eğitilmiş modeller ve tokenizer'lar
- **TRL (Transformer Reinforcement Learning)** — `SFTTrainer` ve `SFTConfig`
- **PEFT (Parameter-Efficient Fine-Tuning)** — LoRA adaptörleri
- **Datasets** — veri seti yükleme ve işleme
- **Evaluate** — değerlendirme metrikleri

## Kullanılan Modeller

- `HuggingFaceTB/SmolLM2-135M` — küçük, CPU'da çalışabilen test modeli (SFT ve LoRA için)
- `meta-llama/Llama-3.2-1B` (isteğe bağlı) — daha büyük ölçekli SFT denemeleri için

## Kurulum

Gerekli bağımlılıkları yükleyin:

```bash
pip install transformers torch accelerate peft trl datasets evaluate
```

## Kullanım

Notebook'u Jupyter, Google Colab veya VS Code üzerinden açıp tüm hücreleri sırayla çalıştırmanız yeterlidir:

```bash
jupyter notebook sft_lora_pipeline.ipynb
```

Not: Modeller ilk çalıştırmada Hugging Face Hub'dan indirilir, bu nedenle internet bağlantısı gerekir. Daha hızlı indirme ve yüksek hız limitleri için `HF_TOKEN` ortam değişkeni tanımlanabilir.

## Bölüm Detayları

### 1. Chat Templates

Hugging Face'in `apply_chat_template` API'si ile sistem promptları ve rol tabanlı mesajlar yapılandırılır. Bu bölümde şunlar gösterilir:

- **Sistem promptu**: Modelin genel davranışını belirleyen talimat.
- **Kullanıcı ve asistan rolleri**: Mesajların `role` ve `content` alanlarıyla yapılandırılması.
- **`add_generation_prompt=True`**: Modelin yanıt üretmeye başlaması için uygun özel token'ların eklenmesi.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

sohbet = [
    {"role": "system", "content": "Sen yardımcı bir asistansın."},
    {"role": "user", "content": "Merhaba, nasılsın?"}
]

sablonlu_metin = tokenizer.apply_chat_template(
    sohbet,
    tokenize=False,
    add_generation_prompt=True
)
print(sablonlu_metin)
```

### 2. Supervised Fine-Tuning (SFT)

`SFTTrainer` ve `SFTConfig` kullanılarak etiketli bir veri seti üzerinde denetimli ince ayar yapılır. Bu bölümde şunlar gösterilir:

- **`Dataset` nesnesi oluşturma**: `datasets` kütüphanesi ile metin verisinin hazırlanması.
- **`SFTConfig` parametreleri**: `num_train_epochs`, `per_device_train_batch_size`, `learning_rate`, `max_length` gibi temel eğitim ayarları.
- **`SFTTrainer` ile eğitim**: Modelin belirtilen veri seti üzerinde ince ayarı.

```python
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM

veri = [
    {"text": "Soru: Türkiye'nin başkenti neresidir?\nCevap: Ankara"},
    {"text": "Soru: 2 + 2 kaç eder?\nCevap: 4"},
]
dataset = Dataset.from_list(veri)

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    max_length=64,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### 3. Low Rank Adaptation (LoRA)

LoRA, model ağırlıklarını dondurup yalnızca düşük ranklı matrisleri eğiterek bellekten tasarruf sağlar. Bu bölümde şunlar gösterilir:

- **`LoraConfig` yapılandırması**: `r` (rank), `lora_alpha`, `lora_dropout`, `target_modules` parametreleri.
- **PEFT ile adaptör uygulama**: `get_peft_model` veya `SFTTrainer` içinde `peft_config` kullanımı.
- **Bellek avantajı**: Tüm model yerine yalnızca adaptör ağırlıklarının güncellenmesi.

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)
trainer.train()
```

### 4. Değerlendirme

`evaluate` kütüphanesi ile modelin görev performansı ölçülür. Bu bölümde şunlar gösterilir:

- **`evaluate.load("accuracy")`**: Doğruluk metriğinin yüklenmesi.
- **Tahmin ve referans karşılaştırması**: Model çıktılarının etiketlerle karşılaştırılması.
- **Çok boyutlu değerlendirme**: Yalnızca `accuracy` değil; güvenlik ve robustluk gibi ek metriklerin önemine dair notlar.

```python
import evaluate

accuracy = evaluate.load("accuracy")
tahminler = [1, 0, 1, 1, 0, 1]
referanslar = [1, 0, 0, 1, 0, 1]

sonuc = accuracy.compute(predictions=tahminler, references=referanslar)
print(sonuc)
```

## Çıktı Örneği

SFT eğitimi sonrası model yanıtı:

```
Test Sorusu: Soru: Türkiye'nin başkenti neresidir?
Cevap:
Modelin Yanıtı: Soru: Türkiye'nin başkenti neresidir?
Cevap: Ankara
```

Değerlendirme örneği:

```
Accuracy: 0.8333
```

## Lisans

Bu proje eğitim amaçlıdır. Kullanılan önceden eğitilmiş modellerin lisans koşulları Hugging Face Hub'daki ilgili model sayfalarında belirtilmiştir.

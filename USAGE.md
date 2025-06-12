# Clickbait Spoiling Implementation

## 📋 Tóm tắt

Đây là implementation của bài báo clickbait spoiling sử dụng GPT-2 medium với các hyperparameters được tối ưu hóa dựa trên phân tích nghiên cứu.

### 🎯 Hyperparameters chính (theo paper):
- **Model**: GPT-2 medium (~355M parameters)
- **Input**: postText + targetParagraph (max 512 tokens)
- **Loss**: Sparse categorical cross-entropy (causal LM)
- **Epochs**: 10
- **Decoding**: Conservative (greedy/small beam) để đạt BLEU cao

## 🚀 Cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Download NLTK data (nếu cần)
python -c "import nltk; nltk.download('punkt')"
```

## 📊 Cấu trúc dữ liệu

Dataset SemEval-2023 với format:
- **3,200 mẫu training** (train.jsonl)
- **800 mẫu validation** (validation.jsonl)
- Mỗi mẫu chứa: postText, targetParagraphs, spoiler, tags

## 🔧 Sử dụng

### 1. Training Model (Fine-tuning GPT-2)

```bash
# Training đầy đủ với hyperparameters tối ưu
cd clickbait-spoiling
python clickbait_spoiling.py
```

**Hyperparameters được sử dụng:**
- Learning rate: 5e-5 (conservative)
- Batch size: 4 (với gradient accumulation = 2)
- Scheduler: Cosine decay
- Epochs: 10 (như paper)
- Early stopping trên validation BLEU

### 2. Quick Testing (Không cần training)

```bash
# Test với pre-trained GPT-2 medium
python quick_spoiler.py --test

# Interactive mode
python quick_spoiler.py

# Single example
python quick_spoiler.py \
    --post "You Won't Believe What This Celebrity Did Next" \
    --context "The celebrity announced retirement to pursue marine biology" \
    --strategy conservative
```

### 3. Batch Generation

```bash
# Generate spoilers cho toàn bộ validation set
python quick_spoiler.py \
    --batch data/validation.jsonl \
    --output validation_predictions.json \
    --strategy conservative
```

### 4. Evaluation (So sánh với paper)

```bash
# Đánh giá kết quả với metrics chuẩn
python evaluate_spoilers.py \
    --predictions validation_predictions.json \
    --compare \
    --output detailed_evaluation.json
```

## 📈 Kết quả mong đợi

### Paper Results:
- **BLEU-4**: 0.58
- **ROUGE-L**: 0.68
- **BERTScore**: 0.46
- **METEOR**: 0.47

### Strategies để cải thiện:

#### 🎯 Conservative Strategy (Gần paper nhất)
```python
# Temperature = 0.0 (greedy)
# Beam size = 1 hoặc 4
# Max tokens = 30
```

#### 🔄 Beam Search Strategy
```python
# Temperature = 0.0
# Beam size = 4
# No repeat ngram = 3
```

#### 🎨 Creative Strategy (Đa dạng hơn)
```python
# Temperature = 0.7
# Top-p = 0.8
# Max tokens = 50
```

## 🔍 Phân tích kết quả

### Nếu BLEU/ROUGE thấp nhưng BERTScore cao:
➡️ Model hiểu đúng ý nhưng diễn đạt khác → Cần generation conservative hơn

### Nếu kết quả quá dài:
➡️ Giảm `max_new_tokens`, tăng `no_repeat_ngram_size`

### Nếu BLEU < 0.3:
➡️ Cần training lâu hơn hoặc điều chỉnh preprocessing

## 🛠️ Debugging & Optimization

### 1. Kiểm tra data preprocessing
```bash
# Xem sample data
head -1 data/train.jsonl | python -m json.tool
```

### 2. Monitor training
```bash
# Tensorboard logs sẽ được lưu trong ./results/
tensorboard --logdir ./results
```

### 3. Test generation strategies
```bash
# So sánh các strategies
python quick_spoiler.py --test
```

## 📝 Examples

### Input:
```
Post: "Scientists Hate Him for This One Weird Discovery"
Context: "A researcher found that turmeric can reduce inflammation significantly."
```

### Expected Output:
```
Conservative: "turmeric reduces inflammation"
Beam: "turmeric can reduce inflammation"
Creative: "a common spice helps fight inflammation"
```

## ⚡ Quick Start

```bash
# 1. Clone và setup
cd clickbait-spoiling
pip install -r requirements.txt

# 2. Test ngay với pre-trained model
python quick_spoiler.py --test

# 3. Generate cho validation set
python quick_spoiler.py --batch data/validation.jsonl --output results.json

# 4. Evaluate
python evaluate_spoilers.py --predictions results.json --compare
```

## 🎯 Tips để đạt kết quả gần paper

1. **Preprocessing**: Chỉ lấy paragraph ngắn nhất chứa answer
2. **Generation**: Dùng greedy hoặc beam nhỏ (≤4)
3. **Training**: 10 epochs với early stopping
4. **Evaluation**: Đảm bảo lowercase + tokenization đúng

## 📞 Troubleshooting

### GPU Memory Issues:
```bash
# Giảm batch size
export CUDA_VISIBLE_DEVICES=0
# Hoặc sử dụng gradient checkpointing
```

### Dependencies Issues:
```bash
# Reinstall transformers
pip install --upgrade transformers torch
```

### Low BLEU scores:
- Kiểm tra generation temperature (phải = 0.0)
- Kiểm tra beam size (≤4)
- Kiểm tra preprocessing input 
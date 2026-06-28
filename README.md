# Fake News Detection

Ứng dụng web phát hiện tin tức giả mạo (**Fake News Detection**) sử dụng hai mô hình AI song song: **BERT** (deep learning) và **Naive Bayes** (machine learning truyền thống). Người dùng nhập đoạn văn bản tin tức, hệ thống trả về kết quả dự đoán từ cả hai mô hình để so sánh.

---

## Tính năng

- **Giao diện web** đơn giản, nhập văn bản và nhận kết quả ngay lập tức
- **BERT** (`bert-base-uncased`) — mô hình transformer độ chính xác cao
- **Naive Bayes + TF-IDF** — mô hình nhẹ, dự đoán nhanh
- **So sánh kết quả** từ hai mô hình cùng một lúc
- **Pipeline huấn luyện** hoàn chỉnh cho cả hai mô hình

---

## Cấu trúc dự án

```
fake-news-detection/
├── app.py                  # Flask web server
├── predict.py              # Load model và dự đoán
├── train_BERT.py           # Huấn luyện mô hình BERT (chạy trên Kaggle/GPU)
├── train_naive_bayes.py    # Huấn luyện mô hình Naive Bayes
├── utils.py                # Tiền xử lý văn bản, load dataset
├── data/
│   ├── True.csv            # Dataset tin thật
│   └── Fake.csv            # Dataset tin giả
├── BERT_model/             # Model BERT đã huấn luyện 
├── naive_bayes_model.pkl   # Model Naive Bayes đã huấn luyện
└── templates/
    └── index.html          # Giao diện web
```

---

## Yêu cầu hệ thống

- Python 3.8+
- GPU (khuyến nghị cho BERT — có thể dùng Kaggle/Google Colab miễn phí)
- RAM ≥ 8GB

---

## Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/duytanst2004/fake-news-detection.git
cd fake-news-detection

# 2. Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Cài dependencies
pip install flask torch transformers scikit-learn pandas nltk tqdm
```

---

## Chuẩn bị dữ liệu

Tải dataset từ Kaggle: [Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Đặt file vào thư mục `data/`:

```
data/
├── True.csv
└── Fake.csv
```

Cấu trúc CSV yêu cầu có các cột: `title`, `text`.

---

## Huấn luyện mô hình

### Naive Bayes (chạy local)

```bash
python train_naive_bayes.py
```

Kết quả: file `naive_bayes_model.pkl` được lưu vào thư mục gốc.

### BERT (khuyến nghị chạy trên Kaggle/Colab có GPU)

```bash
python train_BERT.py
```

> **Lưu ý:** Cập nhật đường dẫn dataset trong `train_BERT.py` nếu không dùng Kaggle:
> ```python
> fake_path = 'data/Fake.csv'
> true_path = 'data/True.csv'
> ```

Kết quả: thư mục `BERT_model/` chứa model và tokenizer đã lưu.

---

## Chạy ứng dụng web

Đảm bảo đã có `BERT_model/` và `naive_bayes_model.pkl` trước khi chạy.

```bash
python app.py
```

Mở trình duyệt tại: [http://localhost:5000](http://localhost:5000)

---

## Hướng dẫn sử dụng

1. Truy cập `http://localhost:5000`
2. Dán hoặc nhập đoạn văn bản tin tức vào ô nhập liệu
3. Nhấn **Predict**
4. Kết quả hiển thị từ cả hai mô hình: `Real` hoặc `Fake`

---

## Mô tả kỹ thuật

### Pipeline xử lý văn bản (`utils.py`)

```
Văn bản thô
    │
    ▼
Loại bỏ ký tự đặc biệt, số
    │
    ▼
Chuyển về chữ thường
    │
    ▼
Loại bỏ stopwords (NLTK)
    │
    ▼
Văn bản sạch
```

### Mô hình BERT

| Thông số | Giá trị |
|---|---|
| Base model | `bert-base-uncased` |
| Số nhãn | 2 (Fake / Real) |
| Max token length | 100 |
| Batch size | 32 |
| Epochs | 3 |
| Optimizer | AdamW, lr = 2e-5 |

### Mô hình Naive Bayes

| Thông số | Giá trị |
|---|---|
| Vectorizer | TF-IDF |
| Max features | 5.000 |
| Classifier | MultinomialNB |
| Test split | 20% |

### Quy trình dự đoán (`predict.py`)

```
Văn bản đầu vào
    │
    ├──► BERT Tokenizer → BERT Model → Argmax → Real / Fake
    │
    └──► TF-IDF Transform → Naive Bayes → Real / Fake
```

---

## Mô tả các file

| File | Chức năng |
|---|---|
| `app.py` | Flask app, nhận POST request, gọi `predict()`, render kết quả |
| `predict.py` | Load cả hai model, cung cấp hàm `predict(text)` trả về dict kết quả |
| `train_BERT.py` | Load data → tokenize → train BERT → evaluate → lưu model |
| `train_naive_bayes.py` | Load data → TF-IDF → train Naive Bayes → evaluate → lưu pkl |
| `utils.py` | `clean_text()` làm sạch văn bản, `load_dataset()` đọc và ghép CSV |

---

## Độ chính xác

| Mô hình | Accuracy |
|---|---|
| BERT | ~98–99% |
| Naive Bayes | ~93–95% |

> Kết quả thực tế phụ thuộc vào dataset và số epoch huấn luyện.

---

## Demo

### Home
![home.png](image/home.png)

### Kết quả
![result.png](image/result.png)
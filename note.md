# Fake News Detection Application

## File train_BERT.py

### import library

#### `import pandas as pd`

#### `import torch`
- "PyTorch" là framework chính để xây dựng và huấn luyện mô hình deep learning.
- `torch`
  + Làm việc với tensor (`torch.tensor(...)`)
  + Kiểm tra thiết bị (CPU/GPU): `torch.device(...)`

#### Function `load_data`

### Function: `prepare_data`

#### Mục đích
- Chuyển danh sách văn bản (texts) và nhãn (labels) thành định dạng tensor mà BERT và PyTorch có thể sử dụng để huấn luyện hoặc đánh giá.

#### `def prepare_data(texts, labels, tokenizer, max_length=100, batch_size=32):`
- texts: danh sách các đoạn văn bản (chuỗi) - dữ liệu đầu vào.
- lables: Series của pandas chứa các nhãn tương ứng (0 hoặc 1).
- tokenizer: bộ totkenizer của BERT dùng để biến văn bản thành token ID.
- max_length: độ dài tối đa cho mỗi chuỗi văn bản. Nếu quá dài thì cắt bớt.
- batch_size: số lượng mẫu dữ liệu trong mỗi batch dùng để huấn luyện.

#### `encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors='pt')`
- `tokenizer(...)`: gọi bộ tokenizer của BERT để mã hóa văn bản thành
  . input_ids: ID của các token.
  . attention_mask: mặt nạ để mô hình biết token nào là thực, token nào là padding.
- `truncation=True`: cắt chuỗi nếu vượt quá max_length
- `padding=True`: thêm token "PAD" để chuỗi ngắn được "đệm" cho đủ độ dài.
- `return_tensors='pt'`: trả về kết quả dưới dạng Tensor PyTorch (thay vì List)
- Kết quả `encodings` là một dict chứa:
`{
  'input_ids': Tensor(shape: [số mẫu, max_length]),
  'attention_mask': Tensor(shape: [số mẫu, max_length])
}`

#### `dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels.values))`
- `TensorDataset`: gộp nhiều tensor lại với nhau thành một dataset PyTorch.
- Mỗi phần tử của dataset sẽ có dạng tuple: `(input_ids, attention_mask, lable)`
- `lables.values`: chuyển nhã từ pandas Series thành NumPy array, rồi thành tensor.

#### `loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)`
- `DataLoader`: tạo một bộ sinh dữ liệu (dataloader) để dễ dàng duyệt qua dữ liệu theo từng batch.
- `batch_size`: số mẫu mỗi batch.
- `shuffle=True`: xáo trộn dữ liệu trước mỗi epoch để huấn luyện hiệu quả hơn.

### Function: `train_model`

#### Mục tiêu
- Huấn luyện mô hình BERT với dữ liệu đã được chuẩn bị sẵn (qua `DataLoader`) trong nhiều epoch, dùng optimizer để cập nhật trọng số dựa trên hàm mất mát.

#### `def train_model(model, train_loader, optimizer, device, epochs=3)`
- `model`: mô hình BERT đã được khởi tạo (với num_labels=2)
- `train_loader`: đối tượng `DataLoader` chứa dữ liệu huấn luyện (input_ids, attention_mask, labels)
- `optimizer`: bộ tối ưu hóa (AdamW thường dùng với BERT)
- `device`: thiết bị huấn luyện (cpu hoặc cuda)
- `epochs`: số lần lặp toàn bộ dữ liệu huấn luyện

#### `model.train()`
- Đặt mô hình ở chế độ "train mode", giúp:
  + Bật dropout, batchnorm (nếu có).
  + Cho mô hình cập nhật trọng số

#### Vòng lặp qua các epoch
``` python
for epoch in range(epochs):
  print(f'Epoch: {epoch + 1}/{epochs})')
```
- Mỗi vòng lặp epoch đi qua toàn bộ dataset một lần.
- In ra tiến trình để dễ theo dõi.


#### `for batch in tqdm(train_loader):`
- Duyệt từng `batch` trong `train_loader` (dùng `tqdm` để có thanh tiến trình đẹp)
- Mỗi batch chứa:
  + `input_ids`: token ID (mã hóa câu thành số nguyên)
  + `attention_mask`: mặt nạ xác định token nào là thực (1), đâu là padding(0).
  + `labels`: nhãn (0 hoặc 1)

#### `input_ids, attention_mask, lables = [b.to(device) for b in batch]`
- Duyệt các phần tử của `batch` (3 tensor), đưa chúng lên `device` (cuda) nếu có.
- Mỗi batch giờ đây đã sẵn sàng cho huấn luyện.

#### `outputs = model(input_ids, attention_mask=attention_mask, labels=labels)`
- Gọi mô hình với input (`input_ids`, `attention_mask`, `labels`).
- Vì có `labels`, mô hình sẽ
  + Tự động tính toán "loss" (dựa trên `CrossEntropyLoss`)
- `outputs` là một `SequenceClassifierOutput`, có thuộc tính
  + `loss`: giá trị mất mát.
  + `logits`: xác suất đầu ra thô.

#### Backpropagation
``` python
loss = outputs.loss
loss.backward()
```
- Lấy giá trị loss.
- Gọi `.backward()` để tính toán "gradient" ngược lại các layer trong mô hình.

#### Tối ưu và reset gradient
``` python
optimizer.step()
optimizer().zero_grad()
```
- `optimizer.step()`: cập nhật trọng số mô hình theo gradient.
- `optimizer.zero_grad()`: reset gradient sau mỗi bước cập nhật, để tránh cộng dồn sai.

#### Gradient là gì?
- Định nghĩa
  + "Gradient (đạo hàm riêng)" là một vector chỉ hướng và độ lớn mà hàm mất mát (loss function) "tăng nhanh nhất" theo từng tham số (weights) trong mạng nơ-ron.
- Nói cách đơn giản
  + Gradient cho biết: "Nếu tôi thay đổi một chút tham số này, thì loss tăng hay giảm bao nhiêu?"
  + Dựa vào đó, ta có thể điểu chỉnh các tham số sao cho loss giảm dần, tức là mô hình học tốt hơn.
- Ví dụ trực quan: Giả sử bạn đang đứng trên một quả đồi và muốn xuống chân đồi (giảm loss).
  + Gradient giống như chiếc la bàn chỉ hướng "dốc nhất để đi xuống".
  + Mỗi bước bạn đi theo hướng ngược lại của gradient -> bạn sẽ từ từ xuống đốc -> đến đáy -> mô hình tốt dần lên.

### Function: `evaluate_model`

#### Mục tiêu
- Hàm này thực hiện "dự đoán (prediction) trên tập test và trả về"
  + Danh sách các "label thật" (`true_labels`)
  + Danh sách các "label mô hình dự đoán" (`predictions`)
- Sau đó, ta có thể dùng 2 danh sách này để tính toán độ chính xác (accuracy), F1-score,... bằng `classification_report()`

#### `def evaluate_model(model, test_leader, device):`
- `model`: mô hình BERT đã được huấn luyện.
- `test_loader`: dữ liệu kiểm tra (test set), chia theo batch.
- `device`: dùng CPU hoặc GPU

#### `mode.eval()`
- Đặt mô hình vào "eval mode"
  + Tắt dropout (giúp kết quả ổn định).
  + BatchNorm (nếu có) hoạt động khác so với train mode.

#### Tạo list lưu kết quả
``` python
prediction = []
true_labels = []
```
- `predictions`: để lưu nhãn dự đoán.
- `true_labels`: để lưu thật tương ứng.

#### `with torch.no_grad()`
- Vô hiệu hóa autograd (gradient tracking) để
  + Tiết kiệm bộ nhớ
  + Tăng tốc tính toán
  + Tránh rò rỉ gradient vào quá trình huấn luyện

#### `for batch in test_loader`

#### `input_ids, attention_mask, lables = [b.to(device) for b in batch`

#### Gọi mô hình để dự đoán
``` python
outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits
```
- Không đưa lables vào đây -> mô hình "chỉ dự đoán", không tính "loss".
- `outputs.logits`: đầu ra thô (chưa qua softmax) của mô hình, dạng shape `[batch_size, num_labels]`.

#### `preds = torch.argmax(logits, dim=1)`
- Lấy chỉ số có giá trị lớn nhất trong mỗi vector `logits` -> chính là `nhãn dự đoán` (0 hoặc 1).

#### Chuyển kết quả về CPU và lưu
``` python
predictions.extend(preds.cpu().numpy())
true_labels.extend(labels.cpu().numpy())
```
- `.cpu()`: đưa kết quả từ GPU về CPU.
- `.numpy()`: chuyển từ tensor sang mảng NumPy để dễ xử lý sau.
- `.extend(...)`: nối các kết quả từ batch vào list tổng.

#### `return true_labels, predictions`

### Function: `main()`

#### Code
``` python
fake_path = '/kaggle/input/fake-news-datasets/data/Fake.csv'
true_path = '/kaggle/input/fake-news-datasets/data/True.csv'

df = load_data(fake_path, true_path)

X = df['text']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### `tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')`
- Tải tokenizer của mô hình `bert-base-uncased` từ thư viện `transformers`.
- Tokenizer sẽ biến đổi văn bản thành các token ID (đầu vào của BERT).

#### Code
``` python
train_loader = prepare_data(X_train, y_train, tokenizer)
test_loader = prepare_data(X_test, y_test, tokenizer)
```

#### Khởi tạo mô hình
``` python
model = BertForSequenClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')
model.to(device)
```

#### `optimizer = AdamW(model.parameters(), lr=2e-5)`
- Dùng `AdamW(...)` (phiên bản tối ưu hơn của Adam, hỗ trợ weight decay).
- `lr=2e-5`: learning rate thường dùng cho fine-tuning BERT.

#### Code
``` python
train_model(model, train_loader, optimizer, device)

true_labels, predictions = evaluate_model(model, test_loader, device)
print(classification_report(true_labels, predictions, target_names=['Fake', 'Real']))

model.save_pretrained('./BERT_model')
tokenizer.save_pretrained('./BERT_model)
```

## Input test app

### Fake Sentences
- Content Snippet: "A new study PROVES that 5G networks are the REAL cause of the COVID-19 pandemic. Governments are covering it up to push vaccines!"

### True Sentences
- Headline: "U.S. Federal Reserve Raises Interest Rates by a Quarter Percentage Point"

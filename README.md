# Emotion AI: Hệ thống Phân loại Cảm xúc dựa trên kiến trúc Transformer

Dự án này là đồ án cuối kỳ cho môn học **Học Thống kê** (Statistical Learning). Mục tiêu của dự án là xây dựng một ứng dụng phân loại cảm xúc người dùng từ văn bản tiếng Anh bằng cách tinh chỉnh mô hình Transformer (BERT).

##  Tính năng chính
* **Mô hình AI mạnh mẽ**: Sử dụng `BERT-base-uncased` được tinh chỉnh trên tập dữ liệu `dair-ai/emotion`.
* **Phân loại 6 nhãn cảm xúc**: Sadness, Joy, Love, Anger, Fear, Surprise.
* **Kiến trúc Server-Client**:
    * **Backend**: FastAPI xử lý logic inference (dự đoán) mô hình.
    * **Frontend**: Streamlit cung cấp giao diện chatbot tương tác thời gian thực.

##  Cấu trúc thư mục
```text
Emotions-Classification/
├── backend/                # API Server (FastAPI)
│   ├── main.py             # Script chạy server
|   ├── requirements.txt     
│   └── Dockerfile
├── frontend/               # UI (Streamlit)
│   ├── app.py              # Script chạy giao diện chatbot
│   ├── requirements.txt
|   └── Dockerfile  
├── model/                  # Nơi chứa model sau khi tải từ link.txt             
├── notebooks/              
│   └── notebook.ipynb      # File huấn luyện (Kaggle/Colab)
├── README.md               # Hướng dẫn dự án
├── docker-compose.yml
└── link.txt                # Link drive để tải trọng số model: config.json, modelsafetensors...  
```

## Yêu cầu hệ thống
- Python: 3.9 trở lên.
- RAM: Tối thiểu 4GB (Khuyên dùng 8GB để load mô hình mượt mà).
- GPU: Không bắt buộc (Inference chạy tốt trên CPU).

## Hướng dẫn cài đặt và khởi chạy
```bash
git clone https://github.com/vdthcmus123/Emotions-Classification.git
cd Emotions-Classification
```

- Bước 1: Chuẩn bị mô hình: Đảm bảo đã tải các file trọng số mô hình (model.safetensors, config.json, tokenizer.json,...) từ `link.txt` vào thư mục `model/`.

- Bước 2: Khởi chạy Backend: 
Mở terminal tại thư mục dự án: backend chạy localhost tại port 8000
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- Bước 3: Khởi chạy Frontend: 
Mở một terminal mới: frontend chạy localhost tại port 8501
```bash
cd frontend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
## Chạy đơn giản với duy nhất 1 lệnh 
- Điều hướng vào folder `Emotions-Classification`, lần đầu tiên chạy
```bash
docker-compose up --build
```
- Các lần chạy sau đó, chỉ cần chạy **docker-compose up**
- Nếu có thay đổi về thư viện trong ``requirements.txt`` thì chạy lại **docker-compose up --build**
- Nếu có thay đổi về frontend hoặc backend thì chạy lại **docker-compose restart frontend/backend**
- Ứng dụng khởi chạy tại http://localhost:8501

## Thông tin mô hình và Tập dữ liệu
- Tập dữ liệu: dair-ai/emotion (16,000 mẫu train, 2,000 mẫu val, 2,000 mẫu test).

- Kiến trúc: BERT-base-uncased.

- Kỹ thuật tối ưu: Sử dụng WeightedTrainer để xử lý mất cân bằng nhãn.

- Threshold Tuning để cải thiện F1-Score cho các lớp cảm xúc hiếm gặp.

- Hiệu năng: Đạt Accuracy khoảng 93% trên tập test.

## Nhóm thực hiện
|MSSV|Họ và Tên|
|----|---------|
|23120093|Vũ Duy Thụ|
|23120089|Đỗ Quốc Thịnh|
|23120153|Cù Văn Nhựt|

Giáo viên lý thuyết: TS. Ngô Minh Nhựt

Giáo viên thực hành: Thầy Lê Long Quốc

Đơn vị: Trường Đại học Khoa học Tự nhiên - VNUHCM (HCMUS).

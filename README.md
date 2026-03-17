# Duky AI RAG System

Hệ thống chatbot hỗ trợ thông tin dự án Duky AI sử dụng RAG (Retrieval-Augmented Generation) với Gemini API và ChromaDB.

## 🚀 Cài đặt

1. **Clone repository:**
   ```bash
   git clone https://github.com/Tiens0710/DUKY-RAG.git
   cd dukyai-rag
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình môi trường:**
   - Copy file `.env.example` thành `.env`.
   - Nhập `GEMINI_API_KEY` của bạn vào file `.env`.

## 💻 Sử dụng

Chạy chương trình chính:
```bash
python -m app.main
```

## 📂 Cấu trúc thư mục

- `app/`: Mã nguồn chính.
- `src/`: Chứa dữ liệu dataset (.jsonl).
- `chroma_db/`: Cơ sở dữ liệu Vector (tự động tạo).

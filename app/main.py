import os
import json
import time
from app.core.database import DukyDatabase
from app.core.rag import DukyRAG

def main():
    print("🚀 Khởi chạy Duky AI RAG System...")
    
    try:
        db = DukyDatabase()
        rag = DukyRAG()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    # Kiểm tra nạp dữ liệu nếu DB trống
    if db.count() == 0:
        jsonl_path = os.getenv("JSONL_FILE_PATH", "./src/rag_chunks_v2.jsonl")
        if os.path.exists(jsonl_path):
            print(f"📦 Đang nạp dữ liệu từ {jsonl_path}...")
            # Logic nạp dữ liệu đơn giản cho lần đầu
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                chunks = [json.loads(line) for line in f if line.strip()]
            
            # (Ở đây có thể thêm logic embed batch nếu cần, 
            # nhưng để đơn giản ta giả định database đã được chuẩn bị 
            # hoặc sẽ được nạp qua một script ingest riêng)
            print("⚠️ Database đang trống. Bạn cần chạy lệnh ingest trước (tính năng đang phát triển).")
        else:
            print(f"⚠️ Không tìm thấy file dữ liệu tại {jsonl_path}")

    print(f"✅ Hệ thống sẵn sàng! (Hiện có {db.count()} chunks)")
    print("-" * 60)

    while True:
        question = input("❓ Nhập câu hỏi (gõ 'exit' để thoát): ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            break
        if not question:
            continue

        print("-" * 60)
        start_time = time.time()

        try:
            # 1. Embed & Query
            query_vec = rag.get_embedding(question)
            results = db.query(query_embeddings=[query_vec])
            
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dists = results['distances'][0]

            print("📎 Chunks tìm được:")
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                print(f"  [{i+1}] tool={meta['tool']} | score={1-dist:.2f}")
                print(f"       {doc[:100]}...")
            print()

            # 2. Generate (Streaming)
            context = "\n\n".join(docs)
            print("🤖 ", end="", flush=True)
            
            start_gen_time = time.time()
            stream = rag.generate_answer_stream(question, context)
            
            full_response = ""
            for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text
            
            end_time = time.time()
            
            print(f"\n\n⏱️ Thời gian xử lý: {end_time - start_time:.2f}s (Truy vấn: {start_gen_time - start_time:.2f}s | Sinh lời: {end_time - start_gen_time:.2f}s)")
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    main()

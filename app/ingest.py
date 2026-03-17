import json
import os
import time
from app.core.database import DukyDatabase
from app.core.rag import DukyRAG
from dotenv import load_dotenv

load_dotenv()

def build_embed_text(chunk):
    """Ghép content + keywords + questions để tìm kiếm chính xác hơn"""
    parts = [chunk['content']]
    kw = chunk['metadata'].get('keywords', [])
    qs = chunk['metadata'].get('questions', [])
    if kw: parts.append('Từ khóa: ' + ', '.join(kw))
    if qs: parts.append('Câu hỏi: ' + ' | '.join(qs))
    return ' '.join(parts)

def ingest_data():
    print("🧹 Khởi tạo Database và RAG Client...")
    db = DukyDatabase()
    rag = DukyRAG()
    
    jsonl_path = os.getenv("JSONL_FILE_PATH", "./src/rag_chunks_v2.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"❌ Không tìm thấy file {jsonl_path}")
        return

    print(f"📖 Đọc dữ liệu từ {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    existing_ids = db.get_existing_ids()
    new_chunks = [c for c in chunks if c['id'] not in existing_ids]

    if not new_chunks:
        print("✅ Dữ liệu đã đầy đủ, không cần nạp thêm.")
        return

    print(f"📦 Đang nạp {len(new_chunks)} chunks mới...")
    
    BATCH_SIZE = 10
    total = len(new_chunks)
    
    for i in range(0, total, BATCH_SIZE):
        batch = new_chunks[i:i+BATCH_SIZE]
        b_ids, b_docs, b_metas, b_embeds = [], [], [], []

        for chunk in batch:
            try:
                # 1. Tạo text để embed
                embed_text = build_embed_text(chunk)
                # 2. Gọi Embedding API
                embedding = rag.get_embedding(embed_text)
                
                b_ids.append(chunk['id'])
                b_docs.append(chunk['content'])
                b_metas.append({
                    'tool': chunk['metadata'].get('tool', ''),
                    'section': chunk['metadata'].get('section', ''),
                    'source': chunk['metadata'].get('source', 'jsonl')
                })
                b_embeds.append(embedding)
            except Exception as e:
                print(f"  ⚠️ Lỗi chunk {chunk['id']}: {e}")

        if b_ids:
            db.upsert_chunks(b_ids, b_docs, b_metas, b_embeds)
            
        print(f"  ✅ Tiến độ: {min(i + BATCH_SIZE, total)}/{total}")
        time.sleep(1)

    print("🎉 Hoàn tất nạp dữ liệu!")

if __name__ == "__main__":
    ingest_data()

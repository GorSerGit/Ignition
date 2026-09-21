import os
import sys
import time
import glob
import argparse
import fitz
from config import Config
from database import AHDatabase
from memory import AHMemory
from parser import parse_text_with_recovery

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc if page.get_text("text").strip()])
        doc.close()
        return text
    except Exception as e:
        print(f"Ошибка PDF {pdf_path}: {e}")
        return ""

def load_corpus(corpus_dir: str) -> str:
    if not os.path.exists(corpus_dir): return ""
    files = glob.glob(os.path.join(corpus_dir, "**", "*.pdf"), recursive=True) + \
            glob.glob(os.path.join(corpus_dir, "**", "*.txt"), recursive=True)
    
    texts = []
    for f in files:
        if f.endswith(".pdf"): texts.append(extract_text_from_pdf(f))
        elif f.endswith(".txt"):
            with open(f, "r", encoding="utf-8") as file: texts.append(file.read())
            
    full_text = "\n\n".join(texts)
    print(f"📚 Загружено слов: {len(full_text.split())}")
    return full_text

def main():
    parser = argparse.ArgumentParser(description="Воспламенение 1.0 | Построение АГ-памяти")
    parser.add_argument("--test", type=int, default=None, help="Режим тестирования: обработать только N первых групп")
    parser.add_argument("--reset", action="store_true", help="Очистить БД и хэши перед запуском")
    args = parser.parse_args()

    print("=" * 50)
    print("  ВОСПЛАМЕНЕНИЕ 1.0 | Вариант В: Клетка")
    print("=" * 50)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    db_path = os.path.join(Config.OUTPUT_DIR, "ah_memory.db")
    
    if args.reset and os.path.exists(db_path):
        os.remove(db_path)
        print("🗑 База данных очищена.")
        
    db = AHDatabase(db_path)
    memory = AHMemory(db)
    
    text = load_corpus(Config.INPUT_DIR)
    if not text:
        print("❌ Корпус пуст."); sys.exit(1)
        
    print("\n=== Запуск парсинга ===")
    start_time = time.time()
    parse_text_with_recovery(text, memory, test_limit=args.test)
    
    print("\n=== Дедупликация и статистика ===")
    db.merge_duplicate_links()
    stats = memory.db.get_stats()
    print(f"📊 ИТОГОВАЯ СТАТИСТИКА ГРАФА:")
    print(f"  S (Символы): {stats['S']}")
    print(f"  P (Факты):   {stats['P']}")
    print(f"  L (Связи):   {stats['L']}")
    print(f"⏱ Время работы: {time.time() - start_time:.1f} сек")
    
    db.close()

if __name__ == "__main__":
    main()

import os
import json
import sqlite3
from collections import defaultdict
from config import Config

# Путь к базе данных
DB_PATH = os.path.join(Config.OUTPUT_DIR, "ah_memory.db")

def get_node_label(conn, uid: str) -> str:
    """
    Умная функция для вывода UID в читаемый текст.
    Если это Символ (S) -> возвращает его первое слово (лемму).
    Если это Факт (P) -> находит его предикат и возвращает его (выделяем эмодзи 🟩).
    """
    # 1. Ищем в символах
    row_s = conn.execute("SELECT r_text FROM symbols WHERE uid = ?", (uid,)).fetchone()
    if row_s:
        texts = json.loads(row_s[0])
        return texts[0] if texts else uid
    
    # 2. Ищем в фактах (гиперсвязях)
    row_f = conn.execute("SELECT mt FROM facts WHERE uid = ?", (uid,)).fetchone()
    if row_f:
        mt = json.loads(row_f[0])
        pred_uid = mt.get("predicate_uid")
        if pred_uid:
            pred_row = conn.execute("SELECT r_text FROM symbols WHERE uid = ?", (pred_uid,)).fetchone()
            if pred_row:
                pred_texts = json.loads(pred_row[0])
                return f"🟩[{pred_texts[0]}]"  # Факты выделяем зеленым квадратом
        return f"🟩[Факт {uid[:6]}]"
        
    return f"[{uid[:6]}]"

def run_validation():
    if not os.path.exists(DB_PATH):
        print("❌ База данных не найдена. Запустите main.py --test 5")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    print("="*70)
    print("🛡 ВАЛИДАЦИЯ И ВИЗУАЛИЗАЦИЯ ГРАФА (Воспламенение 1.0)")
    print("="*70)
    
    # ==========================================
    # 1. ПОЛНАЯ ТЕКСТОВАЯ КАРТА ГРАФА
    # ==========================================
    print("\n🗺️ ПОЛНАЯ КАРТА ГРАФА (Текстовая визуализация)")
    print("-" * 70)
    
    # Вывод вершин (Символов)
    symbols = conn.execute("SELECT uid, r_text FROM symbols").fetchall()
    print(f"📍 ВЕРШИНЫ (Symbols) - Всего: {len(symbols)}")
    for s in symbols:
        texts = json.loads(s["r_text"])
        lemma = texts[0] if texts else "?"
        paradigm = ", ".join(texts[1:4]) + ("..." if len(texts) > 4 else "")
        print(f"  🔹 [{s['uid'][:8]}] {lemma}  (парадигма: {paradigm})")
        
    # Вывод связей (Раскрытие гиперграфов)
    print(f"\n🔗 СВЯЗИ (Links) - Всего: {conn.execute('SELECT COUNT(*) FROM links').fetchone()[0]}")
    links = conn.execute("SELECT * FROM links ORDER BY link_id, w DESC").fetchall()
    
    # Группируем связи по типу для удобства чтения
    links_by_type = defaultdict(list)
    for l in links:
        links_by_type[l["link_id"]].append(l)
        
    for link_type, group in links_by_type.items():
        print(f"\n  📎 Тип связи: {link_type} ({len(group)} шт.)")
        for l in group[:30]: # Ограничим вывод, если связей станет очень много
            src_label = get_node_label(conn, l["e1"])
            dst_label = get_node_label(conn, l["e2"])
            print(f"     {src_label} --({link_type}, w={l['w']:.2f}, cnt={l['count']})--> {dst_label}")
        if len(group) > 30:
            print(f"     ... и еще {len(group) - 30} связей этого типа.")

    # ==========================================
    # 2. ПРОВЕРКА ИНВАРИАНТОВ (ОАГ)
    # ==========================================
    print("\n" + "="*70)
    print("🔄 ПРОВЕРКА ИНВАРИАНТОВ (Ориентированный Ациклический Граф)")
    
    def has_cycle(link_type):
        adj = defaultdict(list)
        rows = conn.execute("SELECT e1, e2 FROM links WHERE link_id = ?", (link_type,)).fetchall()
        for r in rows: 
            adj[r["e1"]].append(r["e2"])
            # Заранее добавляем цели, чтобы defaultdict не менял размер словаря внутри dfs
            if r["e2"] not in adj:
                adj[r["e2"]] = [] 
                
        visited, rec_stack = set(), set()
        def dfs(v):
            visited.add(v); rec_stack.add(v)
            for n in adj[v]:
                if n not in visited:
                    if dfs(n): return True
                elif n in rec_stack: return True
            rec_stack.remove(v)
            return False
            
        # Итерируемся по статическому списку ключей
        for node in list(adj.keys()):
            if node not in visited:
                if dfs(node): return True
        return False

    isa_cycle = has_cycle("IS-A")
    follow_cycle = has_cycle("FOLLOW")
    print(f"  {'✅' if not isa_cycle else '❌'} Иерархии IS-A: {'Ациклические' if not isa_cycle else 'ЕСТЬ ЦИКЛЫ!'}")
    print(f"  {'✅' if not follow_cycle else '❌'} Эпизоды FOLLOW: {'Ациклические' if not follow_cycle else 'ЕСТЬ ЦИКЛЫ!'}")

    # ==========================================
    # 3. ПРОВЕРКА СТАТИСТИЧЕСКИХ ВЕСОВ
    # ==========================================
    print("\n⚖️ ПРОВЕРКА СТАТИСТИЧЕСКИХ ВЕСОВ:")
    all_links = conn.execute("SELECT * FROM links").fetchall()
    structural = {"IS-A", "FOLLOW", "PREDICATE"}
    weight_errors = 0
    for l in all_links:
        if l["link_id"] in structural:
            if l["w"] != 1.0: weight_errors += 1
        else:
            expected_w = l["count"] / (l["count"] + 1.0)
            if abs(l["w"] - expected_w) > 0.01: weight_errors += 1
            
    print(f"  {'✅ Все веса рассчитаны верно!' if weight_errors == 0 else f'❌ Ошибок в весах: {weight_errors}'}")

    # ==========================================
    # 4. СТАТИСТИКА
    # ==========================================
    print("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    stats = {
        "S (Символы)": conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
        "P (Факты)": conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
        "L (Связи)": conn.execute("SELECT COUNT(*) FROM links").fetchone()[0],
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    print("\n" + "="*70)
    print("✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА.")
    print("="*70)
    
    conn.close()

if __name__ == "__main__":
    run_validation()

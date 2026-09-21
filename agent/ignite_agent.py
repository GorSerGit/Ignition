#!/usr/bin/env python3
"""
Когнитивный агент на базе АГ-памяти.
ИСПРАВЛЕНО: Умный сбор фактов (Smart Context) для предотвращения информационного шума.
"""
import os
import sys
import json
import re
import pymorphy3
from collections import defaultdict
from config import Config
from database import AHDatabase
from memory import AHMemory
from ignition import IgnitionEngine
from parser import get_giga_client

morph = pymorphy3.MorphAnalyzer()

def perceive_text(text: str, memory: AHMemory) -> list[str]:
    words = re.findall(r'\b[а-яёa-z]+\b', text.lower())
    seed_uids = []
    stopwords = {'что', 'как', 'где', 'когда', 'почему', 'зачем', 'кто', 'какой', 'это', 'этот', 'все', 'она', 'он', 'оно', 'они', 'мы', 'вы', 'ты', 'не', 'и', 'а', 'в', 'на', 'с', 'по', 'для', 'от', 'к', 'из', 'является', 'такой'}
    
    for w in words:
        if w in stopwords or len(w) < 3: continue
        parsed = morph.parse(w)
        if parsed and parsed[0].tag.POS == 'NOUN':
            lemma = parsed[0].normal_form
            sym = memory.db.get_symbol_by_canonical(lemma)
            if not sym:
                for s in memory.db.get_all_symbols():
                    if lemma in [word.lower() for word in s.r_text]:
                        sym = s; break
            if sym and sym.uid not in seed_uids:
                seed_uids.append(sym.uid)
                
    if not seed_uids:
        print("  ⚠ Быстрый поиск не дал результатов. Запрашиваем БЯМ...")
        try:
            client = get_giga_client()
            resp = client.chat({
                "messages": [
                    {"role": "system", "content": "Извлеки из текста ключевые биологические термины (существительные в начальной форме). Верни ТОЛЬКО JSON-массив строк. Пример: ['митохондрия', 'АТФ']"},
                    {"role": "user", "content": text}
                ], "temperature": 0.1, "max_tokens": 100
            })
            terms = json.loads(re.search(r'\[.*\]', resp.choices[0].message.content).group(0))
            for term in terms:
                sym = memory.db.get_symbol_by_canonical(term.lower())
                if not sym:
                    for s in memory.db.get_all_symbols():
                        if term.lower() in [word.lower() for word in s.r_text]:
                            sym = s; break
                if sym and sym.uid not in seed_uids:
                    seed_uids.append(sym.uid)
        except Exception as e:
            print(f"  ❌ Ошибка БЯМ при восприятии: {e}")
    return seed_uids

def collect_facts_smart(seed_uids: list[str], activations: dict, db: AHDatabase, max_facts: int = 30) -> list[dict]:
    """
    🧠 УМНЫЙ СБОР ФАКТОВ: 
    Приоритет отдается фактам, где участвуют seed-узлы или узлы с высокой активацией.
    """
    seed_set = set(seed_uids)
    fact_scores = defaultdict(float)
    
    # 1. Жесткий приоритет: факты, где есть наши seed-узлы
    for uid in seed_uids:
        links = db.get_links_from(uid) + db.get_links_to(uid)
        for l in links:
            if l.e1_type == "P": fact_scores[l.e1] += 100.0
            if l.e2_type == "P": fact_scores[l.e2] += 100.0
            
    # 2. Приоритет: факты, где есть узлы с высокой активацией
    for uid, act in activations.items():
        if act < 0.4: continue 
        links = db.get_links_from(uid) + db.get_links_to(uid)
        for l in links:
            if l.e1_type == "P": fact_scores[l.e1] += act
            if l.e2_type == "P": fact_scores[l.e2] += act

    # Сортируем и берем ТОП-N фактов (максимум 30, чтобы LLM не захлебнулась)
    top_fact_uids = sorted(fact_scores.keys(), key=lambda x: -fact_scores[x])[:max_facts]
    
    facts = []
    for f_uid in top_fact_uids:
        row = db.conn.execute("SELECT * FROM facts WHERE uid = ?", (f_uid,)).fetchone()
        if not row: continue
        
        fact_data = dict(row)
        pred_links = db.get_links_from(f_uid)
        predicate = "?"
        roles = {}
        for pl in pred_links:
            if pl.link_id == "PREDICATE":
                sym_row = db.conn.execute("SELECT r_text FROM symbols WHERE uid=?", (pl.e2,)).fetchone()
                predicate = json.loads(sym_row[0])[0] if sym_row else "?"
            else:
                sym_row = db.conn.execute("SELECT r_text FROM symbols WHERE uid=?", (pl.e2,)).fetchone()
                if sym_row:
                    roles[pl.link_id] = json.loads(sym_row[0])[0]
                    
        facts.append({
            "predicate": predicate,
            "roles": roles,
            "text_span": json.loads(fact_data.get("pr", "{}")).get("text_span", "")
        })
        
    return facts

def main():
    db_path = os.path.join(Config.OUTPUT_DIR, "ah_memory.db")
    if not os.path.exists(db_path):
        print("❌ База данных не найдена."); return

    db = AHDatabase(db_path)
    memory = AHMemory(db)
    engine = IgnitionEngine(db)

    print("="*60)
    print("🧠 КОГНИТИВНЫЙ АГЕНТ (Воспламенение 1.0)")
    print("Введите вопрос. Для выхода: exit")
    print("="*60)

    while True:
        query = input("\n❓ Ваш вопрос: ").strip()
        if query.lower() in ('exit', 'выход', 'quit'): break
        if not query: continue

        print("\n--- ЭТАП 1: ВОСПРИЯТИЕ ---")
        seed_uids = perceive_text(query, memory)
        if not seed_uids:
            print("❌ Не удалось найти концепты в графе."); continue
        
        seed_labels = [engine._get_node_label(uid) for uid in seed_uids]
        print(f"✅ Найдены вершины: {seed_labels}")

        print("\n--- ЭТАП 2: ФОКУС АКТИВАЦИИ ---")
        result = engine.query(seed_uids)
        
        print("🔥 Трассировка воспламенения:")
        for line in result["trace_log"]:
            print(f"   {line}")

        print("\n--- ЭТАП 3: УМНЫЙ СБОР ФАКТОВ ---")
        facts = collect_facts_smart(seed_uids, result["activations"], db, max_facts=30)
        print(f"✅ Отсортировано и отобрано {len(facts)} самых релевантных фактов.")
        
        if not facts:
            print("❌ В графе не нашлось фактов для ответа."); continue

        print("\n--- ЭТАП 4: ГЕНЕРАЦИЯ ОТВЕТА (LLM) ---")
        context_text = ""
        for i, f in enumerate(facts):
            roles_str = ", ".join([f"{k}: {v}" for k, v in f["roles"].items()])
            context_text += f"Факт {i+1}: [{f['predicate']}] {roles_str}.\n"

        try:
            client = get_giga_client()
            response = client.chat({
                "messages": [
                    {"role": "system", "content": "Ты — биолог-эксперт. Отвечай на вопрос СТРОГО на основе предоставленных фактов. Не выдумывай. Отвечай кратко и по сути."},
                    {"role": "user", "content": f"Вопрос: {query}\n\nДоступные факты из памяти:\n{context_text}\n\nОтвет:"}
                ],
                "temperature": 0.2,
                "max_tokens": 500
            })
            answer = response.choices[0].message.content.strip()
            print(f"\n💡 ОТВЕТ АГЕНТА:\n{answer}")
        except Exception as e:
            print(f"❌ Ошибка генерации LLM: {e}")

    db.close()

if __name__ == "__main__":
    main()

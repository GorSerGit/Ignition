import os
import json
import re
import time
import hashlib
from gigachat import GigaChat
from config import Config
import requests.exceptions as req_exc

_giga_client = None
_last_model = None  # для отслеживания смены модели

def get_giga_client(model_name: str = None):
    global _giga_client, _last_model
    if model_name is None:
        model_name = Config.MODEL_NAME
    if _giga_client is not None and _last_model == model_name:
        return _giga_client

    credentials = Config.GIGACHAT_CREDENTIALS.strip()
    if not credentials:
        import base64
        auth_string = f"{Config.GIGACHAT_CLIENT_ID.strip()}:{Config.GIGACHAT_CLIENT_SECRET.strip()}"
        credentials = base64.b64encode(auth_string.encode()).decode()

    _giga_client = GigaChat(
        credentials=credentials,
        verify_ssl_certs=False,
        model=model_name,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        timeout=60,
    )
    _last_model = model_name
    return _giga_client

# 🧬 ЖЕСТКИЙ ПРОМПТ: Требуем от БЯМ начальную форму, чтобы избежать дубликатов
SYSTEM_PROMPT_GROUP = """Ты — строгий модуль извлечения фактов.
Из текста извлеки факты в виде JSON-МАССИВА.
КРИТИЧЕСКОЕ ПРАВИЛО: Все значения ролей (SUBJECT, OBJECT и т.д.) и предикат должны быть СТРОГО В НАЧАЛЬНОЙ ФОРМЕ (именительный падеж, единственное число для существительных, инфинитив для глаголов). 
Никаких окончаний, никаких множественных чисел! (Например: не "митохондрии", а "митохондрия"; не "производят", а "производить").

ОБЯЗАТЕЛЬНЫЕ поля:
"predicate": глагол в начальной форме
"template": один из [DISCOVER, PRODUCE, LOCATE, CAUSE_EFFECT, CREATE, CLASSIFY, INTERACT, CHANGE_STATE]
"roles": словарь {роль: значение в начальной форме}. Роли: SUBJECT, OBJECT, LOCATION, TIME, CAUSE, INSTRUMENT, MATERIAL, EFFECT.

Правила:
1. SUBJECT и OBJECT заполняй ВСЕГДА.
2. Для CLASSIFY: SUBJECT = частное, OBJECT = общее (это создаст связь IS-A).
3. Отвечай ТОЛЬКО валидным JSON-массивом, без markdown и пояснений.
Пример:
Текст: "Митохондрии производят АТФ."
Ответ: [{"predicate": "производить", "template": "PRODUCE", "roles": {"SUBJECT": "митохондрия", "OBJECT": "АТФ"}}]
"""

def get_chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def split_and_group(text: str, window: int = 3) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    groups = []
    for i in range(0, len(sentences), window):
        groups.append(' '.join(sentences[i:i+window]))
    return groups

def clean_llm_json(content: str) -> list:
    content = re.sub(r'^```(?:json)?\s*', '', content.strip())
    content = re.sub(r'\s*```$', '', content)
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if not match: return []
    json_str = re.sub(r',\s*([\]}])', r'\1', match.group(0)) # Fix trailing commas
    try:
        data = json.loads(json_str)
        return data if isinstance(data, list) else [data]
    except:
        return []

def parse_text_with_recovery(text: str, memory, test_limit: int = None, model_name: str = None):
    groups = split_and_group(text)
    if test_limit:
        groups = groups[:test_limit]
        print(f"🧪 ТЕСТОВЫЙ РЕЖИМ: Обрабатываем только {test_limit} групп.")
        
    total = len(groups)
    saved, skipped = 0, 0
    print(f"📝 Всего групп: {total}")
    
    for i, group in enumerate(groups):
        chunk_hash = get_chunk_hash(group)
        
        # 🛡 ПРОВЕРКА ПРОДОЛЖЕНИЯ (Crash Recovery)
        if memory.db.is_chunk_processed(chunk_hash):
            skipped += 1
            continue
            
        print(f"  [{i+1}/{total}] Парсинг: {group[:60]}...")
        
        facts = []
        for attempt in range(3):
            try:
                client = get_giga_client(model_name=model_name)
                response = client.chat({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_GROUP},
                        {"role": "user", "content": f"Текст: {group}"}
                    ],
                    "temperature": 0.1, 
                    "max_tokens": 2048,
                })
                facts = clean_llm_json(response.choices[0].message.content)
                break
            except Exception as e:
                print(f"    ⚠ Ошибка БЯМ (попытка {attempt+1}): {e}")
                time.sleep(2 ** attempt)
                
        for fact in facts:
            fact["_source_text"] = group
            if isinstance(fact.get("roles"), dict):
                try:
                    memory.ingest_parsed_fact(fact)
                    saved += 1
                except Exception as e:
                    print(f"    ❌ Ошибка сохранения: {e}")
                    
        # 🛡 ФИКСАЦИЯ ПРОГРЕССА В БД
        memory.db.mark_chunk_processed(chunk_hash)
        time.sleep(0.2)
        
    print(f"✅ Завершено. Сохранено фактов: {saved}, Пропущено (уже обработано): {skipped}")

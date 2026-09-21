#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-приложение «Воспламенение 1.0»
Flask + vis.js
Модификация: генерация ответа через Ollama (поддержка параметра model)
"""
import os
import sys
import json
import re
import threading
import logging
import requests
from datetime import datetime
from difflib import get_close_matches
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import networkx as nx

from config import Config, HYPERPARAMS
from database import AHDatabase
from memory import AHMemory
from ignition import IgnitionEngine
from parser import parse_text_with_recovery, get_giga_client
from ignite_agent import perceive_text, collect_facts_smart
import val

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key'
CORS(app)

db = None
memory = None
engine = None
db_path = None
corpus_text = None
parse_status = {"running": False, "progress": 0, "log": ""}
log_messages = []


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ
# =============================================================================
def clean_label(text):
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
    text = ''.join(ch for ch in text if ch.isprintable() and ord(ch) < 0x10000)
    if len(text) > 40:
        text = text[:37] + "..."
    return text


def add_log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    log_messages.append(f"[{ts}] [{level}] {msg}")
    if len(log_messages) > 1000:
        log_messages.pop(0)
    logger.info(msg)


def find_symbol_fuzzy(word: str):
    """4-уровневый поиск: canonical → r_text exact → подстрока → нечёткий."""
    if not db or not word:
        return None
    w = word.strip().lower()
    if not w:
        return None

    sym = db.get_symbol_by_canonical(w)
    if sym:
        return sym

    for s in db.get_all_symbols():
        if any(w == x.lower() for x in s.r_text):
            return s

    for s in db.get_all_symbols():
        if any(w in x.lower() for x in s.r_text):
            return s

    pool = []
    for s in db.get_all_symbols():
        for x in s.r_text:
            pool.append((x.lower(), s))
    if pool:
        matches = get_close_matches(w, [p[0] for p in pool], n=1, cutoff=0.65)
        if matches:
            for x, s in pool:
                if x == matches[0]:
                    add_log(f"🔍 Нечёткий поиск: '{word}' → '{matches[0]}'")
                    return s
    return None


def _build_trace_subgraph(uids, steps=0):
    """
    Строит подграф из указанных UID со ВСЕМИ связями между ними.
    steps=0 — без расширения, только сами узлы и связи между ними.
    Возвращает (nodes_data, edges_data).
    """
    if not db or not uids:
        return [], []

    uid_set = set(str(u) for u in uids)

    # Загружаем метки символов только для нужных узлов
    symbol_labels = {}
    for s in db.get_all_symbols():
        if s.uid in uid_set:
            label = s.r_text[0] if s.r_text else s.uid[:8]
            symbol_labels[s.uid] = clean_label(label) or s.uid[:8]

    # Загружаем факты
    facts_data = {}
    if uid_set:
        placeholders = ','.join(['?'] * len(uid_set))
        facts_rows = db.conn.execute(
            f"SELECT uid, mt FROM facts WHERE uid IN ({placeholders})",
            list(uid_set)
        ).fetchall()
        for uid, mt_json in facts_rows:
            try:
                facts_data[uid] = json.loads(mt_json)
            except Exception:
                pass

    # Формируем nodes_data
    nodes_data = []
    for uid in uid_set:
        if uid in symbol_labels:
            label = symbol_labels[uid]
            color = '#87CEEB'
            ntype = 'symbol'
        else:
            mt = facts_data.get(uid, {})
            pred_uid = mt.get("predicate_uid")
            roles = mt.get("roles", {})

            pred_label = symbol_labels.get(pred_uid, "?") if pred_uid else "?"
            subject_label = symbol_labels.get(roles.get("SUBJECT"), "") if roles.get("SUBJECT") else ""
            object_label = symbol_labels.get(roles.get("OBJECT"), "") if roles.get("OBJECT") else ""

            if subject_label and object_label:
                label = f"{pred_label}({subject_label}, {object_label})"
            elif subject_label:
                label = f"{pred_label}({subject_label})"
            elif pred_label != "?":
                label = f"F:{pred_label}"
            else:
                label = f"Fact:{uid[:6]}"

            label = clean_label(label) or f"Fact:{uid[:6]}"
            color = '#90EE90'
            ntype = 'fact'

        nodes_data.append({
            "id": str(uid),
            "label": label,
            "color": color,
            "type": ntype
        })

    # 🔥 Собираем ВСЕ связи между выбранными узлами ОДНИМ запросом
    edges_data = []
    seen_edges = set()

    if uid_set:
        placeholders = ','.join(['?'] * len(uid_set))
        links_rows = db.conn.execute(
            f"SELECT e1, e2, link_id, w FROM links WHERE e1 IN ({placeholders}) OR e2 IN ({placeholders})",
            list(uid_set) + list(uid_set)
        ).fetchall()

        for e1, e2, lid, w in links_rows:
            if e1 in uid_set and e2 in uid_set:
                edge_key = (str(e1), str(e2))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    try:
                        weight = float(w) if w else 0.5
                    except Exception:
                        weight = 0.5

                    link_type = str(lid) if lid else "LINK"
                    is_structural = link_type in ["PREDICATE", "SUBJECT", "OBJECT", "IS-A", "FOLLOW"]

                    edges_data.append({
                        "from": str(e1),
                        "to": str(e2),
                        "label": clean_label(link_type),
                        "width": float(weight) * 2.5 if is_structural else float(weight) * 1.5,
                        "color": {
                            "color": '#ff6b6b' if link_type == "PREDICATE" else
                                    '#51cf66' if link_type in ["SUBJECT", "OBJECT"] else
                                    '#ffd43b' if link_type == "IS-A" else
                                    '#74c0fc' if link_type == "FOLLOW" else
                                    '#868e96',
                            "highlight": '#ff9800'
                        },
                        "font": {
                            "color": '#e6edf3',
                            "size": 11,
                            "strokeWidth": 2,
                            "strokeColor": '#010409'
                        },
                        "arrows": "to" if link_type in ["SUBJECT", "OBJECT", "PREDICATE"] else "none"
                    })

    return nodes_data, edges_data


def _get_top_nodes(n=20):
    """Возвращает топ-N узлов по степени + их метки."""
    all_nodes = {}

    for s in db.get_all_symbols():
        label = s.r_text[0] if s.r_text else s.uid[:8]
        label = clean_label(label) or s.uid[:8]
        all_nodes[s.uid] = {"label": label, "type": "symbol"}

    sym_label_map = {uid: info["label"] for uid, info in all_nodes.items()}

    facts_rows = db.conn.execute("SELECT uid, mt FROM facts").fetchall()
    for uid, mt_json in facts_rows:
        if not uid:
            continue
        pred_label = "?"
        subject_label = ""
        object_label = ""
        try:
            mt = json.loads(mt_json)
            pred_uid = mt.get("predicate_uid")
            if pred_uid and pred_uid in sym_label_map:
                pred_label = sym_label_map[pred_uid]
            roles = mt.get("roles", {})
            if "SUBJECT" in roles and roles["SUBJECT"] in sym_label_map:
                subject_label = sym_label_map[roles["SUBJECT"]]
            if "OBJECT" in roles and roles["OBJECT"] in sym_label_map:
                object_label = sym_label_map[roles["OBJECT"]]
        except Exception:
            pass

        if subject_label and object_label:
            label = f"{pred_label}({subject_label}, {object_label})"
        elif subject_label:
            label = f"{pred_label}({subject_label})"
        elif pred_label != "?":
            label = f"F:{pred_label}"
        else:
            label = f"Fact:{uid[:6]}"

        label = clean_label(label) or uid[:6]
        all_nodes[uid] = {"label": label, "type": "fact"}

    degrees = {}
    try:
        rows = db.conn.execute("""
            SELECT e1 AS uid, COUNT(*) AS cnt FROM links GROUP BY e1
            UNION ALL
            SELECT e2 AS uid, COUNT(*) AS cnt FROM links GROUP BY e2
        """).fetchall()
        for uid, cnt in rows:
            degrees[uid] = degrees.get(uid, 0) + cnt
    except Exception as e:
        add_log(f"⚠ Ошибка подсчёта степеней: {e}", "ERROR")

    for uid in all_nodes:
        if uid not in degrees:
            degrees[uid] = 0

    sorted_uids = sorted(degrees.keys(), key=lambda u: -degrees[u])
    top_uids = sorted_uids[:n]
    top_nodes = {uid: all_nodes[uid] for uid in top_uids if uid in all_nodes}
    return top_nodes, degrees


# =============================================================================
# ГЕНЕРАЦИЯ ЧЕРЕЗ OLLAMA
# =============================================================================
def generate_with_ollama(prompt: str, model: str, temperature: float = 0.2, max_tokens: int = 500) -> str:
    """Отправляет запрос к локальному Ollama и возвращает сгенерированный текст."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        add_log(f"Ошибка вызова Ollama: {e}", "ERROR")
        return f"[Ошибка генерации: {e}]"


# =============================================================================
# МАРШРУТЫ
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load_db', methods=['POST'])
def load_db():
    global db, memory, engine, db_path
    data = request.get_json()
    path = data.get('path', Config.OUTPUT_DIR + "/ah_memory.db")
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": f"Файл не найден: {path}"}), 404
    try:
        db = AHDatabase(path)
        memory = AHMemory(db)
        engine = IgnitionEngine(db)
        db_path = path
        add_log(f"БД загружена: {path}")
        return jsonify({"status": "ok", "stats": db.get_stats(), "path": path})
    except Exception as e:
        add_log(f"Ошибка загрузки БД: {e}", "ERROR")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    if not db:
        return jsonify({"status": "error", "message": "БД не загружена"}), 400
    try:
        return jsonify({"status": "ok", "stats": db.get_stats()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/graph', methods=['GET'])
def get_graph():
    if not db:
        return jsonify({"status": "error", "message": "БД не загружена"}), 400
    search = request.args.get('search', '').strip().lower()
    try:
        symbols = db.get_all_symbols()
        G = nx.DiGraph()

        symbol_labels = {}
        for s in symbols:
            label = s.r_text[0] if s.r_text else s.uid[:8]
            label = clean_label(label) or s.uid[:8]
            G.add_node(s.uid, label=label, type="symbol")
            symbol_labels[s.uid] = label

        facts_rows = db.conn.execute("SELECT uid, mt FROM facts").fetchall()
        for row in facts_rows:
            uid = row[0]
            mt_json = row[1]
            if not uid:
                continue

            pred_label = "?"
            subject_label = ""
            object_label = ""
            try:
                mt = json.loads(mt_json)
                pred_uid = mt.get("predicate_uid")
                if pred_uid and pred_uid in symbol_labels:
                    pred_label = symbol_labels[pred_uid]
                roles = mt.get("roles", {})
                if "SUBJECT" in roles and roles["SUBJECT"] in symbol_labels:
                    subject_label = symbol_labels[roles["SUBJECT"]]
                if "OBJECT" in roles and roles["OBJECT"] in symbol_labels:
                    object_label = symbol_labels[roles["OBJECT"]]
            except Exception:
                pass

            if subject_label and object_label:
                label = f"{pred_label}({subject_label}, {object_label})"
            elif subject_label:
                label = f"{pred_label}({subject_label})"
            elif pred_label != "?":
                label = f"F:{pred_label}"
            else:
                label = f"Fact:{uid[:6]}"

            label = clean_label(label) or f"Fact:{uid[:6]}"
            G.add_node(uid, label=label, type="fact")

        links = db.get_all_links()
        for l in links:
            if l.e1 in G and l.e2 in G:
                try:
                    w = float(l.w) if l.w else 0.5
                except (ValueError, TypeError):
                    w = 0.5
                G.add_edge(l.e1, l.e2, label=str(l.link_id), weight=w)

        target_uid = None
        if search:
            for uid, lbl in symbol_labels.items():
                if search in lbl.lower():
                    target_uid = uid
                    break

            if not target_uid:
                for row in facts_rows:
                    uid = row[0]
                    mt_json = row[1]
                    try:
                        mt = json.loads(mt_json)
                        pred_uid = mt.get("predicate_uid")
                        if pred_uid and pred_uid in symbol_labels:
                            if search in symbol_labels[pred_uid].lower():
                                target_uid = uid
                                break
                    except Exception:
                        pass

        if target_uid:
            nodes_sub = set([target_uid])
            neighbors_with_weights = []
            for neighbor in list(G.successors(target_uid)) + list(G.predecessors(target_uid)):
                if neighbor != target_uid:
                    max_w = 0.0
                    if G.has_edge(target_uid, neighbor):
                        max_w = max(max_w, G.edges[target_uid, neighbor].get('weight', 0.5))
                    if G.has_edge(neighbor, target_uid):
                        max_w = max(max_w, G.edges[neighbor, target_uid].get('weight', 0.5))
                    neighbors_with_weights.append((neighbor, max_w))

            neighbors_with_weights.sort(key=lambda x: -x[1])
            MAX_SEARCH_NODES = 40
            for neighbor, w in neighbors_with_weights[:MAX_SEARCH_NODES - 1]:
                nodes_sub.add(neighbor)

            G = G.subgraph(nodes_sub).copy()
            add_log(f"🔍 Поиск '{search}': найдено {len(nodes_sub)} узлов вокруг {target_uid[:8]}")

        MAX_DISPLAY = 200
        if len(G) > MAX_DISPLAY and not search:
            nodes = list(G.nodes)[:MAX_DISPLAY]
            G = G.subgraph(nodes).copy()

        nodes_data = []
        for n in G.nodes:
            label = clean_label(str(G.nodes[n].get('label', n[:8]))) or n[:8]
            color = '#87CEEB' if G.nodes[n].get('type') == 'symbol' else '#90EE90'
            nodes_data.append({
                "id": str(n),
                "label": label,
                "color": color,
                "type": G.nodes[n].get('type')
            })

        edges_data = []
        for u, v, data in G.edges(data=True):
            edge_label = clean_label(str(data.get('label', '')))
            weight = data.get('weight', 0.5)
            if not isinstance(weight, (int, float)):
                weight = 0.5
            edges_data.append({
                "from": str(u), "to": str(v),
                "label": edge_label, "width": float(weight) * 2
            })

        return jsonify({"status": "ok", "nodes": nodes_data, "edges": edges_data})

    except Exception as e:
        add_log(f"Ошибка построения графа: {e}", "ERROR")
        import traceback
        add_log(traceback.format_exc(), "ERROR")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/ignite', methods=['POST'])
def ignite():
    if not db:
        return jsonify({"status": "error", "message": "БД не загружена"}), 400
    data = request.get_json()
    seed_words = data.get('seed', [])
    if not seed_words:
        return jsonify({"status": "error", "message": "Укажите seed-слова"}), 400

    seed_uids, not_found = [], []
    for word in seed_words:
        sym = find_symbol_fuzzy(word)
        if sym:
            seed_uids.append(sym.uid)
        else:
            not_found.append(word)

    if not seed_uids:
        return jsonify({
            "status": "error",
            "message": f"Ни один seed-узел не найден. Не найдены: {', '.join(not_found)}"
        }), 404

    result = engine.query_with_history(seed_uids)

    highlighted = set(seed_uids)
    for tick in result.get("history", []):
        for uid, x in tick.get("activations", {}).items():
            if x >= 0.05:
                highlighted.add(uid)

    return jsonify({
        "status": "ok",
        "history": result["history"],
        "final_working_memory": result["final_working_memory"],
        "seed_uids": seed_uids,
        "trace": result["trace_log"],
        "highlight_uids": list(highlighted),
        "not_found": not_found
    })


# ========== ИЗМЕНЁННЫЙ МАРШРУТ: использует Ollama ==========
@app.route('/api/ask', methods=['POST'])
def ask_agent():
    if not db:
        return jsonify({"status": "error", "message": "БД не загружена"}), 400
    data = request.get_json()
    question = data.get('question', '').strip()
    model_name = data.get('model', 'llama3.1:8b')   # модель для Ollama
    if not question:
        return jsonify({"status": "error", "message": "Вопрос пуст"}), 400
    try:
        seed_uids = perceive_text(question, memory)
        if not seed_uids:
            return jsonify({
                "status": "ok",
                "answer": "Не удалось найти концепты в графе.",
                "trace": ["Не найдено seed-узлов"],
                "facts": [], "highlight_uids": [],
                "trace_graph": {"nodes": [], "edges": []}
            })

        result = engine.query(seed_uids)
        trace = result["trace_log"]
        all_facts = collect_facts_smart(seed_uids, result["activations"], db, max_facts=30)

        if not all_facts:
            return jsonify({
                "status": "ok",
                "answer": "В графе не нашлось фактов для ответа.",
                "trace": trace, "facts": [],
                "highlight_uids": list(seed_uids),
                "trace_graph": {"nodes": [], "edges": []}
            })

        activations = result["activations"]
        for f in all_facts:
            role_uids = [v for v in f.get("roles", {}).values() if isinstance(v, str)]
            role_uids.append(f.get("predicate_uid", ""))
            f["_max_activation"] = max([activations.get(u, 0.0) for u in role_uids] + [0.0])

        all_facts.sort(key=lambda f: -f["_max_activation"])
        facts = all_facts[:5]

        ctx = ""
        for i, f in enumerate(facts):
            roles = ", ".join(f"{k}: {v}" for k, v in f["roles"].items())
            ctx += f"Факт {i+1}: [{f['predicate']}] {roles}.\n"

        prompt = f"Ты — биолог-эксперт. Отвечай СТРОГО на основе фактов. Кратко.\n\nВопрос: {question}\n\nФакты:\n{ctx}\n\nОтвет:"
        # Генерация через Ollama
        answer = generate_with_ollama(prompt, model_name, temperature=0.2, max_tokens=500)
        add_log(f"Ответ на: {question[:50]}...")

        hl = set(seed_uids)
        for f in facts:
            if f.get("uid"):
                hl.add(f["uid"])
            if f.get("predicate_uid"):
                hl.add(f["predicate_uid"])
            for role_name, role_value in f.get("roles", {}).items():
                if role_value and isinstance(role_value, str):
                    hl.add(role_value)

        trace_nodes, trace_edges = _build_trace_subgraph(list(hl), steps=0)

        return jsonify({
            "status": "ok", "answer": answer,
            "trace": trace, "facts": facts,
            "highlight_uids": list(hl),
            "trace_graph": {
                "nodes": trace_nodes,
                "edges": trace_edges,
                "seed_uids": [str(u) for u in seed_uids]
            }
        })
    except Exception as e:
        add_log(f"Ошибка агента: {e}", "ERROR")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/extract_fact', methods=['POST'])
def extract_fact():
    if not db or not memory:
        return jsonify({"status": "error", "message": "БД не загружена"}), 400
    data = request.get_json()
    sentence = data.get('sentence', '').strip()
    if not sentence:
        return jsonify({"status": "error", "message": "Предложение пустое"}), 400
    try:
        client = get_giga_client()
        prompt = (
            'Извлеки факт из предложения в JSON.\n'
            f'Предложение: "{sentence}"\n'
            'Верни ТОЛЬКО JSON:\n'
            '{"predicate":"инфинитив","subject":"кто/что",'
            '"object":"кого/что","template":"PRODUCE|DISCOVER|LOCATE|'
            'CAUSE_EFFECT|CREATE|CLASSIFY|INTERACT"}\n'
            'Если нельзя: {"error":"не удалось"}'
        )
        resp = client.chat({
            "messages": [
                {"role": "system", "content": "Лингвист. Только JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1, "max_tokens": 250
        })
        ans = resp.choices[0].message.content.strip()
        m = re.search(r'\{[^{}]+\}', ans, re.DOTALL)
        if not m:
            return jsonify({"status": "error", "message": f"Нет JSON: {ans[:200]}"}), 400
        fd = json.loads(m.group())
        if "error" in fd:
            return jsonify({"status": "error", "message": fd["error"]}), 400
        for k in ("predicate", "subject", "object"):
            if not fd.get(k):
                return jsonify({"status": "error", "message": f"Нет поля {k}"}), 400

        fact = {
            "predicate": fd["predicate"],
            "template": fd.get("template", "PRODUCE"),
            "roles": {"SUBJECT": fd["subject"], "OBJECT": fd["object"]},
            "_source_text": sentence
        }
        node = memory.ingest_parsed_fact(fact)
        if not node:
            return jsonify({"status": "error", "message": "Не удалось создать факт"}), 500

        add_log(f"✅ Факт: {fd['predicate']}({fd['subject']}, {fd['object']})")
        hl = [node.uid]
        for w in [fd["predicate"], fd["subject"], fd["object"]]:
            s = find_symbol_fuzzy(w)
            if s:
                hl.append(s.uid)

        return jsonify({
            "status": "ok", "fact": fd, "uid": node.uid,
            "highlight_uids": hl, "message": "Факт добавлен"
        })
    except Exception as e:
        add_log(f"Ошибка extract_fact: {e}", "ERROR")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/add_fact', methods=['POST'])
def add_fact_api():
    if not db or not memory:
        return jsonify({"status": "error", "message": "БД не загружена"}), 400
    data = request.get_json()
    pred = data.get('predicate', '').strip()
    subj = data.get('subject', '').strip()
    obj = data.get('object', '').strip()
    tmpl = data.get('template', 'PRODUCE')
    if not pred or not subj or not obj:
        return jsonify({"status": "error", "message": "Заполните все поля"}), 400
    fact = {
        "predicate": pred, "template": tmpl,
        "roles": {"SUBJECT": subj, "OBJECT": obj},
        "_source_text": "Ручной ввод"
    }
    try:
        node = memory.ingest_parsed_fact(fact)
        if node:
            add_log(f"✅ Факт: {pred}({subj}, {obj})")
            hl = [node.uid]
            for w in [pred, subj, obj]:
                s = find_symbol_fuzzy(w)
                if s:
                    hl.append(s.uid)
            return jsonify({
                "status": "ok", "uid": node.uid,
                "message": "Факт добавлен", "highlight_uids": hl
            })
        return jsonify({"status": "error", "message": "Не удалось"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/parse', methods=['POST'])
def start_parse():
    global corpus_text, parse_status
    if not db:
        return jsonify({"status": "error", "message": "БД не загружена"}), 400
    if parse_status["running"]:
        return jsonify({"status": "error", "message": "Уже запущен"}), 400
    data = request.get_json()
    text = data.get('text', '') or corpus_text
    if not text:
        return jsonify({"status": "error", "message": "Нет текста"}), 400
    limit = data.get('limit', 0)
    parse_status = {"running": True, "progress": 0, "log": "Запуск...\n"}

    def run():
        global parse_status
        try:
            import io as _io
            from contextlib import redirect_stdout
            f = _io.StringIO()
            with redirect_stdout(f):
                parse_text_with_recovery(text, memory, test_limit=limit or None)
            parse_status["log"] = f.getvalue()
            parse_status["progress"] = 100
            add_log("Парсинг завершён")
        except Exception as e:
            parse_status["log"] = f"Ошибка: {e}\n"
            add_log(f"Ошибка парсинга: {e}", "ERROR")
        finally:
            parse_status["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "ok", "message": "Парсинг запущен"})


@app.route('/api/parse_status', methods=['GET'])
def parse_status_ep():
    return jsonify(parse_status)


@app.route('/api/hyperparams', methods=['GET', 'POST'])
def hyperparams():
    if request.method == 'GET':
        return jsonify({"status": "ok", "params": HYPERPARAMS})
    data = request.get_json()
    for k, v in data.items():
        if k in HYPERPARAMS:
            HYPERPARAMS[k] = float(v)
    if engine:
        for k, v in HYPERPARAMS.items():
            if hasattr(engine, k):
                setattr(engine, k, v)
    return jsonify({"status": "ok"})


@app.route('/api/validate', methods=['POST'])
def validate():
    if not db:
        return jsonify({"status": "error", "message": "БД не загружена"}), 400
    try:
        import io as _io
        from contextlib import redirect_stdout
        f = _io.StringIO()
        old = val.DB_PATH
        val.DB_PATH = db_path
        try:
            with redirect_stdout(f):
                val.run_validation()
        finally:
            val.DB_PATH = old
        return jsonify({"status": "ok", "output": f.getvalue()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify({"logs": log_messages[-200:]})


@app.route('/api/load_corpus', methods=['POST'])
def load_corpus():
    global corpus_text
    data = request.get_json()
    path = data.get('path', '')
    if not path or not os.path.exists(path):
        return jsonify({"status": "error", "message": "Нет папки"}), 400
    from main import load_corpus as lcf
    text = lcf(path)
    if not text:
        return jsonify({"status": "error", "message": "Корпус пуст"}), 400
    corpus_text = text
    add_log(f"Корпус: {len(text.split())} слов")
    return jsonify({
        "status": "ok", "word_count": len(text.split()),
        "preview": text[:500]
    })


# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    default_db = os.path.join(Config.OUTPUT_DIR, "ah_memory.db")
    if os.path.exists(default_db):
        try:
            db = AHDatabase(default_db)
            memory = AHMemory(db)
            engine = IgnitionEngine(db)
            db_path = default_db
            print(f"✅ Автозагрузка БД: {default_db}")
            add_log(f"Автозагрузка БД: {default_db}")
        except Exception as e:
            print(f"❌ Ошибка автозагрузки: {e}")
    else:
        print(f"⚠️ БД не найдена: {default_db}")
    app.run(host='0.0.0.0', port=5000, debug=True)

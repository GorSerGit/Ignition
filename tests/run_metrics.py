#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_metrics.py — расчёт метрик для хакатона «Воспламенение 1.0».
Без диагностики графа (M1, M3 считаются отдельно — они требуют
ручной разметки и сборщика мусора соответственно).

Прогоняет RAG (FAISS + Ollama) и АГ-память (Flask /api/ask) по датасету,
оценивает ответы через GigaChat и считает:
  - базовые: completeness, accuracy, explainability, factual_consistency
  - производные: delta_explainability, delta_factual, hallucination
  - M2: ExplainScore (глубина трассировки × correct)
  - M4: Δ_explainability (trace_score_AG − trace_score_RAG)
  - M5: RobustnessGain (устойчивость между моделями)

Запуск:
    python run_metrics.py --models qwen2.5:7b qwen2.5:14b qwen2.5:32b --limit 20
    python run_metrics.py --models qwen2.5:7b --limit 5   # быстрый тест

Изменения:
  - chain_depth теперь использует неориентированный BFS
    (в графе рёбра направлены от фактов к символам, поэтому
     направленный BFS от seed-узлов давал depth=0);
  - M2 учитывает правильность ответа (correct_i) через оценку судьи.
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ─── RAG-зависимости ───
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("⚠️ faiss / sentence-transformers не установлены — RAG пропущен.")

# ─── GigaChat-судья ───
try:
    from parser import get_giga_client
    HAS_JUDGE = True
except ImportError:
    HAS_JUDGE = False
    print("⚠️ get_giga_client недоступен — оценки судьи будут заглушкой 3.0")


# ═══════════════════════════════════════════════════════════════
# 1. OLLAMA КЛИЕНТ
# ═══════════════════════════════════════════════════════════════
OLLAMA_URL = "http://localhost:11434"
AG_API_URL = "http://localhost:5000/api/ask"


def ollama_generate(model: str, prompt: str,
                    temperature: float = 0.2, max_tokens: int = 500) -> str:
    """Запрос к Ollama /api/chat."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[Ollama error: {e}]"


def ollama_check(model: str) -> bool:
    """Проверяет, что модель установлена."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        return model in names or any(n.startswith(model.split(":")[0]) for n in names)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# 2. RAG-СИСТЕМА
# ═══════════════════════════════════════════════════════════════
class RAGSystem:
    def __init__(self, rag_dir: str, model: str):
        self.model = model
        idx = os.path.join(rag_dir, "faiss_index.bin")
        chk = os.path.join(rag_dir, "chunks.pkl")
        emb = os.path.join(rag_dir, "embedding_model")

        if not (os.path.exists(idx) and os.path.exists(chk) and os.path.exists(emb)):
            raise FileNotFoundError(f"RAG-файлы не найдены в {rag_dir}")

        self.index = faiss.read_index(idx)
        with open(chk, "rb") as f:
            self.chunks = pickle.load(f)
        self.embedder = SentenceTransformer(emb)

    def answer(self, question: str, k: int = 3, max_tokens: int = 250) -> str:
        emb = self.embedder.encode([question], convert_to_numpy=True)
        _, indices = self.index.search(emb, k)
        chunks = [self.chunks[i] for i in indices[0]]
        trimmed = [" ".join(c.split()[:120]) for c in chunks]
        ctx = "\n---\n".join(trimmed)[:3500]
        prompt = f"Контекст:\n{ctx}\n\nВопрос: {question}\n\nОтвет:"
        return ollama_generate(self.model, prompt, temperature=0.6, max_tokens=max_tokens)


# ═══════════════════════════════════════════════════════════════
# 3. АГ-СИСТЕМА (через Flask API)
# ═══════════════════════════════════════════════════════════════
class AGSystem:
    def __init__(self, url: str = AG_API_URL):
        self.url = url

    def ask(self, question: str, model: str) -> Dict:
        """Возвращает полный JSON-ответ агента."""
        try:
            r = requests.post(
                self.url,
                json={"question": question, "model": model},
                timeout=180,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "error", "answer": f"[AG error: {e}]",
                    "trace_graph": {"nodes": [], "edges": [], "seed_uids": []}}

    def answer(self, question: str, model: str) -> str:
        return self.ask(question, model).get("answer", "")


# ═══════════════════════════════════════════════════════════════
# 4. СУДЬЯ GigaChat
# ═══════════════════════════════════════════════════════════════
class Judge:
    def __init__(self):
        self.client = get_giga_client() if HAS_JUDGE else None

    def evaluate(self, question: str, answer: str,
                 reference: str) -> Dict[str, float]:
        default = {"completeness": 3.0, "accuracy": 3.0,
                   "explainability": 3.0, "factual_consistency": 3.0}
        if not self.client:
            return default

        prompt = (
            "Ты — эксперт по оценке ответов ИИ-систем. "
            "Оцени ответ по четырём критериям (1–5). Выдай только JSON:\n"
            '{"completeness": число, "accuracy": число, '
            '"explainability": число, "factual_consistency": число}\n\n'
            f"Вопрос: {question}\n"
            f"Эталон: {reference}\n"
            f"Ответ: {answer}\n"
            "Оценка:"
        )
        try:
            resp = self.client.chat({
                "messages": [
                    {"role": "system",
                     "content": "Ты — строгий эксперт. Отвечай только JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            })
            raw = resp.choices[0].message.content.strip()
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                scores = json.loads(m.group())
                for k in default:
                    scores.setdefault(k, 3.0)
                return scores
        except Exception as e:
            print(f"   ⚠️ Judge error: {e}")
        return default


# ═══════════════════════════════════════════════════════════════
# 5. M2 и M4: trace_score и chain_depth
# ═══════════════════════════════════════════════════════════════
def trace_score(response_json: Dict) -> float:
    """1.0 если в ответе есть непустой trace_graph, иначе 0."""
    tg = response_json.get("trace_graph", {})
    nodes = tg.get("nodes", [])
    edges = tg.get("edges", [])
    return 1.0 if (nodes and edges) else 0.0


def chain_depth(response_json: Dict) -> int:
    """
    Длина самой длинной цепочки в trace_graph.

    ИСПРАВЛЕНО: неориентированный BFS.
    В графе АГ-памяти рёбра идут от фактов к символам
    (fact --SUBJECT--> symbol), поэтому направленный BFS
    от seed-узлов давал depth=0. Неориентированный обход
    корректно проходит по всем связям вокруг seed.
    """
    tg = response_json.get("trace_graph", {})
    nodes = tg.get("nodes", [])
    edges = tg.get("edges", [])
    if not nodes:
        return 0

    # Строим НЕориентированный список смежности
    adj = {}
    for e in edges:
        u, v = e.get("from"), e.get("to")
        if u is None or v is None:
            continue
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    seeds = set(tg.get("seed_uids", []))
    if not seeds:
        return 1

    visited = {}
    queue = [(s, 0) for s in seeds]
    max_depth = 0
    while queue:
        node, d = queue.pop(0)
        if node in visited:
            continue
        visited[node] = d
        max_depth = max(max_depth, d)
        for nxt in adj.get(node, []):
            if nxt not in visited:
                queue.append((nxt, d + 1))
    return max_depth


def is_correct(ag_scores: Dict[str, float]) -> float:
    """
    Оценка correct_i ∈ [0, 1] для M2.
    Считаем ответ правильным, если судья поставил factual_consistency ≥ 3.5.
    Иначе — 0 (вклад вопроса в M2 обнуляется).
    """
    return 1.0 if ag_scores.get("factual_consistency", 0.0) >= 3.5 else 0.0


# ═══════════════════════════════════════════════════════════════
# 6. ОСНОВНОЙ ПРОГОН
# ═══════════════════════════════════════════════════════════════
def run_for_model(model: str, dataset_path: str, limit: Optional[int],
                  rag_dir: str, judge: Judge):
    print(f"\n{'='*70}")
    print(f"🚀 МОДЕЛЬ: {model}")
    print(f"{'='*70}")

    df = pd.read_csv(dataset_path)
    if limit:
        df = df.head(limit)

    # ── RAG ──
    rag = None
    if HAS_RAG:
        try:
            rag = RAGSystem(rag_dir, model)
            print(f"✅ RAG загружен из {rag_dir}")
        except Exception as e:
            print(f"⚠️ RAG не загрузился: {e}")

    ag = AGSystem()

    rows = []
    for i, row in df.iterrows():
        q = row["question"]
        ref = row["reference_answer"]
        print(f"\n  [{i+1}/{len(df)}] {q[:60]}...")

        # RAG
        rag_ans = rag.answer(q) if rag else "[RAG недоступен]"
        rag_scores = judge.evaluate(q, rag_ans, ref)
        rag_trace = 0.0
        rag_depth = 0

        # АГ
        ag_resp = ag.ask(q, model)
        ag_ans = ag_resp.get("answer", "")
        ag_scores = judge.evaluate(q, ag_ans, ref)
        ag_trace = trace_score(ag_resp)
        ag_depth = chain_depth(ag_resp)
        correct = is_correct(ag_scores)

        rows.append({
            "question": q,
            "reference": ref,
            "rag_answer": rag_ans,
            "ag_answer": ag_ans,

            "rag_completeness": rag_scores["completeness"],
            "rag_accuracy": rag_scores["accuracy"],
            "rag_explainability": rag_scores["explainability"],
            "rag_factual_consistency": rag_scores["factual_consistency"],

            "ag_completeness": ag_scores["completeness"],
            "ag_accuracy": ag_scores["accuracy"],
            "ag_explainability": ag_scores["explainability"],
            "ag_factual_consistency": ag_scores["factual_consistency"],

            "rag_trace_score": rag_trace,
            "ag_trace_score": ag_trace,
            "rag_chain_depth": rag_depth,
            "ag_chain_depth": ag_depth,
            "ag_correct": correct,
        })

        print(f"     RAG expl={rag_scores['explainability']:.1f}  |  "
              f"AG expl={ag_scores['explainability']:.1f}  "
              f"trace={ag_trace:.0f}  depth={ag_depth}  correct={correct:.0f}")

    df_res = pd.DataFrame(rows)

    # ── M2: ExplainScore (упрощённый по постановке) ──
    # M2 = (1/N) · Σ [correct_i · (d_i / d_max) · trace_complete_i]
    d_max = 6.0
    m2_per_question = (
        df_res["ag_correct"]
        * (df_res["ag_chain_depth"] / d_max)
        * df_res["ag_trace_score"]
    )
    m2_ag = m2_per_question.mean()

    # ── Агрегация по модели ──
    summary = {
        "model": model,
        "N_questions": len(df_res),

        "rag_completeness": df_res["rag_completeness"].mean(),
        "rag_accuracy": df_res["rag_accuracy"].mean(),
        "rag_explainability": df_res["rag_explainability"].mean(),
        "rag_factual_consistency": df_res["rag_factual_consistency"].mean(),

        "ag_completeness": df_res["ag_completeness"].mean(),
        "ag_accuracy": df_res["ag_accuracy"].mean(),
        "ag_explainability": df_res["ag_explainability"].mean(),
        "ag_factual_consistency": df_res["ag_factual_consistency"].mean(),

        "delta_explainability": (df_res["ag_explainability"].mean()
                                 - df_res["rag_explainability"].mean()),
        "delta_factual": (df_res["ag_factual_consistency"].mean()
                          - df_res["rag_factual_consistency"].mean()),
        "hallucination_rag": 6.0 - df_res["rag_factual_consistency"].mean(),
        "hallucination_ag": 6.0 - df_res["ag_factual_consistency"].mean(),
        "delta_hallucination": ((6.0 - df_res["rag_factual_consistency"].mean())
                                - (6.0 - df_res["ag_factual_consistency"].mean())),

        "M2_rag": 0.0,
        "M2_ag": m2_ag,

        "M4_delta_trace": (df_res["ag_trace_score"].mean()
                           - df_res["rag_trace_score"].mean()),
        "M4_delta_expl": (df_res["ag_explainability"].mean()
                          - df_res["rag_explainability"].mean()),

        "avg_chain_depth_ag": df_res["ag_chain_depth"].mean(),
        "correct_rate_ag": df_res["ag_correct"].mean(),
    }

    return summary, df_res


# ═══════════════════════════════════════════════════════════════
# 7. КРАСИВЫЙ ВЫВОД
# ═══════════════════════════════════════════════════════════════
def print_final_report(summaries: List[Dict]):
    print("\n\n" + "█" * 70)
    print("█" + " " * 22 + "🔥 ВОСПЛАМЕНЕНИЕ 1.0 — ИТОГИ" + " " * 20 + "█")
    print("█" * 70)

    print("\n📊 СРАВНЕНИЕ RAG vs АГ по моделям")
    header = f"{'Метрика':<26} " + " ".join(f"{s['model']:<14}" for s in summaries)
    print(header)
    print("-" * len(header))

    metrics = [
        ("N вопросов", "N_questions", "{:.0f}"),
        ("Completeness RAG", "rag_completeness", "{:.2f}"),
        ("Completeness AG", "ag_completeness", "{:.2f}"),
        ("Accuracy RAG", "rag_accuracy", "{:.2f}"),
        ("Accuracy AG", "ag_accuracy", "{:.2f}"),
        ("Explainability RAG", "rag_explainability", "{:.2f}"),
        ("Explainability AG", "ag_explainability", "{:.2f}"),
        ("Factual RAG", "rag_factual_consistency", "{:.2f}"),
        ("Factual AG", "ag_factual_consistency", "{:.2f}"),
        ("Δ Explainability", "delta_explainability", "{:+.2f}"),
        ("Δ Factual", "delta_factual", "{:+.2f}"),
        ("Hallucination RAG", "hallucination_rag", "{:.2f}"),
        ("Hallucination AG", "hallucination_ag", "{:.2f}"),
        ("Δ Hallucination", "delta_hallucination", "{:+.2f}"),
        ("M2 AG", "M2_ag", "{:.3f}"),
        ("M4 Δ trace", "M4_delta_trace", "{:+.2f}"),
        ("Ср. длина цепочки", "avg_chain_depth_ag", "{:.2f}"),
        ("Correct rate AG", "correct_rate_ag", "{:.2f}"),
    ]

    for label, key, fmt in metrics:
        vals = " ".join(f"{fmt.format(s.get(key, 0)):<14}" for s in summaries)
        print(f"{label:<26} {vals}")

    # ── M5: RobustnessGain ──
    robustness_gain = 0.0
    if len(summaries) >= 2:
        print("\n📊 M5: RobustnessGain (устойчивость к классу БЯМ)")
        sorted_s = sorted(summaries, key=lambda x: x["model"])
        small = sorted_s[0]
        large = sorted_s[-1]

        f1_ag_small = small["M2_ag"]
        f1_ag_large = large["M2_ag"]
        f1_rag_small = small["rag_factual_consistency"] / 5.0
        f1_rag_large = large["rag_factual_consistency"] / 5.0

        robustness_gain = (f1_ag_large - f1_ag_small) - (f1_rag_large - f1_rag_small)
        print(f"   Модель small:  {small['model']}")
        print(f"   Модель large:  {large['model']}")
        print(f"   M2 AG small:   {f1_ag_small:.3f}")
        print(f"   M2 AG large:   {f1_ag_large:.3f}")
        print(f"   RAG small:     {f1_rag_small:.3f}")
        print(f"   RAG large:     {f1_rag_large:.3f}")
        print(f"   RobustnessGain = {robustness_gain:+.3f}   "
              f"{'✅ гипотеза подтверждена' if robustness_gain > 0 else '❌ гипотеза опровергнута'}")

    # ── Частичная оценка (M2 + M4 + M5) ──
    print("\n" + "━" * 70)
    print("⚠️ M1 (F1 ролей) и M3 (GC) не вычисляются в этой версии —")
    print("   они требуют ручной разметки и сборщика мусора соответственно.")
    print("━" * 70)

    m2 = float(np.mean([s["M2_ag"] for s in summaries]))
    m4 = float(np.mean([s["M4_delta_expl"] for s in summaries]))
    m5 = robustness_gain

    partial = 0.30 * m2 + 0.20 * m4 + 0.15 * m5
    max_partial = 0.30 + 0.20 + 0.15

    print(f"🎯 Частичный балл (M2 + M4 + M5):")
    print(f"   0.30·{m2:.3f} + 0.20·{m4:.3f} + 0.15·{m5:.3f} = {partial:.3f}")
    print(f"   Из максимума {max_partial:.2f} (без M1 и M3)")
    if max_partial > 0:
        print(f"   Нормированный: {partial / max_partial:.3f}")
    print("━" * 70)


# ═══════════════════════════════════════════════════════════════
# 8. ГЛАВНАЯ
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"],
                        help="Модели Ollama для тестирования")
    parser.add_argument("--dataset", default="questions_dataset.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rag_dir", default="rag_offline_package")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"metrics_{ts}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Результаты: {args.output_dir}")

    # ── Проверка окружения ──
    print("\n🔍 Проверка окружения...")
    try:
        r = requests.get("http://localhost:5000/api/stats", timeout=3)
        stats = r.json().get("stats", {})
        if stats.get("P", 0) == 0:
            print("   ❌ БД пуста (P=0). Запусти 'python main.py'")
            sys.exit(1)
        print(f"   ✅ Flask (АГ) отвечает: S={stats.get('S', 0)}, "
              f"P={stats.get('P', 0)}, L={stats.get('L', 0)}")
    except Exception as e:
        print(f"   ❌ Flask не отвечает: {e}")
        print("   Запусти 'python app.py' в отдельном окне")
        sys.exit(1)

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        print(f"   ✅ Ollama отвечает, моделей: {len(r.json().get('models', []))}")
    except Exception:
        print("   ❌ Ollama не отвечает. Запусти 'ollama serve'")
        sys.exit(1)

    judge = Judge()
    summaries = []

    for model in args.models:
        if not ollama_check(model):
            print(f"\n⚠️ Модель {model} не найдена в Ollama. Пропускаю.")
            continue

        try:
            summary, df_res = run_for_model(
                model, args.dataset, args.limit, args.rag_dir, judge
            )
            summaries.append(summary)

            detail_path = os.path.join(
                args.output_dir, f"details_{model.replace(':', '_')}.csv"
            )
            df_res.to_csv(detail_path, index=False, encoding="utf-8")
            print(f"   💾 Детали: {detail_path}")

        except Exception as e:
            print(f"❌ Ошибка на модели {model}: {e}")
            import traceback
            traceback.print_exc()

    if not summaries:
        print("\n❌ Не удалось прогнать ни одну модель.")
        sys.exit(1)

    print_final_report(summaries)

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n💾 Сводка: {summary_path}")

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"models": summaries}, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON:   {json_path}")


if __name__ == "__main__":
    main()

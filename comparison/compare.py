#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение RAG и АГ-системы с использованием одной модели Ollama.
Судья: GigaChat.
Запуск: python compare.py --dataset questions.csv --model llama3.1:8b --limit 10
"""

import os
import json
import time
import pickle
import argparse
import requests
import pandas as pd
from typing import Dict, List, Optional

import faiss
from sentence_transformers import SentenceTransformer

# Импортируем GigaChat-функцию из parser (она использует Config)
from parser import get_giga_client

# =============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ OLLAMA
# =============================================================================
def ollama_generate(prompt: str, model: str, temperature: float = 0.2, max_tokens: int = 500) -> str:
    """Генерация через Ollama API."""
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
        print(f"Ошибка Ollama: {e}")
        return f"[Ошибка генерации: {e}]"

# =============================================================================
# 2. RAG-СИСТЕМА (использует Ollama)
# =============================================================================

class RAGSystem:
    def __init__(self, index_path: str, chunks_path: str, embed_model_path: str, model_name: str):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        self.embed_model = SentenceTransformer(embed_model_path)
        self.model_name = model_name   # модель для Ollama

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        emb = self.embed_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(emb, k)
        return [self.chunks[i] for i in indices[0]]

    def answer(self, question: str, k: int = 3, max_new_tokens: int = 250) -> str:
        raw_chunks = self.retrieve(question, k)
        trimmed = [' '.join(chunk.split()[:120]) for chunk in raw_chunks]
        context = "\n---\n".join(trimmed)
        if len(context) > 3500:
            context = context[:3500] + "..."
        prompt = f"""Контекст:
{context}

Вопрос: {question}

Ответ:"""
        return ollama_generate(prompt, self.model_name, temperature=0.6, max_tokens=max_new_tokens)

# =============================================================================
# 3. КЛИЕНТ ДЛЯ АГ-СИСТЕМЫ (передаёт модель Ollama)
# =============================================================================

class AGSystem:
    def __init__(self, api_url: str = "http://localhost:5000/api/ask"):
        self.url = api_url

    def answer(self, question: str, model: str) -> str:
        try:
            resp = requests.post(self.url, json={"question": question, "model": model}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("answer", "АГ-система не вернула ответа.")
        except Exception as e:
            return f"Ошибка АГ: {e}"

# =============================================================================
# 4. СУДЬЯ – GigaChat
# =============================================================================

class JudgeGigaChat:
    def __init__(self, model_name: str = None):
        self.client = get_giga_client(model_name)

    def evaluate(self, question: str, answer: str, reference: str) -> Dict[str, float]:
        prompt = f"""Ты — эксперт по оценке ответов ИИ-систем. Оцени ответ по четырём критериям (каждый от 1 до 5, где 5 — наилучший). Выдай только JSON:
{{"completeness": число, "accuracy": число, "explainability": число, "factual_consistency": число}}

Вопрос: {question}

Эталонный ответ: {reference}

Оцениваемый ответ: {answer}

Оценка:"""
        try:
            resp = self.client.chat({
                "messages": [
                    {"role": "system", "content": "Ты — строгий эксперт. Отвечай только JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            })
            raw = resp.choices[0].message.content.strip()
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                for key in ["completeness", "accuracy", "explainability", "factual_consistency"]:
                    if key not in scores:
                        scores[key] = 3.0
                return scores
            return {"completeness": 3.0, "accuracy": 3.0, "explainability": 3.0, "factual_consistency": 3.0}
        except Exception as e:
            print(f"Ошибка оценки GigaChat: {e}")
            return {"completeness": 3.0, "accuracy": 3.0, "explainability": 3.0, "factual_consistency": 3.0}

# =============================================================================
# 5. ОСНОВНОЙ ПРОЦЕСС
# =============================================================================

def run_comparison(dataset_path: str, rag: RAGSystem, ag: AGSystem,
                   judge_model: Optional[str], output_file: str, limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    if limit:
        df = df.head(limit)

    judge = JudgeGigaChat(judge_model)
    results = []

    for idx, row in df.iterrows():
        q = row['question']
        ref = row['reference_answer']
        print(f"Вопрос {idx+1}/{len(df)}: {q[:60]}...")

        rag_ans = rag.answer(q)
        ag_ans = ag.answer(q, rag.model_name)   # та же модель Ollama
        time.sleep(0.5)

        scores_rag = judge.evaluate(q, rag_ans, ref)
        scores_ag = judge.evaluate(q, ag_ans, ref)

        results.append({
            "question": q,
            "reference": ref,
            "rag_answer": rag_ans,
            "ag_answer": ag_ans,
            "rag_completeness": scores_rag.get("completeness", 3.0),
            "rag_accuracy": scores_rag.get("accuracy", 3.0),
            "rag_explainability": scores_rag.get("explainability", 3.0),
            "rag_factual_consistency": scores_rag.get("factual_consistency", 3.0),
            "ag_completeness": scores_ag.get("completeness", 3.0),
            "ag_accuracy": scores_ag.get("accuracy", 3.0),
            "ag_explainability": scores_ag.get("explainability", 3.0),
            "ag_factual_consistency": scores_ag.get("factual_consistency", 3.0),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    return result_df

def aggregate_results(result_df: pd.DataFrame) -> pd.DataFrame:
    ag_cols = ['ag_completeness', 'ag_accuracy', 'ag_explainability', 'ag_factual_consistency']
    rag_cols = ['rag_completeness', 'rag_accuracy', 'rag_explainability', 'rag_factual_consistency']

    agg = {}
    for col in ag_cols + rag_cols:
        agg[col] = result_df[col].mean()

    agg['delta_explainability'] = agg['ag_explainability'] - agg['rag_explainability']
    agg['delta_factual'] = agg['ag_factual_consistency'] - agg['rag_factual_consistency']
    agg['hallucination_rag'] = 6.0 - agg['rag_factual_consistency']
    agg['hallucination_ag'] = 6.0 - agg['ag_factual_consistency']
    agg['delta_hallucination'] = agg['hallucination_rag'] - agg['hallucination_ag']
    return pd.DataFrame([agg])

# =============================================================================
# 6. ЗАПУСК
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="CSV с колонками question, reference_answer")
    parser.add_argument("--model", default="llama3.1:8b", help="Модель Ollama для генерации (RAG и АГ)")
    parser.add_argument("--judge", default=None, help="Модель GigaChat для судьи (если None – берётся из Config)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число вопросов")
    parser.add_argument("--output", default="comparison_results.csv")
    args = parser.parse_args()

    # Пути к файлам RAG (убедитесь, что они существуют)
    INDEX_PATH = "faiss_index.bin"
    CHUNKS_PATH = "chunks.pkl"
    EMBED_PATH = "embedding_model"

    print(f"Загрузка RAG с моделью {args.model}...")
    rag = RAGSystem(INDEX_PATH, CHUNKS_PATH, EMBED_PATH, args.model)
    ag = AGSystem()

    print(f"Запуск сравнения (судья: GigaChat, модель {args.judge or 'из Config'})...")
    df = run_comparison(args.dataset, rag, ag, args.judge, args.output, args.limit)
    summary = aggregate_results(df)
    print("\n=== СВОДКА ===")
    print(summary.to_string(index=False))
    print(f"\nРезультаты сохранены в {args.output}")

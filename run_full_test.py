#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование RAG vs АГ-память с использованием Ollama (генерация)
и GigaChat (оценка). Использует класс OllamaClient из agent.py для единообразия.
Автоматически определяет все установленные модели и тестирует их.
Запуск: python run_full_test.py
"""

import os
import sys
import json
import time
import pickle
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

import requests
import faiss
from sentence_transformers import SentenceTransformer

# Импорт GigaChat из parser
try:
    from parser import get_giga_client
except ImportError:
    print("❌ Не удалось импортировать get_giga_client из parser.py. Убедитесь, что parser.py находится в той же папке.")
    sys.exit(1)

# Импорт OllamaClient из agent.py (используем его для генерации)
try:
    from agent import OllamaClient
except ImportError:
    print("❌ Не удалось импортировать OllamaClient из agent.py. Создадим свой клиент.")
    # fallback – свой простой клиент
    class OllamaClient:
        def __init__(self, base_url: str = "http://localhost:11434"):
            self.base_url = base_url

        def generate(self, model: str, prompt: str, temperature: float = 0.3, timeout: int = 60, num_predict: int = 1024) -> str:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict, "repeat_penalty": 1.1}
            }
            try:
                resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except Exception as e:
                print(f"⚠️ Ошибка модели {model}: {e}")
                return ""

# =============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================
def get_available_models() -> List[str]:
    """Возвращает список установленных моделей в Ollama."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        models = []
        for line in lines[1:]:  # пропускаем заголовок
            parts = line.split()
            if parts:
                model_name = parts[0]
                models.append(model_name)
        return models
    except Exception as e:
        print(f"⚠️ Не удалось получить список моделей Ollama: {e}")
        return []

# =============================================================================
# 2. RAG-СИСТЕМА (использует OllamaClient)
# =============================================================================
class RAGSystem:
    def __init__(self, index_path: str, chunks_path: str, embed_model_path: str, model_name: str):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        self.embed_model = SentenceTransformer(embed_model_path)
        self.model_name = model_name
        self.ollama = OllamaClient()

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
        return self.ollama.generate(self.model_name, prompt, temperature=0.6, num_predict=max_new_tokens)

# =============================================================================
# 3. КЛИЕНТ ДЛЯ АГ-СИСТЕМЫ
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
            print(f"⚠️ Ошибка оценки GigaChat: {e}")
            return {"completeness": 3.0, "accuracy": 3.0, "explainability": 3.0, "factual_consistency": 3.0}

# =============================================================================
# 5. ФУНКЦИИ СРАВНЕНИЯ И АГРЕГАЦИИ
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
        print(f"  Вопрос {idx+1}/{len(df)}: {q[:60]}...")

        rag_ans = rag.answer(q)
        ag_ans = ag.answer(q, rag.model_name)
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

# =============================================================================
# 6. ПРОВЕРКА ОКРУЖЕНИЯ (минимальная)
# =============================================================================
def check_environment():
    issues = []
    base_rag = "rag_offline_package"
    if not os.path.exists(base_rag):
        issues.append(f"❌ Папка {base_rag} не найдена.")
    else:
        for f in ["faiss_index.bin", "chunks.pkl"]:
            if not os.path.exists(os.path.join(base_rag, f)):
                issues.append(f"❌ Файл {base_rag}/{f} не найден.")
        if not os.path.exists(os.path.join(base_rag, "embedding_model")):
            issues.append(f"❌ Папка {base_rag}/embedding_model не найдена.")

    if not os.path.exists("questions_dataset.csv"):
        issues.append("❌ Файл questions_dataset.csv не найден. Укажите путь через --dataset.")

    try:
        requests.get("http://localhost:5000/api/stats", timeout=2)
    except:
        issues.append("❌ Сервер АГ-системы (http://localhost:5000) недоступен. Запустите 'python app.py'.")

    try:
        requests.get("http://localhost:11434", timeout=2)
    except:
        issues.append("⚠️ Ollama (http://localhost:11434) не отвечает. Убедитесь, что она запущена.")

    if issues:
        print("="*70)
        print("⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ (но выполнение может продолжиться):")
        for issue in issues:
            print(f"  {issue}")
        print("="*70)
        return False
    return True

# =============================================================================
# 7. ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="questions_dataset.csv", help="Путь к CSV с вопросами и эталонами")
    parser.add_argument("--models", nargs="+", default=None, help="Список моделей Ollama для тестирования (если не указаны, используются все установленные)")
    parser.add_argument("--judge", default=None, help="Модель GigaChat для судьи")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число вопросов")
    parser.add_argument("--output_dir", default=None, help="Папка для сохранения результатов")
    parser.add_argument("--rag_dir", default="rag_offline_package", help="Папка с файлами RAG")
    args = parser.parse_args()

    check_environment()

    if args.output_dir is None:
        output_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Результаты будут сохранены в: {output_dir}")

    rag_base = args.rag_dir
    index_path = os.path.join(rag_base, "faiss_index.bin")
    chunks_path = os.path.join(rag_base, "chunks.pkl")
    embed_path = os.path.join(rag_base, "embedding_model")

    if not (os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(embed_path)):
        print(f"❌ Файлы RAG не найдены в {rag_base}. Укажите правильный путь через --rag_dir.")
        sys.exit(1)

    # Определяем модели для тестирования
    if args.models:
        models = args.models
    else:
        models = get_available_models()
        if not models:
            print("❌ Не найдено ни одной модели Ollama. Установите модели через 'ollama pull' и запустите снова.")
            sys.exit(1)
        print(f"🔍 Найдены модели Ollama: {', '.join(models)}")

    ag = AGSystem()
    all_summaries = []

    # Создаём клиент для проверки (но не используем проверку)
    ollama_client = OllamaClient()

    for model in models:
        print("\n" + "="*70)
        print(f"🚀 ТЕСТИРОВАНИЕ МОДЕЛИ: {model}")
        print("="*70)

        # Пробуем создать RAG с этой моделью (если модель недоступна, будет ошибка)
        try:
            rag = RAGSystem(index_path, chunks_path, embed_path, model)
        except Exception as e:
            print(f"❌ Ошибка загрузки RAG для {model}: {e}")
            continue

        # Проверяем, отвечает ли модель, отправив тестовый запрос через клиент
        try:
            test_resp = ollama_client.generate(model, "test", temperature=0.1, num_predict=1)
            if not test_resp:
                print(f"⚠️ Модель {model} не отвечает (пустой ответ). Пропускаем.")
                continue
        except Exception as e:
            print(f"⚠️ Модель {model} не отвечает: {e}. Пропускаем.")
            continue

        output_file = os.path.join(output_dir, f"results_{model.replace(':', '_')}.csv")
        print(f"▶️ Запуск сравнения (судья: GigaChat)...")
        df = run_comparison(args.dataset, rag, ag, args.judge, output_file, args.limit)

        summary = {
            "model": model,
            "N_questions": len(df),
            "rag_completeness": df["rag_completeness"].mean(),
            "rag_accuracy": df["rag_accuracy"].mean(),
            "rag_explainability": df["rag_explainability"].mean(),
            "rag_factual_consistency": df["rag_factual_consistency"].mean(),
            "ag_completeness": df["ag_completeness"].mean(),
            "ag_accuracy": df["ag_accuracy"].mean(),
            "ag_explainability": df["ag_explainability"].mean(),
            "ag_factual_consistency": df["ag_factual_consistency"].mean(),
        }
        summary["delta_explainability"] = summary["ag_explainability"] - summary["rag_explainability"]
        summary["delta_factual"] = summary["ag_factual_consistency"] - summary["rag_factual_consistency"]
        summary["hallucination_rag"] = 6.0 - summary["rag_factual_consistency"]
        summary["hallucination_ag"] = 6.0 - summary["ag_factual_consistency"]
        summary["delta_hallucination"] = summary["hallucination_rag"] - summary["hallucination_ag"]
        all_summaries.append(summary)
        print(f"✅ Результаты сохранены в {output_file}")

    if not all_summaries:
        print("❌ Нет завершённых тестов. Проверьте модели и окружение.")
        sys.exit(1)

    summary_df = pd.DataFrame(all_summaries).sort_values("model")
    summary_csv = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "="*70)
    print("📊 СВОДНАЯ ТАБЛИЦА ПО ВСЕМ МОДЕЛЯМ")
    print("="*70)
    print(summary_df.to_string(index=False))
    print(f"\n✅ Сводка сохранена в {summary_csv}")

    # Графики
    try:
        import matplotlib.pyplot as plt
        metrics = [
            ("ag_explainability", "Объяснимость (AG)"),
            ("rag_explainability", "Объяснимость (RAG)"),
            ("ag_factual_consistency", "Фактическая согласованность (AG)"),
            ("rag_factual_consistency", "Фактическая согласованность (RAG)"),
            ("delta_explainability", "Δ Объяснимости (AG - RAG)"),
            ("delta_factual", "Δ Факт. согласованности (AG - RAG)"),
        ]
        for col, label in metrics:
            if col in summary_df.columns:
                plt.figure(figsize=(10, 6))
                plt.bar(summary_df["model"], summary_df[col], color='skyblue', edgecolor='navy')
                plt.title(label, fontsize=14)
                plt.xlabel("Модель генерации", fontsize=12)
                plt.ylabel("Оценка (1-5) / Дельта", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_file = os.path.join(output_dir, f"{col}_plot.png")
                plt.savefig(plot_file, dpi=150)
                plt.close()
                print(f"📊 График сохранён: {plot_file}")

        plt.figure(figsize=(10, 6))
        x = np.arange(len(summary_df["model"]))
        width = 0.35
        plt.bar(x - width/2, summary_df["hallucination_rag"], width, label="RAG")
        plt.bar(x + width/2, summary_df["hallucination_ag"], width, label="АГ-память")
        plt.xlabel("Модель генерации")
        plt.ylabel("Индекс галлюцинаций (6 - fact_cons)")
        plt.title("Галлюцинации (чем ниже, тем лучше)")
        plt.xticks(x, summary_df["model"], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hallucination_plot.png"), dpi=150)
        plt.close()
        print("📊 График галлюцинаций сохранён.")
    except ImportError:
        print("⚠️ Matplotlib не установлен. Графики не построены.")
    except Exception as e:
        print(f"⚠️ Ошибка при построении графиков: {e}")

    print("\n" + "="*70)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ.")
    print(f"📁 Все результаты в папке: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()

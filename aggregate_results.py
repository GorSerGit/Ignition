#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_results.py – сбор и агрегация результатов тестирования.
Запуск: python aggregate_results.py --input results_20250315_120000 --output summary.csv
"""

import os
import sys
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def aggregate_results(input_dir: str, output_csv: str = None):
    """
    Собирает все CSV-файлы из input_dir, вычисляет средние метрики для каждой модели
    и сохраняет сводную таблицу.
    """
    # Находим все CSV-файлы с результатами
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"❌ В папке {input_dir} не найдено CSV-файлов.")
        return

    # Собираем сводные данные
    summaries = []
    for file in csv_files:
        # Извлекаем имя модели из имени файла (после results_)
        basename = os.path.basename(file)
        # Ожидаемый формат: results_llama3.1_8b.csv
        model_name = basename.replace("results_", "").replace(".csv", "").replace("_", ":")
        # Если формат другой, можно попробовать извлечь по-другому
        # Альтернативно: читаем первую строку и берём оттуда (но модель не в данных)

        df = pd.read_csv(file)
        # Вычисляем средние по всем вопросам
        summary = {
            "model": model_name,
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
        # Добавляем дельты
        summary["delta_explainability"] = summary["ag_explainability"] - summary["rag_explainability"]
        summary["delta_factual"] = summary["ag_factual_consistency"] - summary["rag_factual_consistency"]
        summary["hallucination_rag"] = 6.0 - summary["rag_factual_consistency"]
        summary["hallucination_ag"] = 6.0 - summary["ag_factual_consistency"]
        summary["delta_hallucination"] = summary["hallucination_rag"] - summary["hallucination_ag"]

        summaries.append(summary)

    # Создаём DataFrame
    summary_df = pd.DataFrame(summaries)
    # Сортируем по размеру модели (по возможности)
    # Можно отсортировать по имени (если они идут в порядке возрастания)
    summary_df = summary_df.sort_values("model")

    # Сохраняем сводку
    if output_csv is None:
        output_csv = os.path.join(input_dir, "summary.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"✅ Сводная таблица сохранена в {output_csv}")

    # Печатаем в консоль
    print("\n=== СВОДНАЯ ТАБЛИЦА ===")
    print(summary_df.to_string(index=False))

    # Построение графиков (если matplotlib установлен)
    try:
        # Метрики для визуализации
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
                plot_file = os.path.join(input_dir, f"{col}_plot.png")
                plt.savefig(plot_file, dpi=150)
                plt.close()
                print(f"📊 График сохранён: {plot_file}")

        # Дополнительно: галлюцинации
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
        plt.savefig(os.path.join(input_dir, "hallucination_plot.png"), dpi=150)
        plt.close()
        print(f"📊 График галлюцинаций сохранён.")

    except ImportError:
        print("⚠️ Matplotlib не установлен. Графики не построены.")
    except Exception as e:
        print(f"⚠️ Ошибка при построении графиков: {e}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Папка с CSV-файлами результатов")
    parser.add_argument("--output", default=None, help="Путь для сохранения сводного CSV")
    args = parser.parse_args()

    aggregate_results(args.input, args.output)
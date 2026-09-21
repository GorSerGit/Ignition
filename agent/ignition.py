"""
Ignition Engine — процесс воспламенения графа.
ИСПРАВЛЕНО: кэширование меток для предотвращения N+1 запросов и ошибок SQLite.
"""
import math
import json
from collections import defaultdict
from database import AHDatabase

class IgnitionEngine:
    def __init__(self, db: AHDatabase):  # ✅ ИСПРАВЛЕНО: двойные подчёркивания
        self.db = db
        self._label_cache = {}
        
        # 🛡 СБАЛАНСИРОВАННЫЕ ГИПЕРПАРАМЕТРЫ
        self.lambda_decay = 0.20       
        self.threshold_t = 0.40        
        self.max_ticks = 5             
        self.spread_factor = 0.50      
        self.hebbian_eta = 0.02        
        self.seed_anchor = 0.70        # 🔥 Якорение seed-узлов

    def _preload_labels(self):
        """Загружает все метки узлов в кэш одним запросом в начале работы."""
        self._label_cache = {}
        try:
            # Символы
            rows = self.db.conn.execute("SELECT uid, r_text FROM symbols").fetchall()
            for uid, r_text_json in rows:
                r_text = json.loads(r_text_json)
                self._label_cache[uid] = r_text[0] if r_text else uid[:8]
            
            # Факты
            fact_rows = self.db.conn.execute("SELECT uid, mt FROM facts").fetchall()
            for uid, mt_json in fact_rows:
                mt = json.loads(mt_json)
                pred_uid = mt.get("predicate_uid")
                if pred_uid and pred_uid in self._label_cache:
                    self._label_cache[uid] = f"F:{self._label_cache[pred_uid]}"
                else:
                    self._label_cache[uid] = f"Fact:{uid[:6]}"
        except Exception as e:
            print(f"⚠ Ошибка предзагрузки меток: {e}")

    def _get_node_label(self, uid: str) -> str:
        if uid in self._label_cache:
            return self._label_cache[uid]
        
        # Fallback если узла нет в кэше
        try:
            row = self.db.conn.execute("SELECT r_text FROM symbols WHERE uid = ?", (uid,)).fetchone()
            if row:
                r_text = json.loads(row[0])
                return r_text[0] if r_text else uid[:8]
        except Exception:
            pass
            
        return f"[{uid[:6]}]"

    def query(self, seed_uids: list[str]) -> dict:
        self._preload_labels()  # ✅ Кэшируем метки
        
        if not seed_uids:
            return {"answer_uids": [], "working_memory": [], "trace_log": ["Нет seed-узлов"], "activations": {}}

        trace_log = []
        activations = defaultdict(float)
        seed_set = set(seed_uids)

        for uid in seed_uids:
            activations[uid] = 1.0
            trace_log.append(f"[Восприятие] Seed: {self._get_node_label(uid)} (x=1.00)")

        all_links = self.db.get_all_links()
        outgoing = defaultdict(list)
        for link in all_links:
            outgoing[link.e1].append((link.e2, link.w, link.link_id))

        working_memory_history = []

        for tick in range(1, self.max_ticks + 1):
            new_activations = defaultdict(float)
            current_tick_log = []

            for src_uid, src_z in activations.items():
                if src_z < 0.05:
                    continue
                for dst_uid, weight, link_type in outgoing[src_uid]:
                    type_modifier = 0.3 if link_type in ["IS-A", "FOLLOW"] else 1.0
                    signal = src_z * weight * self.spread_factor * type_modifier
                    new_activations[dst_uid] += signal

                    if tick <= 2 and signal > 0.05:
                        src_label = self._get_node_label(src_uid)
                        dst_label = self._get_node_label(dst_uid)
                        current_tick_log.append(
                            f"  {src_label} --({link_type}, w={weight:.2f})--> "
                            f"{dst_label} (Δx={signal:.3f})"
                        )

            for uid in set(list(activations.keys()) + list(new_activations.keys())):
                decayed_old = activations.get(uid, 0.0) * (1.0 - self.lambda_decay)
                incoming_signal = new_activations.get(uid, 0.0)
                raw_x = decayed_old + incoming_signal
                final_x = math.tanh(raw_x)

                # 🔥 ЯКОРЕНИЕ SEED-УЗЛОВ
                if uid in seed_set and final_x < self.seed_anchor:
                    final_x = self.seed_anchor

                if final_x > 0.05:
                    new_activations[uid] = final_x

            working_memory = [uid for uid, x in new_activations.items() if x >= self.threshold_t]
            working_memory_history.append(working_memory)

            top_active = sorted(new_activations.items(), key=lambda x: -x[1])[:5]
            top_str = ", ".join([f"{self._get_node_label(uid)}={x:.2f}" for uid, x in top_active])
            trace_log.append(f"[Такт {tick}] Рабочая память ({len(working_memory)} узлов): {top_str}")
            trace_log.extend(current_tick_log[:5])

            activations = new_activations

            if not new_activations:
                trace_log.append(f"[Стоп] Активность затухла на такте {tick}.")
                break

        final_working_memory = working_memory_history[-1] if working_memory_history else []
        trace_log.append(f"[Итог] В рабочей памяти закрепилось {len(final_working_memory)} концептов.")

        return {
            "answer_uids": final_working_memory,
            "working_memory": final_working_memory,
            "activations": dict(activations),
            "trace_log": trace_log,
        }

    def query_with_history(self, seed_uids: list[str]) -> dict:
        self._preload_labels()  # ✅ Кэшируем метки
        
        if not seed_uids:
            return {"history": [], "final_working_memory": [], "seed_uids": [], "trace_log": []}

        all_links = self.db.get_all_links()
        outgoing = defaultdict(list)
        for link in all_links:
            outgoing[link.e1].append((link.e2, link.w, link.link_id))

        activations = defaultdict(float)
        seed_set = set(seed_uids)

        for uid in seed_uids:
            activations[uid] = 1.0

        history = []
        trace_log = []

        for uid in seed_uids:
            trace_log.append(f"[Восприятие] Seed: {self._get_node_label(uid)} (x=1.00)")

        for tick in range(1, self.max_ticks + 1):
            new_activations = defaultdict(float)
            events = []
            tick_log = []

            for src_uid, src_z in activations.items():
                if src_z < 0.05:
                    continue
                for dst_uid, weight, link_type in outgoing[src_uid]:
                    type_modifier = 0.3 if link_type in ["IS-A", "FOLLOW"] else 1.0
                    signal = src_z * weight * self.spread_factor * type_modifier
                    new_activations[dst_uid] += signal
                    events.append({
                        "src": src_uid,
                        "dst": dst_uid,
                        "signal": round(signal, 4),
                        "link_type": link_type,
                        "weight": round(float(weight), 3) if weight else 0.5
                    })

                    if tick <= 2 and signal > 0.03:
                        tick_log.append(
                            f"  {self._get_node_label(src_uid)} "
                            f"--({link_type}, w={weight:.2f})--> "
                            f"{self._get_node_label(dst_uid)} (Δx={signal:.3f})"
                        )

            for uid in set(list(activations.keys()) + list(new_activations.keys())):
                decayed_old = activations.get(uid, 0.0) * (1.0 - self.lambda_decay)
                incoming = new_activations.get(uid, 0.0)
                raw = decayed_old + incoming
                final_x = math.tanh(raw)

                # 🔥 ЯКОРЕНИЕ SEED-УЗЛОВ
                if uid in seed_set and final_x < self.seed_anchor:
                    final_x = self.seed_anchor

                new_activations[uid] = final_x if final_x > 0.05 else 0.0

            working_memory = [uid for uid, x in new_activations.items() if x >= self.threshold_t]

            history.append({
                "tick": tick,
                "activations": {k: round(v, 4) for k, v in new_activations.items() if v > 0.01},
                "working_memory": working_memory,
                "events": events
            })

            top5 = sorted(new_activations.items(), key=lambda x: -x[1])[:5]
            top_str = ", ".join(f"{self._get_node_label(u)}={x:.2f}" for u, x in top5)
            trace_log.append(f"[Такт {tick}] Рабочая память ({len(working_memory)} узлов): {top_str}")
            trace_log.extend(tick_log[:8])

            activations = new_activations

            if not any(x > 0.05 for x in activations.values()):
                trace_log.append(f"[Стоп] Активность затухла на такте {tick}.")
                break

        final_wm = history[-1]["working_memory"] if history else []
        trace_log.append(f"[Итог] В рабочей памяти закрепилось {len(final_wm)} концептов.")

        return {
            "history": history,
            "final_working_memory": final_wm,
            "seed_uids": seed_uids,
            "trace_log": trace_log
        }

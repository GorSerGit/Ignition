"""
SQLite хранилище для АГ-памяти.
Финальная версия: включает индексы, транзакции и все методы для Ignition Engine.
"""
import sqlite3
import json
import time
from models import AbstractSymbol, ConceptSymbol, Hypernode, Link

class AHDatabase:
    def __init__(self, db_path: str):
        # Разрешаем использование соединения в разных потоках (для веб-сервера)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                uid TEXT PRIMARY KEY,
                canonical TEXT UNIQUE,
                r_text TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS concepts (
                uid TEXT PRIMARY KEY,
                pr TEXT NOT NULL DEFAULT '{}',
                mt TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS facts (
                uid TEXT PRIMARY KEY,
                w REAL NOT NULL DEFAULT 1.0,
                t_star TEXT,
                roles TEXT NOT NULL DEFAULT '{}',
                pr TEXT NOT NULL DEFAULT '{}',
                mt TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS episodes (
                uid TEXT PRIMARY KEY,
                w REAL NOT NULL DEFAULT 1.0,
                t_star TEXT,
                roles TEXT NOT NULL DEFAULT '{}',
                pr TEXT NOT NULL DEFAULT '{}',
                mt TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS links (
                uid TEXT PRIMARY KEY,
                link_id TEXT NOT NULL,
                w REAL NOT NULL DEFAULT 0.5,
                e1 TEXT NOT NULL,
                e2 TEXT NOT NULL,
                e1_type TEXT NOT NULL DEFAULT '',
                e2_type TEXT NOT NULL DEFAULT '',
                count INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS parsed_chunks (
                chunk_hash TEXT PRIMARY KEY,
                timestamp REAL
            );
            
            CREATE INDEX IF NOT EXISTS idx_symbols_canonical ON symbols(canonical);
            CREATE INDEX IF NOT EXISTS idx_links_e1 ON links(e1);
            CREATE INDEX IF NOT EXISTS idx_links_e2 ON links(e2);
            CREATE INDEX IF NOT EXISTS idx_links_type ON links(link_id);
        """)
        
        # Авто-миграция для старых БД (добавление canonical, если его нет)
        c.execute("PRAGMA table_info(symbols)")
        columns = [col[1] for col in c.fetchall()]
        if 'canonical' not in columns:
            c.execute("ALTER TABLE symbols ADD COLUMN canonical TEXT")
            rows = c.execute("SELECT uid, r_text FROM symbols").fetchall()
            for row in rows:
                try:
                    r_text = json.loads(row["r_text"])
                    if r_text:
                        c.execute("UPDATE symbols SET canonical = ? WHERE uid = ?", (r_text[0], row["uid"]))
                except Exception:
                    pass
                    
        # Авто-миграция для links (добавление count, если его нет)
        c.execute("PRAGMA table_info(links)")
        columns = [col[1] for col in c.fetchall()]
        if 'count' not in columns:
            c.execute("ALTER TABLE links ADD COLUMN count INTEGER NOT NULL DEFAULT 1")
            c.execute("UPDATE links SET w = 0.5 WHERE count = 1")
            
        self.conn.commit()

    # ==========================================
    # УПРАВЛЕНИЕ ТРАНЗАКЦИЯМИ (Защита от сбоев)
    # ==========================================
    def begin_transaction(self):
        self.conn.execute("BEGIN TRANSACTION")

    def commit_transaction(self):
        self.conn.execute("COMMIT")

    def rollback_transaction(self):
        try:
            self.conn.execute("ROLLBACK")
        except Exception:
            pass

    # ==========================================
    # CRASH RECOVERY (Хэши chunks)
    # ==========================================
    def is_chunk_processed(self, chunk_hash: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM parsed_chunks WHERE chunk_hash = ?", (chunk_hash,)).fetchone()
        return row is not None

    def mark_chunk_processed(self, chunk_hash: str):
        self.conn.execute("INSERT OR IGNORE INTO parsed_chunks (chunk_hash, timestamp) VALUES (?, ?)", 
                          (chunk_hash, time.time()))
        self.conn.commit()

    # ==========================================
    # S: SYMBOLS (Абстрактные символы)
    # ==========================================
    def save_symbol(self, s: AbstractSymbol):
        canonical = s.canonical if hasattr(s, 'canonical') and s.canonical else (s.r_text[0] if s.r_text else "")
        self.conn.execute(
            "INSERT OR REPLACE INTO symbols (uid, canonical, r_text) VALUES (?, ?, ?)",
            (s.uid, canonical, json.dumps(s.r_text, ensure_ascii=False))
        )
        self.conn.commit()

    def get_symbol_by_canonical(self, canonical: str) -> AbstractSymbol | None:
        """Быстрый поиск по индексу (используется в Ignition Agent)."""
        row = self.conn.execute("SELECT uid, canonical, r_text FROM symbols WHERE canonical = ?", (canonical,)).fetchone()
        if row:
            s = AbstractSymbol(uid=row["uid"], r_text=json.loads(row["r_text"]))
            s.canonical = row["canonical"]
            return s
        return None

    def get_symbol_by_word(self, word: str) -> AbstractSymbol | None:
        """Поиск по любому слову из парадигмы (используется в test_db.py)."""
        word_lower = word.lower().strip()
        # Сначала пробуем быстрый поиск по canonical
        sym = self.get_symbol_by_canonical(word_lower)
        if sym:
            return sym
        # Если не нашли, ищем по всем словоформам (медленнее)
        rows = self.conn.execute("SELECT uid, canonical, r_text FROM symbols").fetchall()
        for row in rows:
            r_text = json.loads(row["r_text"])
            if word_lower in [w.lower() for w in r_text]:
                s = AbstractSymbol(uid=row["uid"], r_text=r_text)
                s.canonical = row["canonical"]
                return s
        return None

    def get_all_symbols(self) -> list[AbstractSymbol]:
        rows = self.conn.execute("SELECT uid, canonical, r_text FROM symbols").fetchall()
        res = []
        for r in rows:
            s = AbstractSymbol(uid=r["uid"], r_text=json.loads(r["r_text"]))
            s.canonical = r["canonical"]
            res.append(s)
        return res

    # ==========================================
    # P: FACTS & H: EPISODES
    # ==========================================
    def save_fact(self, n: Hypernode):
        self.conn.execute(
            "INSERT OR REPLACE INTO facts (uid, w, t_star, roles, pr, mt) VALUES (?, ?, ?, ?, ?, ?)",
            (n.uid, n.w, n.t_star, json.dumps(n.roles, ensure_ascii=False), 
             json.dumps(n.pr, ensure_ascii=False), json.dumps(n.mt, ensure_ascii=False))
        )
        self.conn.commit()

    def save_episode(self, n: Hypernode):
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes (uid, w, t_star, roles, pr, mt) VALUES (?, ?, ?, ?, ?, ?)",
            (n.uid, n.w, n.t_star, json.dumps(n.roles, ensure_ascii=False), 
             json.dumps(n.pr, ensure_ascii=False), json.dumps(n.mt, ensure_ascii=False))
        )
        self.conn.commit()

    # ==========================================
    # L: LINKS (Связи и гиперсвязи)
    # ==========================================
    def save_link(self, l: Link):
        self.conn.execute(
            "INSERT OR REPLACE INTO links (uid, link_id, w, e1, e2, e1_type, e2_type, count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (l.uid, l.link_id, l.w, l.e1, l.e2, l.e1_type, l.e2_type, l.count)
        )
        self.conn.commit()

    def get_links_from(self, node_uid: str) -> list[Link]:
        rows = self.conn.execute("SELECT * FROM links WHERE e1 = ?", (node_uid,)).fetchall()
        return [Link.from_dict(dict(r)) for r in rows]

    def get_links_to(self, node_uid: str) -> list[Link]:
        """Критически важно для сбора фактов в рабочей памяти."""
        rows = self.conn.execute("SELECT * FROM links WHERE e2 = ?", (node_uid,)).fetchall()
        return [Link.from_dict(dict(r)) for r in rows]

    def get_all_links(self) -> list[Link]:
        """Критически важно для Ignition Engine (загрузка графа в память)."""
        rows = self.conn.execute("SELECT * FROM links").fetchall()
        return [Link.from_dict(dict(r)) for r in rows]

    def get_link_by_endpoints(self, link_id: str, e1: str, e2: str) -> Link | None:
        row = self.conn.execute(
            "SELECT * FROM links WHERE link_id = ? AND e1 = ? AND e2 = ? LIMIT 1", 
            (link_id, e1, e2)
        ).fetchone()
        return Link.from_dict(dict(row)) if row else None

    def merge_duplicate_links(self):
        c = self.conn.cursor()
        c.execute("""
            SELECT link_id, e1, e2, SUM(count) as total_count
            FROM links GROUP BY link_id, e1, e2 HAVING COUNT(*) > 1
        """)
        for link_id, e1, e2, total_count in c.fetchall():
            if link_id in ["IS-A", "FOLLOW", "PREDICATE"]:
                new_w = 1.0
            else:
                new_w = total_count / (total_count + 1.0)
                
            c.execute("UPDATE links SET count = ?, w = ? WHERE link_id = ? AND e1 = ? AND e2 = ?", 
                      (total_count, new_w, link_id, e1, e2))
            c.execute("""DELETE FROM links WHERE link_id = ? AND e1 = ? AND e2 = ? 
                         AND rowid NOT IN (SELECT rowid FROM links WHERE link_id = ? AND e1 = ? AND e2 = ? LIMIT 1)""", 
                      (link_id, e1, e2, link_id, e1, e2))
        self.conn.commit()

    # ==========================================
    # СТАТИСТИКА И СЛУЖЕБНОЕ
    # ==========================================
    def get_stats(self) -> dict:
        return {
            "S": self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
            "C": self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
            "P": self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
            "H": self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "L": self.conn.execute("SELECT COUNT(*) FROM links").fetchone()[0],
        }

    def close(self):
        self.conn.close()

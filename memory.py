import pymorphy3
from models import AbstractSymbol, Hypernode, Link, gen_uid
from database import AHDatabase

morph = pymorphy3.MorphAnalyzer()

# Структурные связи монографии Душкина (не требуют статистического взвешивания)
STRUCTURAL_LINKS = {"IS-A", "FOLLOW", "PREDICATE"}

class AHMemory:
    def __init__(self, db: AHDatabase):
        self.db = db
        self.last_fact_uid = None  # Для построения цепочек FOLLOW (эпизодическая память)

    def _resolve_symbol(self, word: str) -> AbstractSymbol:
        """
        Приводит слово к канонической форме и ищет/создает узел.
        Не использует словари, только pymorphy3 и БД.
        """
        if not isinstance(word, str) or not word.strip():
            return None
            
        word = word.strip().lower()
        parsed = morph.parse(word)
        
        # 1. Определяем каноническую форму (лемму)
        if parsed:
            canonical = parsed[0].normal_form
        else:
            canonical = word

        # 2. Ищем существующий узел по канонической форме
        existing = self.db.get_symbol_by_canonical(canonical)
        if existing:
            # Если узел есть, но встретилась новая словоформа — добавляем её в множество R (парадигму)
            if word not in [w.lower() for w in existing.r_text]:
                existing.r_text.append(word)
                self.db.save_symbol(existing)
            return existing

        # 3. Создаем новый Абстрактный Символ (S)
        # Формируем множество первичных символов R (текстовая парадигма)
        r_text = [canonical]
        if parsed:
            # Берем все словоформы из парадигмы, но ограничим до 15, чтобы не раздувать БД
            lexeme_forms = list(set([p.word.lower() for p in parsed[0].lexeme]))[:15]
            for f in lexeme_forms:
                if f not in r_text:
                    r_text.append(f)
                    
        s = AbstractSymbol(uid=gen_uid("S"), r_text=r_text)
        s.canonical = canonical  # Сохраняем для быстрого поиска в БД
        self.db.save_symbol(s)
        return s

    def _add_or_update_link(self, link_id: str, e1: str, e2: str, e1_type: str, e2_type: str) -> Link:
        existing = self.db.get_link_by_endpoints(link_id, e1, e2)
        if existing:
            existing.count += 1
            # 📊 Статистический вес для семантических связей (ролей)
            if link_id in STRUCTURAL_LINKS:
                existing.w = 1.0
            else:
                existing.w = existing.count / (existing.count + 1.0)
            self.db.save_link(existing)
            return existing
        else:
            w = 1.0 if link_id in STRUCTURAL_LINKS else 0.5
            link = Link(uid=gen_uid("L"), link_id=link_id, w=w, 
                        e1=e1, e2=e2, e1_type=e1_type, e2_type=e2_type, count=1)
            self.db.save_link(link)
            return link

    def add_fact(self, predicate_uid: str, roles: dict, source_text: str = "") -> Hypernode:
        n = Hypernode(uid=gen_uid("N"), w=1.0, t_star=None, roles=roles, 
                      pr={"text_span": source_text}, mt={"predicate_uid": predicate_uid})
        self.db.save_fact(n)
        
        self._add_or_update_link("PREDICATE", n.uid, predicate_uid, "P", "S")
        for role_name, symbol_uid in roles.items():
            self._add_or_update_link(role_name, n.uid, symbol_uid, "P", "S")
            
        # 🧬 Эпизодическая память: связываем факты цепочкой FOLLOW
        if self.last_fact_uid and self.last_fact_uid != n.uid:
            self.add_follow_link(self.last_fact_uid, n.uid)
        self.last_fact_uid = n.uid
        
        return n

    def add_isa_link(self, child_uid: str, parent_uid: str):
        if self._would_create_cycle(child_uid, parent_uid, "IS-A"): return
        self._add_or_update_link("IS-A", child_uid, parent_uid, "S", "S")

    def add_follow_link(self, earlier_uid: str, later_uid: str):
        if self._would_create_cycle(earlier_uid, later_uid, "FOLLOW"): return
        self._add_or_update_link("FOLLOW", earlier_uid, later_uid, "P", "P")

    def ingest_parsed_fact(self, fact: dict) -> Hypernode | None:
        predicate_word = fact.get("predicate", "")
        roles_data = fact.get("roles", {})
        source_text = fact.get("_source_text", "")
        
        if not predicate_word or not roles_data: return None
        
        pred_symbol = self._resolve_symbol(predicate_word)
        if not pred_symbol: return None
        
        role_uids = {}
        valid_roles = {"SUBJECT", "OBJECT", "LOCATION", "TIME", "CAUSE", "INSTRUMENT", "MATERIAL", "EFFECT"}
        
        for role_name, role_value in roles_data.items():
            if role_name not in valid_roles: continue
            if isinstance(role_value, list): role_value = role_value[0] if role_value else ""
            if not role_value: continue
            
            sym = self._resolve_symbol(role_value)
            if sym: role_uids[role_name] = sym.uid
            
        # Обработка шаблона CLASSIFY -> Иерархия IS-A
        template = fact.get("template", "")
        if template == "CLASSIFY" and "SUBJECT" in role_uids and "OBJECT" in role_uids:
            self.add_isa_link(role_uids["SUBJECT"], role_uids["OBJECT"])
            
        return self.add_fact(pred_symbol.uid, role_uids, source_text)

    def _would_create_cycle(self, from_uid: str, to_uid: str, link_type: str) -> bool:
        if from_uid == to_uid: return True
        visited = set()
        queue = [to_uid]
        while queue:
            current = queue.pop(0)
            if current == from_uid: return True
            if current in visited: continue
            visited.add(current)
            for l in self.db.get_links_from(current):
                if l.link_id == link_type: queue.append(l.e2)
        return False

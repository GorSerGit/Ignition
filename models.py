"""
Структуры данных АГ-памяти строго по Разделу 3 Монографии.
Кортеж AH = ⟨S, C, P, H, L⟩
"""
from dataclasses import dataclass, field
from typing import Optional
import json
import uuid


def gen_uid(prefix: str = "") -> str:
    """Генерация уникального UID."""
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{short}" if prefix else short


# ─── Множество S: Абстрактные символы 1-го порядка ───
@dataclass
class AbstractSymbol:
    uid: str
    r_text: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"uid": self.uid, "r_text": self.r_text}

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractSymbol":
        return cls(uid=d["uid"], r_text=d.get("r_text", []))


# ─── Множество C: Абстрактные символы 2-го порядка (Концепты) ───
@dataclass
class ConceptSymbol:
    uid: str
    pr: dict = field(default_factory=dict)
    mt: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"uid": self.uid, "pr": self.pr, "mt": self.mt}

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptSymbol":
        return cls(uid=d["uid"], pr=d.get("pr", {}), mt=d.get("mt", {}))


# ─── Шаблон модели управления ───
@dataclass
class Template:
    uid: str
    predicate: str
    roles: list[str]

    def to_dict(self) -> dict:
        return {"uid": self.uid, "predicate": self.predicate, "roles": self.roles}


# ─── Гиперсвязь (факт/эпизод) ───
@dataclass
class Hypernode:
    uid: str
    w: float = 1.0
    t_star: Optional[str] = None
    roles: dict[str, str] = field(default_factory=dict)
    pr: dict = field(default_factory=dict)
    mt: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid, "w": self.w, "t_star": self.t_star,
            "roles": self.roles, "pr": self.pr, "mt": self.mt
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypernode":
        return cls(
            uid=d["uid"], w=d.get("w", 1.0), t_star=d.get("t_star"),
            roles=d.get("roles", {}), pr=d.get("pr", {}), mt=d.get("mt", {})
        )


# ─── Ассоциативная связь ───
@dataclass
class Link:
    uid: str
    link_id: str
    w: float = 0.9
    e1: str = ""
    e2: str = ""
    e1_type: str = ""
    e2_type: str = ""
    count: int = 1          # новое поле – количество встреч

    def to_dict(self) -> dict:
        return {
            "uid": self.uid, "link_id": self.link_id, "w": self.w,
            "e1": self.e1, "e2": self.e2,
            "e1_type": self.e1_type, "e2_type": self.e2_type,
            "count": self.count
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Link":
        return cls(
            uid=d["uid"], link_id=d["link_id"], w=d.get("w", 0.9),
            e1=d["e1"], e2=d["e2"],
            e1_type=d.get("e1_type", ""), e2_type=d.get("e2_type", ""),
            count=d.get("count", 1)
        )

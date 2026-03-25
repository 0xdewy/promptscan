import sqlite3
from typing import List, Dict, Any


class PromptDatabase:
    def __init__(self, db_path: str = "prompts.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            is_injection BOOLEAN NOT NULL
        )
        """)

        conn.commit()
        conn.close()

    def add_prompt(self, text: str, is_injection: bool):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO prompts (text, is_injection)
        VALUES (?, ?)
        """,
            (text, is_injection),
        )

        conn.commit()
        conn.close()

    def add_prompts_batch(self, prompts: List[Dict[str, Any]]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for prompt in prompts:
            cursor.execute(
                """
            INSERT INTO prompts (text, is_injection)
            VALUES (?, ?)
            """,
                (prompt["text"], prompt["is_injection"]),
            )

        conn.commit()
        conn.close()

    def get_all_prompts(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, text, is_injection FROM prompts")
        rows = cursor.fetchall()

        prompts = []
        for row in rows:
            prompts.append({"id": row[0], "text": row[1], "is_injection": bool(row[2])})

        conn.close()
        return prompts

    def get_prompt_by_id(self, prompt_id: int) -> Dict[str, Any] | None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, text, is_injection FROM prompts WHERE id = ?", (prompt_id,)
        )
        row = cursor.fetchone()

        conn.close()

        if row:
            return {"id": row[0], "text": row[1], "is_injection": bool(row[2])}
        return None

    def update_prompt(self, prompt_id: int, text: str, is_injection: bool):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        UPDATE prompts 
        SET text = ?, is_injection = ?
        WHERE id = ?
        """,
            (text, is_injection, prompt_id),
        )

        conn.commit()
        conn.close()

    def delete_prompt(self, prompt_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))

        conn.commit()
        conn.close()

    def search_prompts(self, query: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT id, text, is_injection 
        FROM prompts 
        WHERE text LIKE ?
        """,
            (f"%{query}%",),
        )

        rows = cursor.fetchall()

        prompts = []
        for row in rows:
            prompts.append({"id": row[0], "text": row[1], "is_injection": bool(row[2])})

        conn.close()
        return prompts

    def get_statistics(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM prompts")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prompts WHERE is_injection = 1")
        injections = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prompts WHERE is_injection = 0")
        safe = cursor.fetchone()[0]

        conn.close()

        return {
            "total": total,
            "injections": injections,
            "safe": safe,
            "injection_percentage": (injections / total * 100) if total > 0 else 0,
            "safe_percentage": (safe / total * 100) if total > 0 else 0,
        }

    def clear_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM prompts")

        conn.commit()
        conn.close()

    def export_to_json(self) -> List[Dict[str, Any]]:
        return self.get_all_prompts()

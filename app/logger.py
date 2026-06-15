import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "predictions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            input_data       TEXT NOT NULL,
            income_class     TEXT NOT NULL,
            probability      REAL NOT NULL,
            confidence       TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(input_data: dict, result: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions VALUES (NULL,?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(),
            json.dumps(input_data),
            result["income_class"],
            result["probability_high_income"],
            result["confidence"],
        )
    )
    conn.commit()
    conn.close()


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    total      = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    high_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE income_class='>50K'").fetchone()[0]
    conn.close()
    low_count = total - high_count
    return {
        "total_predictions": total,
        "high_income_count": high_count,
        "low_income_count":  low_count,
        "high_income_pct":   round(high_count / total * 100, 1) if total > 0 else 0.0,
    }

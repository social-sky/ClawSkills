import sqlite3
import os
from pathlib import Path

db_path = Path.home() / ".openclaw" / "lcm.db"
db_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Creating database at: {db_path}")

conn = sqlite3.connect(str(db_path))
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys=ON")
conn.execute("PRAGMA busy_timeout=5000")
conn.commit()

cursor = conn.execute("PRAGMA journal_mode")
print(f"Journal mode: {cursor.fetchone()[0]}")

cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(f"Tables: {[r[0] for r in cursor.fetchall()]}")

conn.close()
print(f"Database created: {db_path.exists()}, size: {db_path.stat().st_size}")

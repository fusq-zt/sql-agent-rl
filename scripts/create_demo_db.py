"""Create a small teaching database; never overwrite an existing database."""
import argparse
from pathlib import Path
import sqlite3

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, default=Path("runs/demo.sqlite"))
args = p.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("xb"):
    pass
try:
    with sqlite3.connect(args.output) as connection:
        connection.executescript((Path(__file__).resolve().parents[1] / "examples/demo.sql").read_text(encoding="utf-8"))
finally:
    if "connection" in locals():
        connection.close()
print(args.output)

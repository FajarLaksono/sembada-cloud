import json, pathlib

# 1. Create .env file
env_path = pathlib.Path(".env")
if not env_path.exists():
    env_path.write_text(
        "# Sembada Cloud - paths relative to project root\nDATA_DIR=data/transformed/parquet\n",
        encoding="utf-8",
    )
    print("[OK] Created .env")
else:
    print("[--] .env already exists")

# 2. The dotenv + path resolution block
DOTENV_BLOCK = """from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
from pathlib import Path
cwd = Path.cwd()
PROJECT_ROOT = cwd.parent if cwd.name == 'notebooks' else cwd
DATA_DIR = Path(os.getenv('DATA_DIR', 'data/transformed/parquet'))
DATA_DIR = (PROJECT_ROOT / DATA_DIR).resolve() if not DATA_DIR.is_absolute() else DATA_DIR
sys.path.insert(0, str(PROJECT_ROOT))"""

OLD_SYSPATH = "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent))"

# 3. Update notebooks
for name in [
    "03a_feature_engineering.ipynb",
    "03b_tabular_models.ipynb",
    "03c_timeseries_forecasting.ipynb",
]:
    path = pathlib.Path("notebooks") / name
    nb = json.loads(path.read_text(encoding="utf-8"))
    changes = 0

    for cell in nb["cells"]:
        src = "".join(cell["source"])

        # a) Replace sys.path line with dotenv block
        if OLD_SYSPATH in src:
            cell["source"] = [
                line.replace(OLD_SYSPATH, DOTENV_BLOCK) for line in cell["source"]
            ]
            changes += 1
            print(f"  [{name}] Replaced sys.path line")

        # b) Remove hardcoded DATA_DIR = Path(...) line
        if "DATA_DIR = pathlib.Path" in src or "DATA_DIR = Path(" in src:
            new_src = []
            for line in cell["source"]:
                stripped = line.strip()
                if stripped.startswith("DATA_DIR = ") and (
                    "pathlib.Path" in stripped or "Path(" in stripped
                ):
                    new_src.append("# DATA_DIR now from .env (see first cell)\n")
                    changes += 1
                    print(f"  [{name}] Commented out DATA_DIR assignment")
                else:
                    new_src.append(line)
            cell["source"] = new_src

    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if changes == 0:
        print(f"  [{name}] No changes needed")

print("\nDone.")

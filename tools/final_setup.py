import json, pathlib, subprocess

# Restore originals from git
subprocess.run(
    [
        "git",
        "restore",
        "--",
        "notebooks/03a_feature_engineering.ipynb",
        "notebooks/03b_tabular_models.ipynb",
        "notebooks/03c_timeseries_forecasting.ipynb",
    ],
    check=True,
)
print("[OK] Restored from git")

# Ensure .env exists
env_path = pathlib.Path(".env")
if not env_path.exists():
    env_path.write_text("DATA_DIR=data/transformed/parquet\n", encoding="utf-8")
    print("[OK] Created .env")

# Prepare the lines to insert (each with trailing newline)
DOTENV_LINES = [
    "from dotenv import load_dotenv, find_dotenv\n",
    "load_dotenv(find_dotenv())\n",
    "import os\n",
    "from pathlib import Path\n",
    "cwd = Path.cwd()\n",
    "PROJECT_ROOT = cwd.parent if cwd.name == 'notebooks' else cwd\n",
    "DATA_DIR = Path(os.getenv('DATA_DIR', 'data/transformed/parquet'))\n",
    "if not DATA_DIR.is_absolute():\n",
    "    DATA_DIR = (PROJECT_ROOT / DATA_DIR).resolve()\n",
    "sys.path.insert(0, str(PROJECT_ROOT))\n",
]

for name in [
    "03a_feature_engineering.ipynb",
    "03b_tabular_models.ipynb",
    "03c_timeseries_forecasting.ipynb",
]:
    path = pathlib.Path("notebooks") / name
    nb = json.loads(path.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src_lines = cell["source"]
        src = "".join(src_lines)

        # 1) Insert DOTENV_BLOCK lines after filterwarnings line
        if "warnings.filterwarnings" in src and "PROJECT_ROOT" not in src:
            for j, line in enumerate(src_lines):
                if "warnings.filterwarnings" in line:
                    for k, dotenv_line in enumerate(DOTENV_LINES):
                        src_lines.insert(j + 1 + k, dotenv_line)
                    print(f"  [{name}] Inserted DOTENV_BLOCK after filterwarnings")
                    break

        # 2) Replace hardcoded DATA_DIR line with comment
        if "DATA_DIR = pathlib.Path" in src and "data/transformed/parquet" in src:
            for j, line in enumerate(src_lines):
                stripped = line.strip()
                if stripped.startswith("DATA_DIR = pathlib.Path") and "data/transformed/parquet" in stripped:
                    src_lines[j] = "# DATA_DIR now from .env (see first cell)\n"
                    print(f"  [{name}] Commented old DATA_DIR assignment")

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

print()
print("Done.")

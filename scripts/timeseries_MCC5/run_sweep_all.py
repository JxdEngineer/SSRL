import re
import subprocess
from pathlib import Path

# ================= USER SETTINGS =================
N = 3  # number of runs per variant

variant_configs = {
    "model_o":  dict(lam_time=100, lam_self=1, lam_psd=100),
    "model_v1": dict(lam_time=100, lam_self=0, lam_psd=0),
    "model_v2": dict(lam_time=100, lam_self=1, lam_psd=0),
    "model_v3": dict(lam_time=100, lam_self=0, lam_psd=100),
}

# ================= PATHS =================
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

CONFIG_PATH = REPO_ROOT / "scripts" / "timeseries_MCC5" / "configs.py"

TRAIN_PY = THIS_DIR / "train.py"
TEST_PY  = THIS_DIR / "test_v2.py"

LOG_DIR = THIS_DIR / "sweep_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ================= HELPERS =================
def replace_assign(text, var, value):
    pattern = rf'^{var}\s*=\s*["\'][^"\']*["\']\s*$'
    repl = f'{var} = "{value}"'
    text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"Failed replacing {var}, replaced {n} lines")
    return text


def replace_multiple(text, mapping):
    for var, value in mapping.items():
        pattern = rf'^{var}\s*=\s*[-+]?\d*\.?\d+\s*$'
        repl = f'{var} = {value}'
        text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"Failed replacing {var}, replaced {n} lines")
    return text


def run_cmd(cmd, log_path, cwd):
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


# ================= CORE =================
def run_one(variant, i):
    cfg = variant_configs[variant]

    ckpt_name = f"{variant}_{i}.pt"
    curve_name = f"{variant.replace('model','loss_curves')}_{i}.png"

    print(f"\n===== {variant} run {i}/{N} =====")
    print("config:", cfg)

    original = CONFIG_PATH.read_text(encoding="utf-8")

    try:
        updated = original

        # update filenames
        updated = replace_assign(updated, "ckpt_name", ckpt_name)
        updated = replace_assign(updated, "curve_name", curve_name)

        # update loss weights
        updated = replace_multiple(updated, cfg)

        CONFIG_PATH.write_text(updated, encoding="utf-8")

        # TRAIN
        # train_log = LOG_DIR / f"train_{variant}_{i}.log"
        # run_cmd(["python", "-u", str(TRAIN_PY)], train_log, TRAIN_PY.parent)
        # print("train done:", train_log)

        # TEST
        test_log = LOG_DIR / f"test_{variant}_{i}.log"
        run_cmd(["python", "-u", str(TEST_PY)], test_log, TEST_PY.parent)
        print("test done:", test_log)

    finally:
        CONFIG_PATH.write_text(original, encoding="utf-8")


def main():
    for variant in variant_configs:
        for i in range(1, N + 1):
            run_one(variant, i)

    print("\n✅ ALL EXPERIMENTS FINISHED")
    print("Logs saved in:", LOG_DIR)


if __name__ == "__main__":
    main()

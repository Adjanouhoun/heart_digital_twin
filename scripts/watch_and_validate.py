"""
Surveille l'entraînement nnU-Net et lance automatiquement
la validation ACDC dès que Dice MYO >= 0.90.

Usage :
    python scripts/watch_and_validate.py
"""
import re
import time
import subprocess
from pathlib import Path

LOG_FILE = Path.home() / "nnunet/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_2026_6_23_04_36_44.txt"
SLO_MYO  = 0.90
SLO_LV   = 0.90
SLO_RV   = 0.85

RE_DICE  = re.compile(r"Pseudo dice \[([^\]]+)\]")
RE_EPOCH = re.compile(r"Epoch (\d+)")

def get_last_dice(log_path):
    content = log_path.read_text()
    epochs = []
    current = {}
    for line in content.splitlines():
        if m := RE_EPOCH.search(line):
            if current:
                epochs.append(current.copy())
            current = {"epoch": int(m.group(1))}
        elif m := RE_DICE.search(line):
            vals = [float(x.strip().replace("np.float32(","").replace(")",""))
                    for x in m.group(1).split(",")]
            if len(vals) >= 3:
                current["rv"]  = vals[0]
                current["myo"] = vals[1]
                current["lv"]  = vals[2]
    if current and "myo" in current:
        epochs.append(current)
    return epochs[-1] if epochs else None

def run_validation():
    print("\n🚀 Lancement de la validation ACDC...")
    result = subprocess.run([
        "python", "scripts/validate_acdc.py",
        "--acdc_dir", str(Path.home() / "Downloads/ACDC/database/training"),
        "--n_patients", "10"
    ], capture_output=True, text=True)
    print(result.stdout[-2000:])
    if result.returncode == 0:
        print("✅ Validation ACDC terminée")
    else:
        print(f"❌ Erreur : {result.stderr[-500:]}")

print("👀 Surveillance nnU-Net — attente SLO MYO ≥ 0.90")
print(f"   Log : {LOG_FILE}\n")

while True:
    last = get_last_dice(LOG_FILE)
    if last:
        epoch = last.get("epoch", "?")
        myo   = last.get("myo", 0)
        lv    = last.get("lv", 0)
        rv    = last.get("rv", 0)

        slo_myo = "✅" if myo >= SLO_MYO else f"❌ ({SLO_MYO - myo:.3f} manquant)"
        slo_lv  = "✅" if lv  >= SLO_LV  else "❌"
        slo_rv  = "✅" if rv  >= SLO_RV  else "❌"

        print(f"Epoch {epoch} | LV={lv:.4f}{slo_lv} | MYO={myo:.4f}{slo_myo} | RV={rv:.4f}{slo_rv}")

        if myo >= SLO_MYO and lv >= SLO_LV and rv >= SLO_RV:
            print(f"\n🎉 TOUS LES SLOs ATTEINTS à l'epoch {epoch} !")
            print(f"   LV={lv:.4f} ✅  MYO={myo:.4f} ✅  RV={rv:.4f} ✅")
            run_validation()
            break

    time.sleep(60)  # Vérifier chaque minute

"""
Tracker MLflow pour l'entraînement nnU-Net ACDC.
Lit le log nnU-Net en temps réel et enregistre dans MLflow.

Usage :
    python scripts/mlflow_nnunet_tracker.py
    
    # En arrière-plan
    nohup python scripts/mlflow_nnunet_tracker.py > ~/nnunet/mlflow_tracker.log 2>&1 &
"""
import re
import time
from pathlib import Path
import mlflow

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_URI   = "http://localhost:5001"
EXPERIMENT   = "nnunet-cardiac-segmentation"
LOG_FILE     = Path.home() / "nnunet/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_2026_6_23_04_36_44.txt"
POLL_INTERVAL_S = 30  # Vérifier toutes les 30 secondes

# ── Patterns de parsing ───────────────────────────────────────────────────────
RE_EPOCH      = re.compile(r"Epoch (\d+)")
RE_TRAIN_LOSS = re.compile(r"train_loss ([-\d.]+)")
RE_VAL_LOSS   = re.compile(r"val_loss ([-\d.]+)")
RE_DICE       = re.compile(r"Pseudo dice \[([^\]]+)\]")
RE_EMA_DICE   = re.compile(r"New best EMA pseudo Dice: ([\d.]+)")
RE_EPOCH_TIME = re.compile(r"Epoch time: ([\d.]+) s")
RE_LR         = re.compile(r"Current learning rate: ([\d.]+)")


def parse_log(log_path: Path) -> list[dict]:
    """Parse le log nnU-Net et retourne une liste d'epochs."""
    if not log_path.exists():
        return []

    content = log_path.read_text()
    epochs = []
    current = {}

    for line in content.splitlines():
        if m := RE_EPOCH.search(line):
            if current and "epoch" in current:
                epochs.append(current.copy())
            current = {"epoch": int(m.group(1))}

        elif m := RE_LR.search(line):
            current["lr"] = float(m.group(1))

        elif m := RE_TRAIN_LOSS.search(line):
            current["train_loss"] = float(m.group(1))

        elif m := RE_VAL_LOSS.search(line):
            current["val_loss"] = float(m.group(1))

        elif m := RE_DICE.search(line):
            dice_vals = [float(x.strip().replace("np.float32(","").replace(")",""))
                        for x in m.group(1).split(",")]
            if len(dice_vals) >= 3:
                current["dice_rv"]  = dice_vals[0]
                current["dice_myo"] = dice_vals[1]
                current["dice_lv"]  = dice_vals[2]

        elif m := RE_EMA_DICE.search(line):
            current["ema_dice"] = float(m.group(1))

        elif m := RE_EPOCH_TIME.search(line):
            current["epoch_time_s"] = float(m.group(1))

    # Ajouter la dernière epoch si complète
    if current and "train_loss" in current:
        epochs.append(current)

    return epochs


def log_to_mlflow(epochs: list[dict], run_id: str | None = None) -> str:
    """Enregistre les epochs dans MLflow. Retourne le run_id."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    if run_id:
        # Reprendre un run existant
        with mlflow.start_run(run_id=run_id):
            for epoch_data in epochs:
                step = epoch_data.get("epoch", 0)
                metrics = {k: v for k, v in epoch_data.items() if k != "epoch"}
                mlflow.log_metrics(metrics, step=step)
        return run_id
    else:
        # Nouveau run
        with mlflow.start_run(run_name="ACDC_2d_fold0_cpu") as run:
            # Paramètres fixes du run
            mlflow.log_params({
                "dataset": "ACDC",
                "n_training_cases": 160,
                "n_validation_cases": 40,
                "configuration": "2d",
                "fold": 0,
                "patch_size": "256x224",
                "batch_size": 56,
                "device": "cpu",
                "machine": "Apple M1 Pro 32GB",
                "ionic_model": "nnUNetTrainer",
                "max_epochs": 1000,
                "initial_lr": 0.01,
                "slo_dice_myo": 0.90,
                "slo_dice_lv": 0.90,
                "slo_dice_rv": 0.85,
            })

            # Métriques par epoch
            for epoch_data in epochs:
                step = epoch_data.get("epoch", 0)
                metrics = {k: v for k, v in epoch_data.items() if k != "epoch"}
                mlflow.log_metrics(metrics, step=step)

                # Afficher la progression
                dice_myo = epoch_data.get("dice_myo", 0)
                dice_lv  = epoch_data.get("dice_lv", 0)
                slo_myo  = "✅" if dice_myo >= 0.90 else "❌"
                slo_lv   = "✅" if dice_lv  >= 0.90 else "❌"
                print(f"  Epoch {step:3d} | "
                      f"train={epoch_data.get('train_loss', 0):.4f} | "
                      f"val={epoch_data.get('val_loss', 0):.4f} | "
                      f"LV={dice_lv:.3f}{slo_lv} | "
                      f"MYO={dice_myo:.3f}{slo_myo} | "
                      f"RV={epoch_data.get('dice_rv', 0):.3f}")

            return run.info.run_id


def watch_and_log():
    """Surveille le log en temps réel et met à jour MLflow."""
    print(f"🔍 Surveillance du log : {LOG_FILE}")
    print(f"📊 MLflow : {MLFLOW_URI}/#{EXPERIMENT}")
    print(f"⏱  Polling toutes les {POLL_INTERVAL_S}s\n")

    # Chercher un run existant pour éviter les doublons
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    existing_runs = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string="run_name = 'ACDC_2d_fold0_cpu'", order_by=["start_time DESC"])
    run_id = existing_runs.iloc[0]["run_id"] if len(existing_runs) > 0 else None
    last_epoch_logged = -1
    if run_id:
        print(f"♻️  Reprise du run existant : {run_id[:8]}...")

    while True:
        epochs = parse_log(LOG_FILE)
        new_epochs = [e for e in epochs if e.get("epoch", -1) > last_epoch_logged]

        if new_epochs:
            print(f"📈 {len(new_epochs)} nouvelle(s) epoch(s) détectée(s) :")
            run_id = log_to_mlflow(new_epochs, run_id)
            last_epoch_logged = max(e.get("epoch", 0) for e in new_epochs)
            print(f"✅ Enregistré dans MLflow (run_id={run_id[:8]}...)\n")

            # Vérifier SLOs
            last = new_epochs[-1]
            dice_myo = last.get("dice_myo", 0)
            if dice_myo >= 0.90:
                print(f"🎉 SLO Dice MYO ≥ 0.90 ATTEINT ! ({dice_myo:.4f})")
                print("→ Lancer : python scripts/validate_acdc.py --n_patients 10")

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    # Enregistrer d'abord les epochs déjà calculées
    print("📥 Import des epochs existantes...")
    epochs = parse_log(LOG_FILE)

    if epochs:
        print(f"   {len(epochs)} epoch(s) trouvée(s) dans le log")
        run_id = log_to_mlflow(epochs)
        print(f"\n✅ Epochs existantes enregistrées (run_id={run_id[:8]}...)")
        print(f"🌐 Voir : {MLFLOW_URI}")
    else:
        print("   Aucune epoch complète trouvée")
        run_id = None

    print("\n🔄 Passage en mode surveillance...")
    watch_and_log()

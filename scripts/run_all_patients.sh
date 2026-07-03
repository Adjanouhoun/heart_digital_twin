#!/bin/bash
export PYTHONPATH=/Users/amadouadjanouhoun/cdt
echo "=== openCARP Batch — 9 patients (tend=50ms) ==="
echo "Debut: $(date)"

for p in patient002 patient003 patient004 patient005 patient006 patient007 patient008 patient009 patient010; do
    echo ""
    echo "--- $p ($(date)) ---"
    python scripts/run_opencarp_patient.py --patient $p --tend 50 --mpi 4
    echo "--- $p termine ($(date)) ---"
done

echo ""
echo "=== Batch termine: $(date) ==="

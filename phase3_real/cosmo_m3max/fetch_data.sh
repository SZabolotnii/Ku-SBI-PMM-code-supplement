#!/usr/bin/env bash
# COSMO gate -- fetch the three data artefacts (LSST Y10 lognormal shifts + the
# pre-trained VMIM ResNet18 compressor) into ./data/.
#
# Sources, in order of preference:
#   1) the FSM authors' public repo  github.com/Shermjj/Direct_FSM   (data/, if present)
#   2) the sbi_lens repo             github.com/DifferentiableUniverseInitiative/sbi_lens
# The three filenames are searched anywhere inside the clones, so minor layout
# changes upstream do not break the script.  Nothing is committed to git -- the
# files are third-party artefacts (see README_UA.md §3 for the license note).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

FILES=(lognormal_shifts_LSSTY10_om_s8_w_bin.npy opt_state_resnet_vmim.pkl params_nd_compressor_vmim.pkl)

need_any=false
for f in "${FILES[@]}"; do [[ -f "data/$f" ]] || need_any=true; done
if ! $need_any; then echo "[ok] all three files already in ./data"; exit 0; fi

try_repo () {
  local url=$1 name=$2
  echo "== trying $name =="
  if git clone --depth 1 "$url" "$TMP/$name" 2>/dev/null; then
    for f in "${FILES[@]}"; do
      if [[ ! -f "data/$f" ]]; then
        found=$(find "$TMP/$name" -name "$f" -type f | head -1 || true)
        if [[ -n "$found" ]]; then cp "$found" "data/$f"; echo "  [got] $f  <- $name"; fi
      fi
    done
  else
    echo "  [skip] clone failed: $url"
  fi
}

try_repo https://github.com/Shermjj/Direct_FSM Direct_FSM
try_repo https://github.com/DifferentiableUniverseInitiative/sbi_lens sbi_lens

echo "== result =="
missing=0
for f in "${FILES[@]}"; do
  if [[ -f "data/$f" ]]; then
    echo "  [ok] $f  sha256=$(shasum -a 256 "data/$f" | cut -c1-16)…"
  else
    echo "  [MISSING] $f"
    missing=1
  fi
done
if [[ $missing -eq 1 ]]; then
  echo "Some files were not found automatically. Locate them manually (git-lfs may be"
  echo "required: 'brew install git-lfs && git lfs install' and re-run), then place"
  echo "them in ./data/ under the exact names above."
  exit 1
fi
echo "All data present. Next: python check_env.py"

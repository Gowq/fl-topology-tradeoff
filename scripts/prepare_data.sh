#!/usr/bin/env bash
# Download the public datasets the experiments read from ./data.
#   - OPPORTUNITY Activity Recognition (UCI #226): downloaded + unzipped here.
#   - MHEALTH (UCI #319): downloaded, checksum-verified, and unzipped here.
#   - CIFAR-10/100: fetched automatically by torchvision at first run, but we
#     pre-fetch CIFAR-10 here so offline/Docker runs work.
# Datasets are NOT committed to this repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

OPP_ZIP="$DATA/OpportunityUCIDataset.zip"
OPP_DIR="$DATA/OpportunityUCIDataset"
OPP_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"

if [ ! -d "$OPP_DIR" ]; then
    if [ ! -f "$OPP_ZIP" ]; then
        echo ">>> Downloading OPPORTUNITY (UCI #226) ..."
        curl -fL --retry 3 -o "$OPP_ZIP" "$OPP_URL" \
            || { echo "ERROR: download failed. Get OpportunityUCIDataset.zip from"; \
                 echo "       https://archive.ics.uci.edu/dataset/226/ and place it in $DATA"; exit 1; }
    fi
    echo ">>> Extracting OPPORTUNITY ..."
    python3 -c "import zipfile,sys; zipfile.ZipFile('$OPP_ZIP').extractall('$DATA')"
fi
echo ">>> OPPORTUNITY ready at $OPP_DIR"

MHEALTH_ZIP="$DATA/mhealth_dataset.zip"
MHEALTH_DIR="$DATA/MHEALTHDATASET"
MHEALTH_URL="https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
MHEALTH_SHA256="16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0"

if [ ! -d "$MHEALTH_DIR" ]; then
    if [ ! -f "$MHEALTH_ZIP" ]; then
        echo ">>> Downloading MHEALTH (UCI #319) ..."
        curl -fL --retry 3 -o "$MHEALTH_ZIP" "$MHEALTH_URL" \
            || { echo "ERROR: MHEALTH download failed: $MHEALTH_URL"; exit 1; }
    fi
    echo "$MHEALTH_SHA256  $MHEALTH_ZIP" | sha256sum --check --status \
        || { echo "ERROR: MHEALTH checksum mismatch; remove $MHEALTH_ZIP and retry"; exit 1; }
    echo ">>> Extracting MHEALTH ..."
    python3 -c "import zipfile; zipfile.ZipFile('$MHEALTH_ZIP').extractall('$DATA')"
fi
echo ">>> MHEALTH ready at $MHEALTH_DIR"

echo ">>> Pre-fetching CIFAR-10 via torchvision ..."
python3 - "$DATA" <<'PY'
import sys, torchvision
torchvision.datasets.CIFAR10(root=sys.argv[1], train=True,  download=True)
torchvision.datasets.CIFAR10(root=sys.argv[1], train=False, download=True)
print("CIFAR-10 ready")
PY

echo ">>> Data preparation complete."

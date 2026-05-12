#!/bin/bash
# Downloads and extracts CIFAR-10 binary version into data/cifar-10-batches-bin/
# After running this script, pass "data/cifar-10-batches-bin" as the data_dir.

set -e

cd "$(dirname "$0")/.."

DATA_DIR="data"
DEST="$DATA_DIR/cifar-10-batches-bin"

if [ -d "$DEST" ]; then
    echo "CIFAR-10 already present at $DEST"
    exit 0
fi

mkdir -p "$DATA_DIR"

URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
ARCHIVE="$DATA_DIR/cifar-10-binary.tar.gz"

echo "Downloading CIFAR-10 binary (~163 MB)..."
if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$ARCHIVE" "$URL"
else
    curl -L --progress-bar -o "$ARCHIVE" "$URL"
fi

echo "Extracting..."
tar -xzf "$ARCHIVE" -C "$DATA_DIR"
rm "$ARCHIVE"

echo "Done. Dataset at: $DEST"
echo ""
echo "Single-GPU training (~3 min, hidden_dim=4096, 20 epochs):"
echo "  ./build/train_single train $DEST cifar10"
echo ""
echo "Distributed training (4 GPUs):"
echo "  mpirun -n 4 ./build/train_dist cifar10 4096"

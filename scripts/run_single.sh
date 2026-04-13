#!/bin/bash

set -e

cd "$(dirname "$0")/.."

./build/train_single train data
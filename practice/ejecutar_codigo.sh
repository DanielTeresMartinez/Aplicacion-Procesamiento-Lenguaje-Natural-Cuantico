#!/bin/bash
# Script para ejecutar toda la parte practica del TFG:
#   Word2Vec y Q-Word2Vec sobre el corpus v1 y v2.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================================"
echo "  Ejecucion de la parte practica del TFG"
echo "  Word2Vec + Q-Word2Vec (corpus v1 y v2)"
echo "========================================================"
echo ""

echo "[1/4] Word2Vec - Small Corpora v1..."
( cd "$SCRIPT_DIR/small_corpora" && python word2vec.py )
echo ""

echo "[2/4] Q-Word2Vec - Small Corpora v1..."
( cd "$SCRIPT_DIR/small_corpora" && python qword2vec.py )
echo ""

echo "[3/4] Word2Vec - Small Corpora v2..."
( cd "$SCRIPT_DIR/small_corpora_v2" && python word2vec_v2.py )
echo ""

echo "[4/4] Q-Word2Vec - Small Corpora v2..."
( cd "$SCRIPT_DIR/small_corpora_v2" && python qword2vec.py )
echo ""

echo "========================================================"
echo "  Practica completada."
echo "========================================================"

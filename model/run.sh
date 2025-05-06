#!/bin/bash

set -e

echo "Running train_scikit.py with objective: price"
python3 train_scikit.py --objective price

echo "Running train_scikit.py with objective: price_psm"
python3 train_scikit.py --objective price_psm

echo "Running train_scikit.py with objective: lognorm_price"
python3 train_scikit.py --objective lognorm_price

echo "Running train_scikit.py with objective: lognorm_price_psm"
python3 train_scikit.py --objective lognorm_price_psm 

echo "Running train_images.py"
python3 train_images.py

echo "Running train_ensemble.py"
python3 train_ensemble.py


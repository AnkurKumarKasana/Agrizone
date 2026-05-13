"""
Master Training Script for Agrizone ML Models
===============================================
Run this script to train ALL models at once.

Usage:
    python train_all_models.py
    python train_all_models.py --crop-only
    python train_all_models.py --pest-only
    python train_all_models.py --disease-only
"""

import sys
import os
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def train_crop():
    print("\n" + "=" * 60)
    print("STEP 1: Training Crop Recommendation Models")
    print("=" * 60)
    from train_crop_model import train_and_evaluate
    start = time.time()
    train_and_evaluate()
    print(f"Crop model training completed in {time.time() - start:.1f}s")


def train_disease():
    print("\n" + "=" * 60)
    print("STEP 2: Setting Up Disease Detection Model")
    print("=" * 60)
    from train_disease_model import build_disease_model
    start = time.time()
    build_disease_model()
    print(f"Disease model setup completed in {time.time() - start:.1f}s")


def train_pest():
    print("\n" + "=" * 60)
    print("STEP 3: Training Pest Alert Models")
    print("=" * 60)
    from train_pest_model import train_pest_models
    start = time.time()
    train_pest_models()
    print(f"Pest model training completed in {time.time() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Train Agrizone ML Models')
    parser.add_argument('--crop-only', action='store_true', help='Train only crop model')
    parser.add_argument('--disease-only', action='store_true', help='Train only disease model')
    parser.add_argument('--pest-only', action='store_true', help='Train only pest model')
    args = parser.parse_args()

    total_start = time.time()

    print("=" * 60)
    print("AGRIZONE - Complete ML Model Training Pipeline")
    print("=" * 60)

    if args.crop_only:
        train_crop()
    elif args.disease_only:
        train_disease()
    elif args.pest_only:
        train_pest()
    else:
        train_crop()
        train_pest()
        train_disease()

    print("\n" + "=" * 60)
    print(f"ALL TRAINING COMPLETE! Total time: {time.time() - total_start:.1f}s")
    print("=" * 60)
    print(f"\nModel artifacts saved to: {os.path.join(SCRIPT_DIR, 'saved_models')}")
    print("\nNext steps:")
    print("  1. Run: python manage.py runserver")
    print("  2. Test crop recommendation at /recommend/")
    print("  3. Test disease detection at /disease/")
    print("  4. Test pest alerts at /alerts/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Quick check script to verify all files are working"""

import os
import sys
import json

print("=" * 70)
print("🔍 COMPREHENSIVE FILE CHECK")
print("=" * 70)
print()

# Check Python files
print("1. Checking Python Files:")
print("-" * 70)
files_to_check = [
    'model.py',
    'config.py', 
    'train_enhanced.py',
    'dataset.py',
    'utils.py',
    'better_eval.py',
    'app.py'
]

passed = 0
failed = 0

for file in files_to_check:
    if os.path.exists(file):
        try:
            # Try to compile the file
            with open(file, 'r', encoding='utf-8') as f:
                compile(f.read(), file, 'exec')
            print(f"  ✅ {file}")
            passed += 1
        except SyntaxError as e:
            print(f"  ❌ {file}: {e}")
            failed += 1
    else:
        print(f"  ⚠️  {file}: Not found")
        failed += 1

print()
print(f"Result: {passed}/{len(files_to_check)} passed")
print()

# Check config values
print("2. Checking Configuration:")
print("-" * 70)
try:
    from config import Config
    print(f"  ✅ DROPOUT_RATE:    {Config.DROPOUT_RATE}")
    print(f"  ✅ L1_LAMBDA:       {Config.L1_LAMBDA}")
    print(f"  ✅ WEIGHT_DECAY:    {Config.WEIGHT_DECAY}")
    print(f"  ✅ IMG_SIZE:        {Config.IMG_SIZE}")
    print(f"  ✅ BATCH_SIZE:      {Config.BATCH_SIZE}")
    print(f"  ✅ LEARNING_RATE:   {Config.LEARNING_RATE}")
    print(f"  ✅ NUM_EPOCHS:      {Config.NUM_EPOCHS}")
    print(f"  ✅ LABEL_SMOOTHING: {Config.USE_LABEL_SMOOTHING}")
except Exception as e:
    print(f"  ❌ Error loading config: {e}")

print()

# Check Kaggle notebook
print("3. Checking Kaggle Notebook:")
print("-" * 70)
try:
    with open('Train_Enhanced_Kaggle_Fixed.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    total_cells = len(nb['cells'])
    code_cells = len([c for c in nb['cells'] if c['cell_type'] == 'code'])
    markdown_cells = total_cells - code_cells
    
    print(f"  ✅ Total cells: {total_cells}")
    print(f"  ✅ Code cells: {code_cells}")
    print(f"  ✅ Markdown cells: {markdown_cells}")
    
    # Find training config
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'batch_size=' in source and 'train_enhanced_model' in source:
                lines = source.split('\n')
                for line in lines:
                    if 'batch_size=' in line and '#' not in line.split('batch_size=')[0]:
                        print(f"  ✅ {line.strip()}")
                    if 'epochs=' in line and 'NUM_EPOCHS' not in line:
                        print(f"  ✅ {line.strip()}")
                    if 'learning_rate=' in line:
                        print(f"  ✅ {line.strip()}")
                break
    
    print(f"  ✅ Kaggle notebook is valid!")
except Exception as e:
    print(f"  ❌ Error checking notebook: {e}")

print()

# Check model files
print("4. Checking Model Files:")
print("-" * 70)
model_files = [
    'models/best_model.pth',
    'models/enhanced_u2net.pth',
    'models/enhanced_u2net_with_pre_trained.pth',
    'models/enhanced_u2net_with_allupdated.pth'
]

for model_file in model_files:
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"  ✅ {model_file} ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  {model_file}: Not found")

print()
print("=" * 70)
print("✅ FILE CHECK COMPLETE!")
print("=" * 70)

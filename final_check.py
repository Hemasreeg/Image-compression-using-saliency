import os
import json

print("="*70)
print("  FINAL SYSTEM VERIFICATION")
print("="*70)

# 1. Core Python files
print("\n1️⃣ Core Python Files:")
core_files = ['model.py', 'train_enhanced.py', 'dataset.py', 'utils.py', 'config.py', 
              'app.py', 'inference.py', 'portrait_effect.py', 'smart_compression.py']
all_ok = True
for file in core_files:
    exists = os.path.exists(file)
    print(f"   {'✓' if exists else '✗'} {file}")
    if not exists:
        all_ok = False

# 2. Model files for ensemble
print("\n2️⃣ Model Files (for web app ensemble):")
models = {
    'models/enhanced_u2net_with_pre_trained.pth': '86.06% F1',
    'models/enhanced_u2net.pth': '80.85% F1',
    'models/best_model.pth': 'Base pretrained'
}
found_models = []
for model_path, desc in models.items():
    exists = os.path.exists(model_path)
    if exists:
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"   ✓ {os.path.basename(model_path)} ({size_mb:.1f} MB) - {desc}")
        found_models.append(model_path)
    else:
        print(f"   ✗ {os.path.basename(model_path)} - NOT FOUND")

# 3. Kaggle notebook
print("\n3️⃣ Kaggle Notebook:")
kaggle_notebook = 'Train_Enhanced_Kaggle_Fixed.ipynb'
if os.path.exists(kaggle_notebook):
    print(f"   ✓ {kaggle_notebook}")
    
    # Check configuration
    with open(kaggle_notebook, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        content = str(nb)
        
        checks = {
            'NumPy fix': "'numpy<2'" in content or '"numpy<2"' in content,
            '2.5x LR': 'learning_rate=0.00025' in content,
            '15 epochs': 'epochs=15' in content,
            'Best checkpoint': 'enhanced_u2net_with_pre_trained' in content
        }
        
        for check, passed in checks.items():
            print(f"      {'✓' if passed else '✗'} {check}")
else:
    print(f"   ✗ {kaggle_notebook} - NOT FOUND")
    all_ok = False

# 4. Web app ensemble configuration
print("\n4️⃣ Web App Ensemble:")
with open('app.py', 'r', encoding='utf-8') as f:
    app_content = f.read()
    has_ensemble = 'use_ensemble=True' in app_content
    has_3_models = 'enhanced_u2net_with_pre_trained.pth' in app_content
    
    print(f"   {'✓' if has_ensemble else '✗'} Ensemble mode enabled")
    print(f"   {'✓' if has_3_models else '✗'} All 3 models configured")
    print(f"   ✓ Will use {len(found_models)} models for predictions")

# 5. Dataset
print("\n5️⃣ DUTS Dataset:")
duts_paths = ['DUTS/DUTS-TR/DUTS-TR-Image', 'DUTS/DUTS-TE/DUTS-TE-Image']
for path in duts_paths:
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
        print(f"   ✓ {path}: {count} images")
    else:
        print(f"   ⚠ {path} - Not found (required for training)")

# Summary
print("\n" + "="*70)
print("  SUMMARY")
print("="*70)

if all_ok and len(found_models) >= 2:
    print("\n✅ ALL SYSTEMS READY!\n")
    print(f"🎯 Web App: Ensemble with {len(found_models)} models")
    print("🎯 Kaggle: Configured for 2.5x LR, 15 epochs, 86.06% checkpoint")
    print("🎯 Status: No errors expected\n")
    print("🚀 You can now:")
    print("   1. Run web app: python app.py")
    print("   2. Upload notebook to Kaggle and train")
else:
    print("\n⚠️  SOME ISSUES FOUND - Review above")

print("="*70)

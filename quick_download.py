"""Quick download script for U2-Net model"""
import urllib.request
import os

print("Downloading U2-Net model (170 MB)...")
print("This may take 2-5 minutes depending on your connection.\n")

url = "https://huggingface.co/spaces/akhaliq/U-2-Net/resolve/main/saved_models/u2net/u2net.pth"
output = "models/best_model.pth"

os.makedirs("models", exist_ok=True)

def report(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    print(f"\rProgress: {percent:.1f}% ({downloaded/1_000_000:.1f}/{total_size/1_000_000:.1f} MB)", end='')

try:
    urllib.request.urlretrieve(url, output, reporthook=report)
    print("\n\n✅ Download complete!")
    print(f"✅ Model saved to: {os.path.abspath(output)}")
    print(f"✅ File size: {os.path.getsize(output)/1_000_000:.1f} MB")
    print("\n🎉 Your model is ready! Run: python app.py")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTrying alternative URL...")
    
    # Fallback URL
    url2 = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    print(f"Downloading from: {url2}")
    try:
        urllib.request.urlretrieve(url2, output, reporthook=report)
        print("\n\n✅ Download complete!")
    except:
        print("\n\n⚠️ Automatic download failed.")
        print("\nManual steps:")
        print("1. Visit: https://github.com/xuebinqin/U-2-Net/tree/master/saved_models/u2net")
        print("2. Download u2net.pth")
        print("3. Rename to best_model.pth")
        print(f"4. Place in: {os.path.abspath('models/')}")

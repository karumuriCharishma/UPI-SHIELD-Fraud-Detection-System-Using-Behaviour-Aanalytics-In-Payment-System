"""
Run this first: python check_setup.py
It will tell you exactly what is installed and what to fix.
"""
import sys
print(f"Python: {sys.version}")

# Check opencv
try:
    import cv2
    print(f"\n[OK] opencv version: {cv2.__version__}")
    if hasattr(cv2, 'wechat_qrcode_WeChatQRCode'):
        print("[OK] WeChatQRCode is available (opencv-contrib installed correctly)")
        try:
            det = cv2.wechat_qrcode.WeChatQRCode()
            print("[OK] WeChatQRCode initialised successfully")
        except Exception as e:
            print(f"[WARN] WeChatQRCode init failed: {e}")
    else:
        print("[FAIL] WeChatQRCode NOT found — you have plain opencv, not contrib")
        print("       Fix: pip uninstall opencv-python-headless -y")
        print("            pip install opencv-contrib-python-headless")
except ImportError:
    print("[FAIL] opencv not installed at all")

# Check pyzbar
try:
    from pyzbar.pyzbar import decode
    print("\n[OK] pyzbar is available")
except Exception as e:
    print(f"\n[WARN] pyzbar not available: {e} (optional, not critical)")

# Quick decode test
try:
    import sys, os
    print("\nTesting QR code decoding with OpenCV's WeChatQRCode...")
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        import cv2, numpy as np
        img = cv2.imread(sys.argv[1])
        det = cv2.wechat_qrcode.WeChatQRCode()
        data, _ = det.detectAndDecode(img)
        print(f"\nDecode test on '{sys.argv[1]}': {data}")
except Exception as e:
    print(f"\nDecode test error: {e}")

print("\nDone. Share this output if you still have issues.")
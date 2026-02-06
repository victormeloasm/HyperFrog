import os
import sys

def verify_headers():
    # Magic bytes defined in HyperFrog v33.1
    EXPECTED_MAGIC = {
        "test.pk":  b'HFPK',
        "test.sk":  b'HFSK',
        "test.enc": b'HFCT'
    }
    
    print(f"{'FILE':<12} | {'MAGIC':<8} | {'STATUS'}")
    print("-" * 35)
    
    all_ok = True
    for filename, magic in EXPECTED_MAGIC.items():
        if not os.path.exists(filename):
            print(f"{filename:<12} | {'MISSING':<8} |  NOT FOUND")
            all_ok = False
            continue
            
        with open(filename, 'rb') as f:
            header = f.read(4)
            
        if header == magic:
            print(f"{filename:<12} | {header.decode():<8} |  OK")
        else:
            print(f"{filename:<12} | {str(header):<8} |  FAIL (Expected {magic})")
            all_ok = False
            
    return all_ok

def verify_roundtrip():
    print("\n[Integrity Check]")
    if not (os.path.exists("test.bin") and os.path.exists("test.dec")):
        print(" Missing test.bin or test.dec")
        return False
        
    with open("test.bin", "rb") as f1, open("test.dec", "rb") as f2:
        original = f1.read()
        decrypted = f2.read()
        
    if original == decrypted:
        print(f" SHA256 Match: {len(original)} bytes recovered successfully.")
        print("   HyperFrog v33.1 is functioning correctly.")
        return True
    else:
        print(" MISMATCH: Decrypted file differs from original.")
        return False

if __name__ == "__main__":
    print("HyperFrog v33.1 - Output Verification\n")
    headers_ok = verify_headers()
    integrity_ok = verify_roundtrip()
    
    if headers_ok and integrity_ok:
        print("\nResult: PASSED")
        sys.exit(0)
    else:
        print("\nResult: FAILED")
        sys.exit(1)
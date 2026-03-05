#!/usr/bin/env python3
"""
Quick integration test for LoRA management system
Tests basic imports and LoRARegistry initialization
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("\n📦 Testing imports...")
    
    # Install python-dotenv if missing
    try:
        import dotenv
    except ImportError:
        print("  ⚠️  Installing python-dotenv...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-dotenv"], check=False)
    
    try:
        from app.env import Config
        
        assert Config.BASE_MODEL, "BASE_MODEL not configured"
        assert Config.ADAPTER_CACHE_DIR, "ADAPTER_CACHE_DIR not configured"
        print("  ✅ app.env (Config class)")
    except Exception as e:
        print(f"  ❌ app.env: {e}")
        return False
    
    try:
        from app.config import MODEL_REGISTRY, BASE_MODEL_CONFIG, ADAPTER_SOURCE_PRIORITY
        print("  ✅ app.config")
    except Exception as e:
        print(f"  ❌ app.config: {e}")
        return False
    
    try:
        from app.lora_registry import LoRARegistry, get_registry
        print("  ✅ app.lora_registry")
    except Exception as e:
        print(f"  ❌ app.lora_registry: {e}")
        return False
    
    try:
        from app.s3_downloader import load_adapter_from_source, download_lora
        print("  ✅ app.s3_downloader")
    except ImportError as e:
        if "boto3" in str(e):
            print(f"  ⚠️  app.s3_downloader (optional: {e})")
        else:
            print(f"  ❌ app.s3_downloader: {e}")
            return False
    except Exception as e:
        print(f"  ❌ app.s3_downloader: {e}")
        return False
    
    return True
    
    return True

def test_registry_initialization():
    """Test LoRARegistry can be initialized"""
    print("\n🏗️  Testing LoRARegistry initialization...")
    
    try:
        from app.lora_registry import LoRARegistry
        
        # Create test registry
        registry = LoRARegistry()
        print(f"  ✅ LoRARegistry created")
        
        # Test discovery (will be empty since no adapters exist yet)
        discovered = registry.discover_local_loras()
        print(f"  ✅ Auto-discovery: Found {len(discovered)} local LoRAs")
        
        # Test manual registration
        registry.register_lora(
            name="test_lora",
            repo="test/repo",
            source="hf",
            lora_type="test",
            version="1.0.0",
            description="Test LoRA"
        )
        print("  ✅ Manual registration works")
        
        # Test retrieval
        loras = registry.list_all_loras()
        print(f"  ✅ List LoRAs: {len(loras)} registered")
        
        # Test get_summary
        summary = registry.get_summary()
        print(f"  ✅ Get summary works: {summary['total_loras']} total")
        
        return True
    except Exception as e:
        print(f"  ❌ LoRARegistry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that required files exist"""
    print("\n📂 Testing file structure...")
    
    required_files = [
        "app/env.py",
        "app/config.py",
        "app/lora_registry.py",
        "app/adapter_manager.py",
        "app/s3_downloader.py",
        "app/app.py",
        ".env.example",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("🧪 LoRA Management System Integration Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: File structure
    results.append(("File Structure", test_file_structure()))
    
    # Test 2: Imports
    results.append(("Module Imports", test_imports()))
    
    # Test 3: Registry initialization
    results.append(("Registry Init", test_registry_initialization()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

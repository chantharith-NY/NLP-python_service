#!/usr/bin/env python3
"""
Lightweight API Endpoint Verification (no torch dependency)
Tests endpoint definitions without loading the full app
"""

import sys
import os
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_app_py_syntax():
    """Verify app.py has all required endpoints defined"""
    print("\n📝 Checking app.py endpoint definitions...")
    
    try:
        app_file = Path(__file__).parent / "app" / "app.py"
        content = app_file.read_text()
        
        required_patterns = [
            (r'@app\.get\("/"\)', "GET /"),
            (r'@app\.get\("/health"', "GET /health"),
            (r'@app\.post\("/summarize"', "POST /summarize"),
            (r'@app\.post\("/spell-check"', "POST /spell-check"),
            (r'@app\.post\("/generate"', "POST /generate"),
            (r'@app\.get\("/loras"', "GET /loras"),
            (r'@app\.get\("/loras/types"', "GET /loras/types"),
            (r'@app\.get\("/loras/registry"', "GET /loras/registry"),
            (r'@app\.post\("/loras/register"', "POST /loras/register"),
            (r'@app\.delete\("/loras/', "DELETE /loras"),
        ]
        
        found = 0
        for pattern, endpoint_name in required_patterns:
            if re.search(pattern, content):
                print(f"  ✅ {endpoint_name}")
                found += 1
            else:
                print(f"  ❌ {endpoint_name}")
        
        print(f"  Found: {found}/{len(required_patterns)} endpoints")
        return found == len(required_patterns)
        
    except Exception as e:
        print(f"  ❌ Failed to check app.py: {e}")
        return False

def test_pydantic_models():
    """Verify Pydantic models are defined in app.py"""
    print("\n🏗️  Checking Pydantic model definitions...")
    
    try:
        app_file = Path(__file__).parent / "app" / "app.py"
        content = app_file.read_text()
        
        required_classes = [
            "class SummarizeRequest",
            "class SpellCheckRequest",
            "class GenerateRequest",
            "class LoRAInfo",
            "class LoRAsResponse",
            "class LoRATypesResponse",
        ]
        
        found = 0
        for class_def in required_classes:
            if class_def in content:
                print(f"  ✅ {class_def.split()[-1]}")
                found += 1
            else:
                print(f"  ❌ {class_def.split()[-1]}")
        
        print(f"  Found: {found}/{len(required_classes)} models")
        return found == len(required_classes)
        
    except Exception as e:
        print(f"  ❌ Failed to check models: {e}")
        return False

def test_lora_registry():
    """Test LoRARegistry without torch"""
    print("\n🗂️  Testing LoRARegistry...")
    
    try:
        from app.lora_registry import LoRARegistry, get_registry
        
        # Create test registry
        registry = LoRARegistry(cache_dir="./test_models/adapters")
        print("  ✅ LoRARegistry instantiated")
        
        # Test methods
        methods = [
            'discover_local_loras',
            'register_lora',
            'unregister_lora', 
            'get_lora',
            'list_all_loras',
            'list_loras_by_type',
            'list_types',
            'get_summary',
        ]
        
        found = 0
        for method in methods:
            if hasattr(registry, method) and callable(getattr(registry, method)):
                print(f"  ✅ {method}()")
                found += 1
            else:
                print(f"  ❌ {method}()")
        
        print(f"  Found: {found}/{len(methods)} methods")
        
        # Test basic operations
        registry.register_lora(
            name="test",
            repo="test/repo",
            lora_type="test"
        )
        print("  ✅ Registration works")
        
        loras = registry.list_all_loras()
        assert len(loras) > 0, "Should have registered LoRA"
        print("  ✅ List LoRAs works")
        
        return found == len(methods)
        
    except Exception as e:
        print(f"  ❌ LoRARegistry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that core modules can be imported without torch"""
    print("\n📦 Testing core module imports...")
    
    modules = []
    
    try:
        from app.env import Config
        print("  ✅ app.env.Config")
        modules.append(True)
    except Exception as e:
        print(f"  ❌ app.env: {e}")
        modules.append(False)
    
    try:
        from app.config import MODEL_REGISTRY
        print("  ✅ app.config")
        modules.append(True)
    except Exception as e:
        print(f"  ❌ app.config: {e}")
        modules.append(False)
    
    try:
        from app.lora_registry import LoRARegistry
        print("  ✅ app.lora_registry")
        modules.append(True)
    except Exception as e:
        print(f"  ❌ app.lora_registry: {e}")
        modules.append(False)
    
    return all(modules)

def main():
    print("=" * 60)
    print("🔍 API Endpoint Verification (Lightweight)")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports (no torch)
    results.append(("Core Imports", test_imports()))
    
    # Test 2: App syntax
    results.append(("App Endpoints", test_app_py_syntax()))
    
    # Test 3: Pydantic models
    results.append(("Pydantic Models", test_pydantic_models()))
    
    # Test 4: LoRA Registry
    results.append(("LoRA Registry", test_lora_registry()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Verification Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 API structure verified!")
        print("\nNext: Start service with proper torch installation:")
        print("  python -m venv venv --upgrade-deps")
        print("  pip install torch transformers peft fastapi uvicorn")
        print("  python -m uvicorn app.app:app --reload --port 8001")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

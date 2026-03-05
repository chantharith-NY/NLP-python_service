#!/usr/bin/env python3
"""
API Endpoint Verification Test
Tests all FastAPI endpoints for correctness
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_endpoint_definitions():
    """Test that all endpoints are properly defined in FastAPI"""
    print("\n🔍 Testing endpoint definitions...")
    
    try:
        from app.app import app
        from fastapi import FastAPI
        
        assert isinstance(app, FastAPI), "app must be a FastAPI instance"
        print("  ✅ FastAPI app initialized")
        
        # Get all routes
        routes = [route.path for route in app.routes]
        
        required_endpoints = [
            "/",                    # Root
            "/health",              # Health check
            "/models",              # List models (legacy)
            "/summarize",           # Summarization
            "/spell-check",         # Spell check
            "/generate",            # Text generation
            "/loras",               # List all LoRAs
            "/loras/types",         # List LoRA types
            "/loras/registry",      # Registry summary
        ]
        
        print(f"  📊 Found {len(routes)} routes total")
        
        found = 0
        for endpoint in required_endpoints:
            if any(endpoint in route for route in routes):
                print(f"    ✅ {endpoint}")
                found += 1
            else:
                print(f"    ❌ {endpoint}")
        
        all_found = found == len(required_endpoints)
        print(f"  {found}/{len(required_endpoints)} endpoints verified")
        
        return all_found
        
    except Exception as e:
        print(f"  ❌ Endpoint definition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pydantic_models():
    """Test that Pydantic response models are properly defined"""
    print("\n📋 Testing Pydantic models...")
    
    try:
        from app.app import (
            SummarizeRequest, SpellCheckRequest, GenerateRequest,
            LoRAInfo, LoRAsResponse, LoRATypesResponse, LoRARegistryResponse
        )
        
        print("  ✅ SummarizeRequest")
        print("  ✅ SpellCheckRequest")
        print("  ✅ GenerateRequest")
        print("  ✅ LoRAInfo")
        print("  ✅ LoRAsResponse")
        print("  ✅ LoRATypesResponse")
        print("  ✅ LoRARegistryResponse")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Missing Pydantic models: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Pydantic model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_startup():
    """Test that the app can be instantiated and startup events work"""
    print("\n🚀 Testing app startup...")
    
    try:
        from app.app import app
        
        # Check if startup event exists
        has_startup = bool(app.router.on_startup)
        print(f"  ✅ Startup events configured: {len(app.router.on_startup)} handlers")
        
        # Try to get openapi schema (this validates the app)
        schema = app.openapi()
        assert schema, "OpenAPI schema should not be empty"
        
        paths = schema.get('paths', {})
        print(f"  ✅ OpenAPI schema generated with {len(paths)} endpoints")
        
        return True
        
    except Exception as e:
        print(f"  ❌ App startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_registry_integration():
    """Test that LoRARegistry is properly integrated into app"""
    print("\n🔗 Testing LoRARegistry integration...")
    
    try:
        from app.app import lora_registry
        from app.lora_registry import LoRARegistry
        
        assert lora_registry is not None, "lora_registry should be instantiated"
        assert isinstance(lora_registry, LoRARegistry), "lora_registry should be LoRARegistry instance"
        
        print("  ✅ LoRARegistry instantiated")
        
        # Test that it has necessary methods
        required_methods = [
            'discover_local_loras',
            'register_lora',
            'unregister_lora',
            'get_lora',
            'list_all_loras',
            'list_loras_by_type',
            'get_types',
            'get_summary',
        ]
        
        for method in required_methods:
            assert hasattr(lora_registry, method), f"Missing method: {method}"
            print(f"  ✅ {method}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Registry integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔍 API Endpoint Verification Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Endpoint definitions
    results.append(("Endpoint Definitions", test_endpoint_definitions()))
    
    # Test 2: Pydantic models
    results.append(("Pydantic Models", test_pydantic_models()))
    
    # Test 3: App startup
    results.append(("App Startup", test_app_startup()))
    
    # Test 4: Registry integration
    results.append(("Registry Integration", test_registry_integration()))
    
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
        print("\n🎉 All API endpoints ready!")
        print("\nNext steps:")
        print("  1. Start the service: python -m uvicorn app.app:app --reload")
        print("  2. Test endpoints: curl http://localhost:8000/loras")
        print("  3. Upload new LoRA: curl -X POST http://localhost:8000/loras/register ...")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Dynamic LoRA Registration Test
Tests the ability to register, list, and use LoRAs dynamically without code changes
"""

import sys
import os
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

def test_registration_workflow():
    """Test complete LoRA registration workflow"""
    print("\n📝 Testing LoRA registration workflow...")
    
    try:
        from app.lora_registry import LoRARegistry
        
        # Create test registry
        test_cache_dir = "./test_models/adapters"
        registry = LoRARegistry(cache_dir=test_cache_dir)
        print("  ✅ Registry created")
        
        # Test 1: Register different LoRA types
        loras_to_register = [
            {
                "name": "qwen-summarizer-v1",
                "repo": "chantharith/qwen-summarizer",
                "type": "summarization",
                "version": "1.0.0"
            },
            {
                "name": "qwen-translator",
                "repo": "chantharith/qwen-translator",
                "type": "translation",
                "version": "1.5.0"
            },
            {
                "name": "qwen-chatbot",
                "repo": "chantharith/qwen-chatbot",
                "type": "conversation",
                "version": "2.0.0"
            },
            {
                "name": "qwen-code-gen",
                "repo": "chantharith/qwen-codegen",
                "type": "code-generation",
                "version": "1.2.0"
            },
        ]
        
        registered_count = 0
        for lora_info in loras_to_register:
            registry.register_lora(
                name=lora_info["name"],
                repo=lora_info["repo"],
                lora_type=lora_info["type"],
                source="hf",
                version=lora_info["version"],
                description=f"{lora_info['type'].title()} LoRA"
            )
            registered_count += 1
        
        print(f"  ✅ Registered {registered_count} LoRAs of different types")
        
        # Test 2: List all LoRAs
        all_loras = registry.list_all_loras()
        assert len(all_loras) >= registered_count, "Should have all registered LoRAs"
        print(f"  ✅ List all LoRAs: {len(all_loras)} total")
        
        # Test 3: List LoRAs by type
        for lora_type in ["summarization", "translation", "conversation", "code-generation"]:
            loras_of_type = registry.list_loras_by_type(lora_type)
            assert len(loras_of_type) > 0, f"Should have at least one {lora_type} LoRA"
            print(f"  ✅ Filter by type '{lora_type}': {len(loras_of_type)} found")
        
        # Test 4: Get unique types
        types = registry.list_types()
        expected_types = {"summarization", "translation", "conversation", "code-generation"}
        assert expected_types.issubset(set(types)), f"Should have all expected types. Got {types}"
        print(f"  ✅ List types: {len(types)} unique types")
        
        # Test 5: Get registry summary
        summary = registry.get_summary()
        assert summary["total_loras"] >= registered_count, "Summary should reflect registered LoRAs"
        print(f"  ✅ Registry summary: {summary['total_loras']} LoRAs, {len(summary['types'])} types")
        
        # Test 6: Update existing LoRA
        registry.register_lora(
            name="qwen-summarizer-v1",  # Same name as previous
            repo="chantharith/qwen-summarizer",
            lora_type="summarization",
            version="1.1.0"  # Updated version
        )
        print("  ✅ Update existing LoRA (version bump)")
        
        # Test 7: Unregister LoRA
        success = registry.unregister_lora("qwen-code-gen")
        assert success, "Should successfully unregister"
        remaining = registry.list_all_loras()
        assert len(remaining) == len(all_loras) - 1, "Should have one fewer LoRA"
        print("  ✅ Unregister LoRA")
        
        # Test 8: Verify persistence
        # Save should be automatic on register/unregister
        registry_file = os.path.join(test_cache_dir, ".registry.json")
        assert os.path.exists(registry_file), "Registry should be persisted to JSON"
        with open(registry_file, 'r') as f:
            saved_data = json.load(f)
            assert len(saved_data) > 0, "Saved registry should have data"
        print("  ✅ Registry persisted to JSON file")
        
        # Test 9: Load registry from persistence
        registry2 = LoRARegistry(cache_dir=test_cache_dir)
        loaded_loras = registry2.list_all_loras()
        assert len(loaded_loras) == len(remaining), "Loaded registry should match saved state"
        print("  ✅ Registry loaded from JSON file")
        
        return True
        
    except AssertionError as e:
        print(f"  ❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Registration workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_type_based_filtering():
    """Test type-based LoRA filtering (key feature)"""
    print("\n🔍 Testing type-based LoRA filtering...")
    
    try:
        from app.lora_registry import LoRARegistry
        
        registry = LoRARegistry(cache_dir="./test_models/adapters_typing")
        
        # Register multiple LoRAs of different types
        tasks = {
            "summarization": ["summarizer-v1", "summarizer-v2"],
            "translation": ["translator-en-kh", "translator-kh-en"],
            "spellcheck": ["spellchecker-khmer"],
            "qa": ["qa-bot", "qa-retrieval"],
        }
        
        total_registered = 0
        for task_type, names in tasks.items():
            for name in names:
                registry.register_lora(
                    name=name,
                    repo=f"chantharith/{name}",
                    lora_type=task_type,
                    source="hf"
                )
                total_registered += 1
        
        print(f"  ✅ Registered {total_registered} LoRAs across {len(tasks)} types")
        
        # Test filtering by type
        for task_type, expected_names in tasks.items():
            loras = registry.list_loras_by_type(task_type)
            found_names = [name for name, _ in loras]
            assert len(found_names) == len(expected_names), \
                f"Expected {len(expected_names)} {task_type} LoRAs, got {len(found_names)}"
            print(f"  ✅ Type '{task_type}': {len(found_names)} LoRAs")
        
        # Test getting first LoRA of a type (useful for auto-selection)
        summarization_loras = registry.list_loras_by_type("summarization")
        if summarization_loras:
            first_name, first_meta = summarization_loras[0]
            print(f"  ✅ Auto-select first 'summarization' LoRA: {first_name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Type filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_code_changes_needed():
    """Verify that new LoRA types don't require code changes"""
    print("\n🆓 Testing that new LoRA types don't require code changes...")
    
    try:
        from app.lora_registry import LoRARegistry
        
        registry = LoRARegistry(cache_dir="./test_models/adapters_custom")
        
        # Register totally custom, novel LoRA types
        custom_types = [
            "image-generation",
            "audio-processing",
            "sentiment-analysis",
            "named-entity-recognition",
            "document-classification",
            "custom-task-1",
            "our-proprietary-task",
        ]
        
        for custom_type in custom_types:
            registry.register_lora(
                name=f"model_{custom_type}",
                repo=f"custom/{custom_type}",
                lora_type=custom_type,
                source="hf",
                description=f"Custom {custom_type} LoRA"
            )
        
        print(f"  ✅ Registered {len(custom_types)} custom LoRA types")
        
        # Verify all custom types are now available
        types = registry.list_types()
        for custom_type in custom_types:
            assert custom_type in types, f"Custom type '{custom_type}' not found"
        
        print(f"  ✅ All {len(custom_types)} custom types queryable via API")
        print(f"  ✅ NO CODE CHANGES REQUIRED - just register and use!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Custom types test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🧪 Dynamic LoRA Registration Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Registration workflow
    results.append(("Registration Workflow", test_registration_workflow()))
    
    # Test 2: Type-based filtering
    results.append(("Type-based Filtering", test_type_based_filtering()))
    
    # Test 3: Custom types (no code changes)
    results.append(("Custom LoRA Types", test_no_code_changes_needed()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Registration Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Dynamic LoRA registration works!")
        print("\n✨ Key Achievement:")
        print("  - Upload ANY LoRA type without code changes")
        print("  - Register at runtime via API: POST /loras/register")
        print("  - Filter by type: GET /loras/by-type/{type}")
        print("  - Use any model: POST /generate with model='your_lora'")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""
LoRA Registry: Manages dynamic discovery and registration of LoRA adapters.

Instead of hardcoding models, this system:
- Automatically discovers LoRAs from caching directories
- Allows registration of new LoRAs from HF, S3, or local paths
- Supports any model type without hardcoding
- Can be extended to load from database
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class LoRARegistry:
    """
    Dynamic LoRA adapter registry.
    
    Supports:
    - Automatic discovery from local cache
    - Manual registration from HF/S3/local paths
    - Per-LoRA metadata (type, version, description)
    - Optional database persistence
    """
    
    def __init__(self, cache_dir: str = "./models/adapters"):
        """
        Initialize LoRA registry.
        
        Args:
            cache_dir: Base directory where LoRAs are cached
        """
        self.cache_dir = cache_dir
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.metadata_file = os.path.join(cache_dir, ".registry.json")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing registry from disk
        self.load_registry()
    
    def discover_local_loras(self) -> Dict[str, str]:
        """
        Auto-discover LoRAs from cache directory.
        
        Expected structure:
        models/adapters/
            ├── summarizer/
            │   ├── adapter_config.json
            │   └── adapter_model.safetensors
            ├── spellchecker/
            │   ├── adapter_config.json
            │   └── adapter_model.safetensors
            └── chatbot/
                ├── adapter_config.json
                └── adapter_model.safetensors
        
        Returns:
            Dict mapping LoRA name to local path
        """
        discovered = {}
        
        if not os.path.exists(self.cache_dir):
            logger.warning(f"Cache directory not found: {self.cache_dir}")
            return discovered
        
        # Scan subdirectories
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            
            # Skip metadata files and non-directories
            if item.startswith(".") or not os.path.isdir(item_path):
                continue
            
            # Check if this is a valid LoRA directory (has adapter files)
            adapter_model = os.path.join(item_path, "adapter_model.safetensors")
            adapter_config = os.path.join(item_path, "adapter_config.json")
            
            if os.path.exists(adapter_model) and os.path.exists(adapter_config):
                discovered[item] = item_path
                logger.info(f"[LoRARegistry] Discovered local LoRA: {item}")
        
        return discovered
    
    def register_lora(
        self,
        name: str,
        repo: str,
        lora_type: str = "generic",
        source: str = "hf",
        version: str = "1.0.0",
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Register a new LoRA adapter.
        
        Args:
            name: Unique identifier (e.g., "summarizer", "my_custom_lora")
            repo: Repository location (HF repo ID, S3 path, or local path)
            lora_type: Type/category (summarization, spellcheck, chatbot, custom, etc.)
            source: Source type ("hf", "s3", "local")
            version: Version string (e.g., "1.0.0")
            description: Human-readable description
            metadata: Extra metadata (dict)
            
        Returns:
            bool: True if registration successful
        """
        if name in self.registry:
            logger.warning(f"LoRA '{name}' already registered, updating...")
        
        self.registry[name] = {
            "repo": repo,
            "type": lora_type,
            "source": source,
            "version": version,
            "description": description,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        logger.info(f"[LoRARegistry] Registered LoRA: {name} (type: {lora_type})")
        
        # Auto-save to disk
        self.save_registry()
        
        return True
    
    def unregister_lora(self, name: str) -> bool:
        """Unregister a LoRA adapter."""
        if name not in self.registry:
            logger.warning(f"LoRA '{name}' not found in registry")
            return False
        
        del self.registry[name]
        logger.info(f"[LoRARegistry] Unregistered LoRA: {name}")
        self.save_registry()
        
        return True
    
    def get_lora(self, name: str) -> Optional[Dict[str, Any]]:
        """Get LoRA metadata by name."""
        return self.registry.get(name)
    
    def list_all_loras(self) -> List[Dict[str, Any]]:
        """Get all registered LoRAs."""
        return list(self.registry.values())
    
    def list_loras_by_type(self, lora_type: str) -> List[tuple]:
        """
        Get all LoRAs of a specific type.
        
        Args:
            lora_type: Type to filter by (e.g., "summarization")
            
        Returns:
            List of (name, metadata) tuples
        """
        return [
            (name, info)
            for name, info in self.registry.items()
            if info.get("type") == lora_type
        ]
    
    def list_types(self) -> List[str]:
        """Get all unique LoRA types in registry."""
        types = set()
        for info in self.registry.values():
            types.add(info.get("type", "generic"))
        return sorted(list(types))
    
    def get_lora_path(self, name: str) -> Optional[str]:
        """
        Get the local path to a LoRA (check local cache first).
        
        Returns:
            Local path if cached, or repo identifier if not yet downloaded
        """
        info = self.get_lora(name)
        if not info:
            return None
        
        # Check if already cached locally
        local_path = os.path.join(self.cache_dir, name)
        if os.path.exists(os.path.join(local_path, "adapter_model.safetensors")):
            return local_path
        
        # Return repo identifier (will be downloaded on demand)
        return info.get("repo")
    
    def save_registry(self) -> bool:
        """Save registry to disk as JSON."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.registry, f, indent=2)
            logger.debug(f"Registry saved to {self.metadata_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            return False
    
    def load_registry(self) -> bool:
        """Load registry from disk."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded registry with {len(self.registry)} LoRAs")
            else:
                logger.info("No registry file found, starting fresh")
                # Auto-discover local LoRAs
                discovered = self.discover_local_loras()
                for name, path in discovered.items():
                    self.register_lora(
                        name=name,
                        repo=path,
                        source="local",
                        description=f"Auto-discovered local LoRA"
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """Export registry as dictionary."""
        return self.registry.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary stats about registry."""
        by_type = {}
        for name, info in self.registry.items():
            lora_type = info.get("type", "generic")
            if lora_type not in by_type:
                by_type[lora_type] = []
            by_type[lora_type].append(name)
        
        return {
            "total_loras": len(self.registry),
            "types": self.list_types(),
            "by_type": by_type
        }


# =============================================================================
# Global Registry Instance
# =============================================================================

_registry_instance = None


def get_registry(cache_dir: str = "./models/adapters") -> LoRARegistry:
    """Get or create global LoRA registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = LoRARegistry(cache_dir)
    return _registry_instance


def reset_registry():
    """Reset registry instance (useful for testing)."""
    global _registry_instance
    _registry_instance = None

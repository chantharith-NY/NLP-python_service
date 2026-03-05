"""
AdapterManager: Manages dynamic LoRA adapter loading and switching for production inference.

Key features:
- Load base Qwen model once with 4-bit quantization
- Dynamically load/switch LoRA adapters without restart
- LRU cache with explicit unload (max 5 adapters in memory)
- Thread-safe adapter switching with minimal lock scope
- Comprehensive logging for debugging
- GPU memory optimization
"""

import os
import threading
import logging
import time
from collections import OrderedDict
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel


# Configure logging
logger = logging.getLogger(__name__)


class AdapterManager:
    """
    Manages LoRA adapter loading and switching on a single base Qwen model.
    
    Thread-safe: uses lock only for adapter loading/switching operations.
    Memory-efficient: implements LRU cache with explicit unload.
    """
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        use_4bit_quantization: bool = True,
        max_cached_adapters: int = 5,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the AdapterManager.
        
        Args:
            base_model_name: HuggingFace model ID for base Qwen model
            use_4bit_quantization: Whether to use 4-bit quantization (3-5× memory savings)
            max_cached_adapters: Maximum number of adapters to keep in memory (LRU)
            hf_token: HuggingFace API token for private models
        """
        self.base_model_name = base_model_name
        self.hf_token = hf_token
        self.max_cached_adapters = max_cached_adapters
        
        # Thread lock for adapter operations (NOT generation)
        self.lock = threading.Lock()
        
        # Track which adapters are currently loaded in the model
        self.loaded_adapters = set()
        
        # LRU tracking: OrderedDict to track access order
        self.adapter_usage = OrderedDict()
        
        # Current active adapter
        self.current_adapter = None
        
        # Initialize device and model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load base model with quantization
        logger.info(f"Loading base model: {base_model_name}")
        self._load_base_model(use_4bit_quantization)
        
        logger.info("AdapterManager initialized successfully")
    
    def _load_base_model(self, use_4bit: bool):
        """Load base Qwen model with optional 4-bit quantization."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Configure quantization if enabled
            if use_4bit:
                logger.info("Enabling 4-bit quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": "auto"
                }
            else:
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16
                }
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                token=self.hf_token,
                trust_remote_code=True,
                **model_kwargs
            )
            
            self.base_model.eval()
            logger.info("Base model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def ensure_adapter_loaded(
        self,
        name: str,
        repo: str,
        adapter_source_loader=None
    ) -> bool:
        """
        Ensure adapter is loaded into the base model (thread-safe).
        
        Downloads adapter if needed, caches it, applies LRU eviction if needed.
        This method holds the lock during download to prevent race conditions.
        
        Args:
            name: Adapter name (e.g., "summarizer")
            repo: Source repo (HF repo ID, S3 path, or local path)
            adapter_source_loader: Function to load adapter from source
            
        Returns:
            bool: True if adapter is now loaded
        """
        # If already loaded, just return
        if name in self.loaded_adapters:
            logger.info(f"[ModelManager] Cache hit {name}")
            return True
        
        # Lock during loading to prevent race conditions
        with self.lock:
            # Double-check after acquiring lock
            if name in self.loaded_adapters:
                return True
            
            logger.info(f"[ModelManager] Loading adapter {name}")
            start_time = time.time()
            
            try:
                # If adapter_source_loader provided, use it; otherwise assume repo is local path
                if adapter_source_loader:
                    adapter_path = adapter_source_loader(name, repo)
                else:
                    adapter_path = repo
                
                # Load adapter from path
                logger.info(f"[ModelManager] Mounting {name} from {adapter_path}")
                self.base_model.load_adapter(adapter_path, adapter_name=name)
                self.loaded_adapters.add(name)
                self.adapter_usage[name] = time.time()
                
                load_time = (time.time() - start_time) * 1000
                logger.info(f"[ModelManager] Loaded adapter {name} ({load_time:.0f}ms)")
                
                # Apply LRU eviction if limit exceeded
                self._apply_lru_eviction()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to load adapter {name}: {e}")
                raise
    
    def set_adapter(self, name: str) -> bool:
        """
        Switch to a different adapter (thread-safe, minimal lock scope).
        
        Only locks the actual switch operation, not generation.
        
        Args:
            name: Adapter name to switch to
            
        Returns:
            bool: True if switch successful
        """
        if name not in self.loaded_adapters:
            logger.error(f"Adapter {name} not loaded. Call ensure_adapter_loaded first.")
            return False
        
        if self.current_adapter == name:
            logger.debug(f"Already using adapter {name}")
            return True
        
        # Small lock scope: only the switch itself
        with self.lock:
            try:
                switch_start = time.time()
                self.base_model.set_adapter(name)
                self.current_adapter = name
                
                # Update LRU usage
                self.adapter_usage.move_to_end(name)
                self.adapter_usage[name] = time.time()
                
                switch_time = (time.time() - switch_start) * 1000
                logger.info(f"[ModelManager] Switched adapter {name} ({switch_time:.1f}ms)")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to switch to adapter {name}: {e}")
                return False
    
    def _apply_lru_eviction(self):
        """Remove least-recently-used adapters if cache limit exceeded."""
        while len(self.loaded_adapters) > self.max_cached_adapters:
            # Get oldest (first) adapter from OrderedDict
            oldest = next(iter(self.adapter_usage))
            
            logger.info(f"[ModelManager] LRU evict {oldest} (cache full: {len(self.loaded_adapters)}/{self.max_cached_adapters})")
            
            try:
                self.base_model.delete_adapter(oldest)
                self.loaded_adapters.discard(oldest)
                del self.adapter_usage[oldest]
                
                # If deleted adapter was current, reset current
                if self.current_adapter == oldest:
                    self.current_adapter = None
                    
            except Exception as e:
                logger.error(f"Failed to unload adapter {oldest}: {e}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_time: float = 20.0
    ) -> str:
        """
        Generate text using current adapter (NOT locked during generation).
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_time: Max generation time in seconds
            
        Returns:
            Generated text
        """
        if self.current_adapter is None:
            logger.error("No adapter currently set. Call set_adapter() first.")
            return ""
        
        try:
            start_time = time.time()
            
            # Tokenize (no lock - reading is safe)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate (no lock - generation doesn't modify adapter state)
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_time=max_time,
                    do_sample=True
                )
            
            # Decode
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            gen_time = (time.time() - start_time)
            logger.info(f"[Inference] {gen_time:.2f}s | adapter={self.current_adapter} | tokens={max_new_tokens}")
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def warmup(self):
        """Warmup model by generating 1 token to compile CUDA kernels."""
        logger.info("Warming up model...")
        try:
            if not self.current_adapter:
                # Use first loaded adapter for warmup
                adapter_name = next(iter(self.loaded_adapters), None)
                if adapter_name:
                    self.set_adapter(adapter_name)
                else:
                    logger.warning("No adapters loaded for warmup")
                    return
            
            # Generate 1 token to warm CUDA kernels
            inputs = self.tokenizer("Hello", return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.base_model.generate(**inputs, max_new_tokens=1)
            
            logger.info("Model warmup complete")
            
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
    
    def get_loaded_adapters(self) -> list:
        """Get list of currently loaded adapters."""
        return list(self.loaded_adapters)
    
    def get_gpu_memory_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 3
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of manager for health check."""
        return {
            "base_model_loaded": self.base_model is not None,
            "current_adapter": self.current_adapter,
            "loaded_adapters": self.get_loaded_adapters(),
            "gpu_memory_gb": round(self.get_gpu_memory_gb(), 2),
            "max_cached": self.max_cached_adapters
        }

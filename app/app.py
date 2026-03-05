# from fastapi import FastAPI
# from pydantic import BaseModel
# from time import time

# from model_loader import load_models
# from summarizer import summarize_text
# from spell_checker import spell_check_text

# app = FastAPI(title="Khmer NLP Service")

# load_models()


# class NLPRequest(BaseModel):
#     text: str
#     model_key: str


# @app.post("/summarize")
# def summarize(req: NLPRequest):
#     start = time()
#     result = summarize_text(req.model_key, req.text)
#     exec_time = int((time() - start) * 1000)

#     return {
#         "output": result,
#         "execution_time_ms": exec_time
#     }


# @app.post("/spell-check")
# def spell_check(req: NLPRequest):
#     start = time()
#     result = spell_check_text(req.model_key, req.text)
#     exec_time = int((time() - start) * 1000)

#     return {
#         "output": result,
#         "execution_time_ms": exec_time
#     }
"""
NLP Service API - Production-grade FastAPI service for dynamic LoRA model inference.

Endpoints:
  POST /summarize       - Summarize Khmer text
  POST /spell-check     - Check and correct spelling
  POST /generate        - Generate text with any model (flexible)
  GET  /models          - List available models
  GET  /health          - Health check with detailed status
  POST /reload-models   - Reload model registry (optional)
"""

import logging
from typing import Optional, Dict, Any
from time import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from .adapter_manager import AdapterManager
from .s3_downloader import load_adapter_from_source
from .lora_registry import get_registry
from .env import Config
from . import config


# =============================================================================
# Setup Logging
# =============================================================================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=config.LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Request/Response Models
# =============================================================================

class TextRequest(BaseModel):
    """Basic text input request"""
    text: str = Field(..., min_length=1, max_length=10000)


class GenerateRequest(BaseModel):
    """Flexible generation request with full control"""
    model: str = Field(..., description="Model name (e.g., 'summarizer')")
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_time: float = Field(default=20.0, ge=1.0, le=120.0)


class SummarizeRequest(BaseModel):
    """Summarization request"""
    text: str = Field(..., min_length=1, max_length=50000, description="Khmer text to summarize")


class SpellCheckRequest(BaseModel):
    """Spell checking request"""
    text: str = Field(..., min_length=1, max_length=10000, description="Khmer text to correct")


class GenerateResponse(BaseModel):
    """Generation response"""
    response: str
    model: str
    execution_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    base_model_loaded: bool
    loaded_adapters: list
    gpu_memory_gb: float
    max_cached_adapters: int


class ModelsResponse(BaseModel):
    """Models list response"""
    models: list


class ModelInfo(BaseModel):
    """Individual model info"""
    name: str
    max_tokens: int
    temperature: float
    description: str


class LoRAInfo(BaseModel):
    """Individual LoRA info"""
    name: str
    type: str
    version: str
    source: str
    description: str
    repo: str


class LoRARegistrationRequest(BaseModel):
    """Request to register a new LoRA"""
    name: str = Field(..., min_length=1, max_length=100, description="Unique LoRA identifier")
    repo: str = Field(..., description="Repository (HF repo ID, S3 path, or local path)")
    lora_type: str = Field(default="custom", description="Type/category (summarization, spellcheck, etc.)")
    source: str = Field(default="hf", description="Source type (hf, s3, local)")
    version: str = Field(default="1.0.0", description="Version string")
    description: str = Field(default="", description="Human-readable description")


class LoRAsResponse(BaseModel):
    """List of LoRAs response"""
    loras: List[LoRAInfo]
    total: int


class LoRATypesResponse(BaseModel):
    """Available LoRA types response"""
    types: List[str]
    count: int


class LoRARegistryResponse(BaseModel):
    """Full registry response"""
    total_loras: int
    types: List[str]
    by_type: Dict[str, List[str]]


# =============================================================================
# Initialize FastAPI App & AdapterManager
# =============================================================================

app = FastAPI(
    title="NLP Service",
    description="Production-grade Khmer NLP inference with dynamic LoRA models",
    version="2.0.0"
)

# Global adapter manager instance (initialized on startup)
adapter_manager: Optional[AdapterManager] = None

# Global LoRA registry instance
lora_registry = None


@app.on_event("startup")
async def startup_event():
    """Initialize AdapterManager and LoRA Registry on startup"""
    global adapter_manager, lora_registry
    
    logger.info("=" * 80)
    logger.info("Starting NLP Service")
    logger.info("=" * 80)
    
    try:
        # Initialize LoRA registry
        lora_registry = get_registry(Config.ADAPTER_CACHE_DIR)
        logger.info(f"LoRA Registry initialized: {lora_registry.get_summary()}")
        
        # Initialize adapter manager
        adapter_manager = AdapterManager(
            base_model_name=Config.BASE_MODEL,
            use_4bit_quantization=Config.USE_4BIT_QUANTIZATION,
            max_cached_adapters=Config.MAX_ADAPTERS_CACHED,
            hf_token=Config.HUGGINGFACE_HUB_TOKEN
        )
        
        # Load and warmup first available LoRA (if any)
        all_loras = lora_registry.list_all_loras()
        if all_loras:
            default_lora_name = list(lora_registry.registry.keys())[0]
            logger.info(f"Loading and warming up default LoRA: {default_lora_name}")
            
            lora_info = lora_registry.get_lora(default_lora_name)
            lora_path = lora_registry.get_lora_path(default_lora_name)
            
            adapter_manager.ensure_adapter_loaded(
                default_lora_name,
                lora_path,
                adapter_source_loader=load_adapter_from_source
            )
            
            adapter_manager.set_adapter(default_lora_name)
            adapter_manager.warmup()
        
        logger.info("Startup complete - service ready")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint with detailed status.
    
    Returns GPU memory usage, loaded adapters, and readiness status.
    """
    if not adapter_manager:
        return HealthResponse(
            status="initializing",
            base_model_loaded=False,
            loaded_adapters=[],
            gpu_memory_gb=0.0,
            max_cached_adapters=0
        )
    
    status_info = adapter_manager.get_status()
    
    return HealthResponse(
        status="ok",
        **status_info
    )


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """
    List all available LoRAs with metadata.
    
    Returns all registered LoRA adapters with their types and parameters.
    """
    if not adapter_manager or not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    models = []
    
    for lora_name, lora_info in lora_registry.registry.items():
        models.append({
            "name": lora_name,
            "max_tokens": lora_info.get("metadata", {}).get("max_tokens", 256),
            "temperature": lora_info.get("metadata", {}).get("temperature", 0.7),
            "description": lora_info.get("description", "")
        })
    
    return ModelsResponse(models=models)


@app.post("/summarize", response_model=GenerateResponse)
async def summarize(req: SummarizeRequest) -> GenerateResponse:
    """
    Summarize Khmer text.
    
    Uses the first registered LoRA of type "summarization" (if available),
    otherwise falls back to "summarizer" by name.
    
    Returns: Concise summary of input text
    """
    if not adapter_manager or not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        start_time = time()
        
        # Find summarization LoRA: first by type, then by name
        summarizer_lora = None
        
        # Try to find by type first
        summarization_loras = lora_registry.list_loras_by_type("summarization")
        if summarization_loras:
            summarizer_lora = summarization_loras[0][0]
        else:
            # Fall back to "summarizer" by name
            if lora_registry.get_lora("summarizer"):
                summarizer_lora = "summarizer"
        
        if not summarizer_lora:
            available = list(lora_registry.registry.keys())
            raise HTTPException(
                status_code=400,
                detail=f"No summarization LoRA found. Available: {available}"
            )
        
        lora_info = lora_registry.get_lora(summarizer_lora)
        lora_path = lora_registry.get_lora_path(summarizer_lora)
        
        # Ensure adapter is loaded and set active
        adapter_manager.ensure_adapter_loaded(
            summarizer_lora,
            lora_path,
            adapter_source_loader=load_adapter_from_source
        )
        adapter_manager.set_adapter(summarizer_lora)
        
        # Create prompt (Khmer instruction format)
        prompt = f"""<|im_start|>user
សូមសង្ខេបអត្ថបទខាងក្រោមជាអត្ថបទខ្លី និងច្បាស់លាស់។
{req.text}<|im_end|>
<|im_start|>assistant
"""
        
        # Get parameters from metadata if available
        metadata = lora_info.get("metadata", {})
        max_tokens = metadata.get("max_tokens", 200)
        temperature = metadata.get("temperature", 0.3)
        top_p = metadata.get("top_p", 0.9)
        
        # Generate
        response = adapter_manager.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_time=Config.DEFAULT_MAX_TIME
        )
        
        # Clean response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
        
        exec_time = (time() - start_time) * 1000
        
        return GenerateResponse(
            response=response,
            model=summarizer_lora,
            execution_time_ms=exec_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarize failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/spell-check", response_model=GenerateResponse)
async def spell_check(req: SpellCheckRequest) -> GenerateResponse:
    """
    Check and correct spelling in Khmer text.
    
    Uses the first registered LoRA of type "spellcheck" (if available),
    otherwise falls back to "spellchecker" by name.
    
    Returns: Corrected text
    """
    if not adapter_manager or not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        start_time = time()
        
        # Find spellcheck LoRA: first by type, then by name
        spellcheck_lora = None
        
        # Try to find by type first
        spellcheck_loras = lora_registry.list_loras_by_type("spellcheck")
        if spellcheck_loras:
            spellcheck_lora = spellcheck_loras[0][0]
        else:
            # Fall back to "spellchecker" by name
            if lora_registry.get_lora("spellchecker"):
                spellcheck_lora = "spellchecker"
        
        if not spellcheck_lora:
            available = list(lora_registry.registry.keys())
            logger.warning(f"No spellcheck LoRA found. Using first available: {available[0] if available else 'none'}")
            if available:
                spellcheck_lora = available[0]
            else:
                raise HTTPException(status_code=400, detail="No LoRAs available")
        
        lora_info = lora_registry.get_lora(spellcheck_lora)
        lora_path = lora_registry.get_lora_path(spellcheck_lora)
        
        # Ensure adapter is loaded and set active
        adapter_manager.ensure_adapter_loaded(
            spellcheck_lora,
            lora_path,
            adapter_source_loader=load_adapter_from_source
        )
        adapter_manager.set_adapter(spellcheck_lora)
        
        # Create prompt for spell checking
        prompt = f"""<|im_start|>user
សូមពិនិត្យ និងកែសម្រួលកំហុសអក្ខរាវិរុទ្ធក្នុងអត្ថបទខាងក្រោម។
{req.text}<|im_end|>
<|im_start|>assistant
"""
        
        # Get parameters from metadata if available
        metadata = lora_info.get("metadata", {})
        max_tokens = metadata.get("max_tokens", 100)
        temperature = metadata.get("temperature", 0.5)
        top_p = metadata.get("top_p", 0.9)
        
        # Generate
        response = adapter_manager.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_time=Config.DEFAULT_MAX_TIME
        )
        
        # Clean response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
        
        exec_time = (time() - start_time) * 1000
        
        return GenerateResponse(
            response=response,
            model=spellcheck_lora,
            execution_time_ms=exec_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Spell check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Flexible generation endpoint - use ANY registered LoRA with custom parameters.
    
    Args:
        model: LoRA name (any registered LoRA, e.g., "summarizer", "chatbot", "custom")
        prompt: Input prompt
        max_tokens: Maximum tokens to generate (1-2048)
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling p (0.0-1.0)
        max_time: Max generation time in seconds (1-120)
    
    Returns: Generated text and metadata
    """
    if not adapter_manager or not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        start_time = time()
        
        # Validate LoRA exists in registry
        lora_info = lora_registry.get_lora(req.model)
        if not lora_info:
            available = list(lora_registry.registry.keys())
            raise HTTPException(
                status_code=400,
                detail=f"LoRA '{req.model}' not found. Available: {available}"
            )
        
        # Get LoRA path (local cache or repo ID)
        lora_path = lora_registry.get_lora_path(req.model)
        
        # Load and set adapter
        adapter_manager.ensure_adapter_loaded(
            req.model,
            lora_path,
            adapter_source_loader=load_adapter_from_source
        )
        adapter_manager.set_adapter(req.model)
        
        # Generate
        response = adapter_manager.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            max_time=req.max_time
        )
        
        exec_time = (time() - start_time) * 1000
        
        return GenerateResponse(
            response=response,
            model=req.model,
            execution_time_ms=exec_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-models")
async def reload_models() -> Dict[str, Any]:
    """
    Reload model registry (optional - for admin dashboard).
    
    Useful when adding models dynamically to Laravel DB.
    In future, can query Laravel ModelTb table for updated list.
    """
    logger.info("Reload models endpoint called")
    
    # TODO: Implement database-driven model loading
    # For now, just return current registry
    
    available_models = list(config.MODEL_REGISTRY.keys())
    
    return {
        "status": "success",
        "models_loaded": available_models,
        "count": len(available_models)
    }


# =============================================================================
# LoRA Management Endpoints (Dynamic LoRA Registry)
# =============================================================================

@app.get("/loras", response_model=LoRAsResponse)
async def list_loras() -> LoRAsResponse:
    """
    List all registered LoRA adapters with metadata.
    
    Unlike /models, this endpoint returns all LoRAs grouped by type,
    without predetermined types.
    """
    if not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    loras = []
    for name, info in lora_registry.registry.items():
        loras.append(LoRAInfo(
            name=name,
            type=info.get("type", "generic"),
            version=info.get("version", "1.0.0"),
            source=info.get("source", "unknown"),
            description=info.get("description", ""),
            repo=info.get("repo", "")
        ))
    
    return LoRAsResponse(loras=loras, total=len(loras))


@app.get("/loras/types", response_model=LoRATypesResponse)
async def list_lora_types() -> LoRATypesResponse:
    """
    List all unique LoRA types in the registry.
    
    Returns available categories: summarization, spellcheck, chatbot, etc.
    """
    if not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    types = lora_registry.list_types()
    return LoRATypesResponse(types=types, count=len(types))


@app.get("/loras/by-type/{lora_type}", response_model=LoRAsResponse)
async def list_loras_by_type(lora_type: str) -> LoRAsResponse:
    """
    List all LoRAs of a specific type.
    
    Args:
        lora_type: Type to filter by (e.g., "summarization", "spellcheck", "chatbot")
    
    Returns: All LoRAs matching the type
    """
    if not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    loras_by_type = lora_registry.list_loras_by_type(lora_type)
    
    loras = []
    for name, info in loras_by_type:
        loras.append(LoRAInfo(
            name=name,
            type=info.get("type", "generic"),
            version=info.get("version", "1.0.0"),
            source=info.get("source", "unknown"),
            description=info.get("description", ""),
            repo=info.get("repo", "")
        ))
    
    return LoRAsResponse(loras=loras, total=len(loras))


@app.get("/loras/registry", response_model=LoRARegistryResponse)
async def get_registry_summary() -> LoRARegistryResponse:
    """
    Get full registry summary with statistics.
    
    Returns count and grouping of LoRAs by type.
    """
    if not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    summary = lora_registry.get_summary()
    return LoRARegistryResponse(
        total_loras=summary["total_loras"],
        types=summary["types"],
        by_type=summary["by_type"]
    )


@app.post("/loras/register")
async def register_lora(req: LoRARegistrationRequest) -> Dict[str, Any]:
    """
    Register a new LoRA adapter dynamically.
    
    Allows users/admin dashboard to add new LoRAs without restarting.
    
    Args:
        name: Unique identifier (e.g., "my_summarizer_v2")
        repo: Repository location (HF repo ID, S3 path, or local path)
        lora_type: Type category (summarization, spellcheck, custom, etc.)
        source: Source type (hf, s3, local)
        version: Version number
        description: Description
    
    Returns: Registration status
    """
    if not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        success = lora_registry.register_lora(
            name=req.name,
            repo=req.repo,
            lora_type=req.lora_type,
            source=req.source,
            version=req.version,
            description=req.description
        )
        
        if success:
            logger.info(f"New LoRA registered: {req.name}")
            return {
                "status": "success",
                "message": f"LoRA '{req.name}' registered successfully",
                "lora_name": req.name,
                "type": req.lora_type
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to register LoRA")
            
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/loras/{lora_name}")
async def unregister_lora(lora_name: str) -> Dict[str, Any]:
    """
    Unregister a LoRA adapter.
    
    Removes it from the registry (doesn't delete cached files).
    """
    if not lora_registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        success = lora_registry.unregister_lora(lora_name)
        
        if success:
            # If currently loaded, unload it
            if adapter_manager and adapter_manager.current_adapter == lora_name:
                adapter_manager.current_adapter = None
                logger.info(f"Unloaded active LoRA: {lora_name}")
            
            return {
                "status": "success",
                "message": f"LoRA '{lora_name}' unregistered",
                "lora_name": lora_name
            }
        else:
            raise HTTPException(status_code=404, detail=f"LoRA '{lora_name}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unregistration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error: {exc.detail}")
    return {
        "status": "error",
        "detail": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "status": "error",
        "detail": "Internal server error",
        "status_code": 500
    }


# =============================================================================
# Root endpoint
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - info about service"""
    return {
        "service": "NLP Service",
        "version": "2.0.0",
        "status": "ready" if adapter_manager else "initializing",
        "endpoints": {
            "health": "GET /health",
            "models": "GET /models",
            "summarize": "POST /summarize",
            "spell_check": "POST /spell-check",
            "generate": "POST /generate",
            "reload_models": "POST /reload-models",
            "lora_management": {
                "list_all": "GET /loras",
                "list_types": "GET /loras/types",
                "list_by_type": "GET /loras/by-type/{type}",
                "registry_summary": "GET /loras/registry",
                "register": "POST /loras/register",
                "unregister": "DELETE /loras/{name}"
            }
        }
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        workers=Config.API_WORKERS,
        log_level=Config.LOG_LEVEL.lower()
    )
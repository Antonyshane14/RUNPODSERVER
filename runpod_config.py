import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RunPodRTX4090Config:
    """Optimized configuration for RTX 4090 on RunPod PyTorch template"""
    
    # RunPod paths
    WORKSPACE_DIR = Path('/workspace')
    UPLOAD_DIR = WORKSPACE_DIR / 'uploads'
    CACHE_DIR = WORKSPACE_DIR / 'cache'
    MODELS_DIR = WORKSPACE_DIR / 'models'
    LOGS_DIR = WORKSPACE_DIR / 'logs'
    TEMP_DIR = WORKSPACE_DIR / 'temp'
    
    # RTX 4090 optimizations
    GPU_MEMORY_FRACTION = 0.9  # Use 90% of 24GB = ~21.6GB
    MAX_BATCH_SIZE = 8  # Optimized for RTX 4090
    MIXED_PRECISION = True  # Enable for RTX 4090
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    MAX_WORKERS = 1  # Single worker for GPU efficiency
    MAX_CONCURRENT_REQUESTS = 3  # RTX 4090 can handle multiple requests
    
    # Audio processing optimizations
    AUDIO_SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 30  # seconds
    AUDIO_CHUNK_SIZE = 1024
    
    # Model settings optimized for RTX 4090
    WHISPER_MODEL = "large-v3"  # RTX 4090 can handle large models
    EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    AI_VOICE_MODEL = "as1605/Deepfake-audio-detection-V2"
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    
    # LLM settings
    OLLAMA_MODEL = "hermes3:8b"
    OLLAMA_NUM_GPU = 1
    OLLAMA_GPU_LAYERS = 33  # Full GPU utilization
    
    # Memory management
    CLEANUP_TEMP_FILES = True
    CACHE_RETENTION_HOURS = 24
    MODEL_CACHE_SIZE = 10  # Number of models to keep in memory
    
    # RTX 4090 specific CUDA settings
    CUDA_VISIBLE_DEVICES = "0"
    CUDA_MEMORY_POOL = "cuda_malloc_async"
    PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512,roundup_power2_divisions:16"
    
    @classmethod
    def setup_rtx4090_environment(cls):
        """Setup environment variables for RTX 4090 optimization"""
        
        # CUDA optimization
        os.environ['CUDA_VISIBLE_DEVICES'] = cls.CUDA_VISIBLE_DEVICES
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = cls.PYTORCH_CUDA_ALLOC_CONF
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
        
        # Model caching
        os.environ['TRANSFORMERS_CACHE'] = str(cls.MODELS_DIR)
        os.environ['HF_HOME'] = str(cls.MODELS_DIR)
        os.environ['TORCH_HOME'] = str(cls.MODELS_DIR)
        
        # Memory optimization
        os.environ['OMP_NUM_THREADS'] = '8'  # Optimize for CPU threads
        os.environ['MKL_NUM_THREADS'] = '8'
        
        # Logging
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        logger.info("🚀 RTX 4090 environment configured")
    
    @classmethod
    def setup_pytorch_optimizations(cls):
        """Setup PyTorch optimizations for RTX 4090"""
        
        if torch.cuda.is_available():
            # Memory management
            torch.cuda.set_per_process_memory_fraction(cls.GPU_MEMORY_FRACTION)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Mixed precision for RTX 4090
            if cls.MIXED_PRECISION:
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Memory pool optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = cls.PYTORCH_CUDA_ALLOC_CONF
            
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"🎮 GPU: {device_name}")
            logger.info(f"💾 Total Memory: {total_memory:.1f}GB")
            logger.info(f"🔧 Using {cls.GPU_MEMORY_FRACTION*100}% ({total_memory*cls.GPU_MEMORY_FRACTION:.1f}GB)")
        else:
            logger.warning("❌ CUDA not available")
    
    @classmethod
    def setup_directories(cls):
        """Create all necessary directories"""
        for directory in [cls.UPLOAD_DIR, cls.CACHE_DIR, cls.MODELS_DIR, 
                         cls.LOGS_DIR, cls.TEMP_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Directories created in {cls.WORKSPACE_DIR}")
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get detailed device information"""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "device": "cuda:0",
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": props.total_memory,
                "allocated_memory": torch.cuda.memory_allocated(0),
                "cached_memory": torch.cuda.memory_reserved(0),
                "memory_fraction": cls.GPU_MEMORY_FRACTION,
                "mixed_precision": cls.MIXED_PRECISION
            }
        return {"device": "cpu", "name": "CPU"}
    
    @classmethod
    def get_public_url(cls):
        """Get RunPod public URL"""
        pod_id = os.getenv('RUNPOD_POD_ID', 'unknown')
        return f"https://{pod_id}-{cls.PORT}.proxy.runpod.net"

# Initialize on import
RunPodRTX4090Config.setup_rtx4090_environment()
RunPodRTX4090Config.setup_pytorch_optimizations()
RunPodRTX4090Config.setup_directories()

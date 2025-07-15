import os
import sys
import asyncio
import logging
import uvicorn
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging for RunPod
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import configuration
from runpod_config import RunPodRTX4090Config

# Setup environment before importing modules
RunPodRTX4090Config.setup_rtx4090_environment()

# Import main app
from main import app

class RTX4090OptimizedServer:
    """RTX 4090 optimized server for RunPod"""
    
    def __init__(self):
        self.config = RunPodRTX4090Config
        self.setup_server_optimizations()
    
    def setup_server_optimizations(self):
        """Setup server optimizations for RTX 4090"""
        
        # Memory optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(self.config.GPU_MEMORY_FRACTION)
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Setup async optimizations
        if sys.version_info >= (3, 8):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            except AttributeError:
                pass
    
    def get_server_info(self):
        """Get server information"""
        device_info = self.config.get_device_info()
        
        return {
            "server": "Audio Scam Detection API",
            "version": "1.0.0",
            "gpu_optimized": True,
            "device": device_info,
            "public_url": self.config.get_public_url(),
            "workspace": str(self.config.WORKSPACE_DIR),
            "features": {
                "mixed_precision": self.config.MIXED_PRECISION,
                "batch_processing": True,
                "model_compilation": True,
                "memory_optimization": True
            }
        }
    
    def print_startup_info(self):
        """Print startup information"""
        info = self.get_server_info()
        
        print("\n" + "="*60)
        print("🚀 AUDIO SCAM DETECTION SERVER - RTX 4090 OPTIMIZED")
        print("="*60)
        print(f"📡 Public URL: {info['public_url']}")
        print(f"🎮 GPU: {info['device']['name']}")
        print(f"💾 GPU Memory: {info['device']['total_memory'] / 1e9:.1f}GB")
        print(f"⚡ Mixed Precision: {info['features']['mixed_precision']}")
        print(f"🔧 Model Compilation: {info['features']['model_compilation']}")
        print(f"📊 Max Concurrent: {self.config.MAX_CONCURRENT_REQUESTS}")
        print(f"💾 Workspace: {info['workspace']}")
        print("="*60)
        print("🎯 Ready for audio analysis!")
        print("="*60 + "\n")
    
    def run(self):
        """Run the optimized server"""
        self.print_startup_info()
        
        # Run with RTX 4090 optimized settings
        uvicorn.run(
            app,
            host=self.config.HOST,
            port=self.config.PORT,
            workers=self.config.MAX_WORKERS,
            log_level="info",
            access_log=True,
            timeout_keep_alive=30,
            limit_max_requests=1000,
            limit_concurrency=self.config.MAX_CONCURRENT_REQUESTS,
            loop="asyncio"
        )

# Add RTX 4090 specific endpoints to main app
@app.get("/rtx4090-status")
async def rtx4090_status():
    """RTX 4090 specific status endpoint"""
    config = RunPodRTX4090Config
    device_info = config.get_device_info()
    
    # GPU memory information
    if torch.cuda.is_available():
        memory_info = {
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "allocated_memory": torch.cuda.memory_allocated(0),
            "cached_memory": torch.cuda.memory_reserved(0),
            "free_memory": torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
        }
    else:
        memory_info = {"error": "CUDA not available"}
    
    return {
        "gpu": device_info,
        "memory": memory_info,
        "optimizations": {
            "mixed_precision": config.MIXED_PRECISION,
            "gpu_memory_fraction": config.GPU_MEMORY_FRACTION,
            "max_batch_size": config.MAX_BATCH_SIZE,
            "cuda_malloc": config.CUDA_MEMORY_POOL
        },
        "models": {
            "whisper": config.WHISPER_MODEL,
            "emotion": config.EMOTION_MODEL,
            "ai_voice": config.AI_VOICE_MODEL,
            "diarization": config.DIARIZATION_MODEL,
            "llm": config.OLLAMA_MODEL
        }
    }

@app.get("/performance-test")
async def performance_test():
    """Test RTX 4090 performance"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # Performance benchmark
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    # Create test tensor
    test_tensor = torch.randn(1000, 1000, device='cuda')
    
    start_time.record()
    
    # Perform operations
    for _ in range(100):
        result = torch.matmul(test_tensor, test_tensor.T)
        result = torch.relu(result)
    
    end_time.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_time.elapsed_time(end_time)
    
    # Clean up
    del test_tensor, result
    torch.cuda.empty_cache()
    
    return {
        "test": "RTX 4090 Matrix Operations",
        "operations": "100 x (1000x1000 matmul + relu)",
        "elapsed_time_ms": elapsed_time,
        "performance": "excellent" if elapsed_time < 100 else "good" if elapsed_time < 200 else "slow"
    }

# Add startup event for RTX 4090 optimizations
@app.on_event("startup")
async def startup_event():
    """Initialize RTX 4090 optimizations"""
    logger.info("🚀 Starting RTX 4090 optimized server...")
    
    # Warmup GPU
    if torch.cuda.is_available():
        logger.info("🔥 Warming up RTX 4090...")
        warmup_tensor = torch.randn(100, 100, device='cuda')
        _ = torch.matmul(warmup_tensor, warmup_tensor.T)
        del warmup_tensor
        torch.cuda.empty_cache()
        logger.info("✅ RTX 4090 warmed up")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🧹 Cleaning up RTX 4090 resources...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    # Create and run optimized server
    server = RTX4090OptimizedServer()
    server.run()

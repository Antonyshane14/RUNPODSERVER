#!/usr/bin/env python3
"""
Test script to verify HuggingFace token and model access
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hf_token():
    """Test if HF_TOKEN is properly configured"""
    logger.info("🔍 Testing HuggingFace Token Configuration...")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("❌ HF_TOKEN environment variable not found")
        logger.error("💡 Set it with: export HF_TOKEN='your_token_here'")
        logger.error("🌐 Get token from: https://huggingface.co/settings/tokens")
        return False
    
    logger.info(f"✅ HF_TOKEN found: {hf_token[:10]}...")
    return True

def test_model_access():
    """Test if we can access the emotion model"""
    logger.info("🔍 Testing model access...")
    
    try:
        from transformers import AutoConfig, AutoProcessor
        
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        hf_token = os.getenv("HF_TOKEN")
        
        logger.info(f"📥 Testing access to: {model_name}")
        
        # Test config access
        logger.info("🔧 Loading model config...")
        config = AutoConfig.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            cache_dir="/tmp/test_models"
        )
        logger.info("✅ Config loaded successfully")
        
        # Test processor access
        logger.info("🔧 Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            cache_dir="/tmp/test_models",
            trust_remote_code=True
        )
        logger.info("✅ Processor loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model access failed: {e}")
        return False

def test_pytorch_cuda():
    """Test PyTorch and CUDA availability"""
    logger.info("🔍 Testing PyTorch and CUDA...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("⚠️  CUDA not available")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ PyTorch test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Audio Scam Detection System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("HuggingFace Token", test_hf_token),
        ("PyTorch & CUDA", test_pytorch_cuda),
        ("Model Access", test_model_access),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! System ready for deployment.")
    else:
        logger.error("\n💥 Some tests failed. Please fix issues before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()

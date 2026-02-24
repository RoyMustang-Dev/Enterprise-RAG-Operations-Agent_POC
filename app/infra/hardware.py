"""
Hardware Optimization & Diagnostic Service

This module is called on startup to actively map the local machine's architecture.
It ensures the Agentic backend doesn't trigger 50 concurrent crawler threads on a 2-core laptop, 
crashing the OS memory violently.
"""
import os
import logging
import multiprocessing

logger = logging.getLogger(__name__)

class HardwareProbe:
    """
    Examines the physical execution hardware (CPU vs CUDA vs Apple MPS).
    Determines logical scaling multipliers dynamically so the enterprise architecture 
    works natively whether deployed on a 16-Core AWS Server or a cheap student Laptop.
    """
    
    @staticmethod
    def detect_environment() -> dict:
        """
        Pipes the computational status of the Python Host environment.
        
        Returns:
            dict: The normalized mappings determining exact Queue worker limits and Inference targets.
        """
        try:
            # CPU Extraction
            cpu_cores = multiprocessing.cpu_count() or 4
            
            # Target `cpu_cores * 2 + 1` for I/O bound tasks like Crawler networking
            recommended_async_workers = (cpu_cores * 2) + 1
            
            # GPU Extraction (PyTorch hooks)
            has_gpu = False
            device = "cpu"
            gpu_name = "None"
            
            try:
                import torch
                if torch.cuda.is_available():
                    has_gpu = True
                    device = "cuda"
                    # Prevent crashes on older PyTorch versions by safely probing the property mapping
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown CUDA"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                     has_gpu = True
                     device = "mps"
                     gpu_name = "Apple Silicon"
            except ImportError:
                logger.warning("PyTorch not installed. Falling back strictly to CPU mapping.")
                
            config = {
                "cpu_cores": cpu_cores,
                "recommended_async_workers": recommended_async_workers,
                "has_gpu": has_gpu,
                "primary_device": device,
                "gpu_name": gpu_name
            }
            
            logger.info(f"[HARDWARE PROBE] Initialized scaling config: {config}")
            return config
            
        except Exception as e:
            logger.error(f"[HARDWARE PROBE] Catastrophic failure during environment scan: {e}")
            # Failsafe values guaranteeing system doesn't crash on failure
            return {
                "cpu_cores": 2,
                "recommended_async_workers": 4,
                "has_gpu": False,
                "primary_device": "cpu",
                "gpu_name": "None"
            }

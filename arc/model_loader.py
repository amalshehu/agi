"""
Trained Model Loader
Loads trained AGI models for ARC competition
"""

import torch
from pathlib import Path
from typing import Optional
import json
import glob

# Import AGI components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.hybrid_agi import HybridAGI


class TrainedModelLoader:
    """Loads trained ARC models"""
    
    def __init__(self, model_dir: str = "trained_models"):
        self.model_dir = Path(model_dir)
        
    def get_latest_model(self) -> Optional[str]:
        """Get the latest trained model"""
        if not self.model_dir.exists():
            return None
        
        # Find latest neural recognizer model
        neural_models = list(self.model_dir.glob("neural_recognizer_*.pth"))
        if not neural_models:
            return None
        
        # Sort by timestamp and get latest
        latest_neural = sorted(neural_models)[-1]
        timestamp = latest_neural.stem.split("_")[-1]
        
        return timestamp
    
    def load_trained_agi(self, timestamp: Optional[str] = None) -> HybridAGI:
        """Load AGI with trained weights"""
        if timestamp is None:
            timestamp = self.get_latest_model()
        
        if timestamp is None:
            print("⚠️ No trained models found. Using untrained AGI.")
            return HybridAGI("arc_competitor")
        
        print(f"🔄 Loading trained model: {timestamp}")
        
        # Create AGI instance
        agi = HybridAGI("arc_trained")
        
        # Load neural recognizer weights
        neural_path = self.model_dir / f"neural_recognizer_{timestamp}.pth"
        if neural_path.exists():
            agi.neural_recognizer.load_state_dict(torch.load(neural_path))
            print(f"✅ Loaded neural recognizer: {neural_path.name}")
        
        # Load causal model weights
        causal_path = self.model_dir / f"causal_model_{timestamp}.pth"
        if causal_path.exists():
            agi.causal_model.load_state_dict(torch.load(causal_path))
            print(f"✅ Loaded causal model: {causal_path.name}")
        
        # Load training history
        history_path = self.model_dir / f"training_history_{timestamp}.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                training_history = json.load(f)
            print(f"✅ Training history loaded:")
            if training_history.get('accuracies'):
                final_acc = training_history['accuracies'][-1]
                print(f"   Final accuracy: {final_acc:.3f}")
            if training_history.get('consciousness_levels'):
                avg_consciousness = sum(training_history['consciousness_levels']) / len(training_history['consciousness_levels'])
                print(f"   Avg consciousness: {avg_consciousness:.2f}")
        
        return agi
    
    def list_available_models(self):
        """List all available trained models"""
        if not self.model_dir.exists():
            print("No trained models directory found")
            return
        
        neural_models = list(self.model_dir.glob("neural_recognizer_*.pth"))
        if not neural_models:
            print("No trained models found")
            return
        
        print("📋 Available trained models:")
        for model_path in sorted(neural_models):
            timestamp = model_path.stem.split("_")[-1]
            
            # Check for corresponding files
            causal_exists = (self.model_dir / f"causal_model_{timestamp}.pth").exists()
            history_exists = (self.model_dir / f"training_history_{timestamp}.json").exists()
            
            status = "✅ Complete" if causal_exists and history_exists else "⚠️ Partial"
            print(f"  {timestamp}: {status}")


def load_best_trained_agi() -> HybridAGI:
    """Convenience function to load the best trained AGI"""
    loader = TrainedModelLoader()
    return loader.load_trained_agi()


if __name__ == "__main__":
    loader = TrainedModelLoader()
    loader.list_available_models()
    
    # Test loading
    agi = loader.load_trained_agi()
    print(f"\n🧠 Loaded AGI: {agi.model_name}")

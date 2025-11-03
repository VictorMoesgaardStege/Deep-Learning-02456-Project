"""
Weights & Biases (WandB) integration for experiment tracking
"""

import numpy as np
from typing import Optional, Dict
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")


class WandBLogger:
    """
    Logger for tracking experiments with Weights & Biases
    """
    
    def __init__(self, 
                 project_name: str = "neural-network-from-scratch",
                 config: Optional[Dict] = None,
                 enabled: bool = True):
        """
        Initialize WandB logger
        
        Args:
            project_name: Name of the WandB project
            config: Configuration dictionary to log
            enabled: Whether to enable WandB logging
        """
        self.enabled = enabled and WANDB_AVAILABLE
        
        if self.enabled:
            wandb.init(project=project_name, config=config)
            self.run = wandb.run
            print(f"WandB initialized. Run: {self.run.name}")
        else:
            self.run = None
            if enabled and not WANDB_AVAILABLE:
                print("WandB logging disabled: wandb package not available")
            else:
                print("WandB logging disabled")
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None) -> None:
        """
        Log metrics to WandB
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (epoch number, iteration, etc.)
        """
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def log_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            class_names: Optional[list] = None) -> None:
        """
        Log confusion matrix to WandB
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        if self.enabled:
            try:
                # WandB v0.13.0+ API (as specified in requirements.txt)
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=class_names
                    )
                })
            except (TypeError, AttributeError):
                # Fallback for API changes in future WandB versions
                # Note: scikit-learn is required (listed in requirements.txt)
                try:
                    from sklearn.metrics import confusion_matrix as sklearn_cm
                    cm = sklearn_cm(y_true, y_pred)
                    wandb.log({"confusion_matrix_data": wandb.Table(
                        data=cm.tolist(),
                        columns=class_names if class_names else [f"Class {i}" for i in range(cm.shape[1])]
                    )})
                except Exception as e:
                    print(f"Warning: Could not log confusion matrix to WandB: {e}")
    
    def log_image(self, name: str, image_path: str) -> None:
        """
        Log image to WandB
        
        Args:
            name: Name of the image
            image_path: Path to the image file
        """
        if self.enabled:
            wandb.log({name: wandb.Image(image_path)})
    
    def log_model_architecture(self, model_summary: str) -> None:
        """
        Log model architecture summary
        
        Args:
            model_summary: String description of model architecture
        """
        if self.enabled:
            wandb.config.update({"model_architecture": model_summary})
    
    def watch_gradients(self, model) -> None:
        """
        Watch model gradients (if applicable)
        
        Args:
            model: Neural network model
        """
        # For our NumPy implementation, we can log gradient norms manually
        pass
    
    def finish(self) -> None:
        """Finish the WandB run"""
        if self.enabled:
            wandb.finish()
            print("WandB run finished")

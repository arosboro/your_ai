"""Checkpoint state container."""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import mlx.core as mx
from src.config import Config


@dataclass
class Checkpoint:
    """
    Complete training state snapshot.
    
    Contains everything needed to resume training from a specific step.
    """
    step: int
    model_state: Dict[str, mx.array]
    optimizer_state: Dict[str, Any]
    loss_history: List[float]
    config: Config
    random_state: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate checkpoint data."""
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        if not isinstance(self.model_state, dict):
            raise ValueError("model_state must be a dictionary")
        if not isinstance(self.optimizer_state, dict):
            raise ValueError("optimizer_state must be a dictionary")

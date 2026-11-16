from pydantic import BaseModel, Field, field_validator
from typing import Literal

class DLConfig(BaseModel):
    data_root: str = Field(..., description="путь к CIFAR-100")
    subset_per_class: int = Field(100, gt=0, le=500)
    mode: Literal["full", "stream"] = "full"
    batch_size: int = Field(128, gt=0)
    epochs: int = Field(10, gt=0)
    seed: int = Field(42, ge=0)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    early_stop_patience: int = Field(3, gt=0)
    num_workers: int = Field(0, ge=0)
    device: Literal["cpu", "cuda"] = "cpu"

    @field_validator("batch_size")
    @classmethod
    def batch_must_divide(cls, v, info):
        if "subset_per_class" in info.data:
            total = info.data["subset_per_class"] * 100
            if total % v != 0:
                raise ValueError(f"batch_size {v} must divide total {total}")
        return v

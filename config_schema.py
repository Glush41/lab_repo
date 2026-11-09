from pydantic import BaseModel, Field, field_validator
from typing import Literal

class PipelineConfig(BaseModel):
    data_root: str = Field(..., description="путь к CIFAR-100")
    subset_per_class: int = Field(100, gt=0, le=500)
    chunk_size: int = Field(1000, gt=0)
    mode: Literal["full", "chunk"] = "full"
    seed: int = Field(42, ge=0)
    model_type: Literal["sgd", "rf"] = "sgd"
    sgd_alpha: float = Field(1e-4, gt=0)
    rf_n_estimators: int = Field(100, gt=0)
    test_size: float = Field(0.2, gt=0, lt=1)

    @field_validator("chunk_size")
    @classmethod
    def chunk_must_divide(cls, v, info):
        if "subset_per_class" in info.data:
            total = info.data["subset_per_class"] * 100
            if total % v != 0:
                raise ValueError(f"chunk_size {v} must divide total {total}")
        return v

from config_schema import PipelineConfig
from pydantic import ValidationError

cases = [
    {"data_root": "./data", "chunk_size": 333, "mode": "full"},
    {"data_root": "./data", "test_size": 1.5, "mode": "full"},
    {"data_root": "./data", "mode": "invalid"},
]
for i, case in enumerate(cases):
    try:
        PipelineConfig(**case)
        print(f"case {i}: NO ERROR (bad)")
    except ValidationError as e:
        print(f"case {i}: rejected — {e.errors()[0]['msg']}")

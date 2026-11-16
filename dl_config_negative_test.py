from dl_config_schema import DLConfig
from pydantic import ValidationError

cases = [
    {"data_root": "./data", "batch_size": 333, "mode": "full"},
    {"data_root": "./data", "epochs": 0, "mode": "full"},
    {"data_root": "./data", "mode": "invalid"},
]
for i, case in enumerate(cases):
    try:
        DLConfig(**case)
        print(f"case {i}: NO ERROR (bad)")
    except ValidationError as e:
        print(f"case {i}: rejected — {e.errors()[0]['msg']}")

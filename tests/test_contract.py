from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_contract_module():
    path = Path(__file__).resolve().parents[1] / "docker" / "validate_contract.py"
    spec = importlib.util.spec_from_file_location("validate_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ServerArgs:
    def __init__(
        self,
        model_path: str,
        fake_num_kv_blocks: int = 15_509,
        page_size: int = 256,
        context_length: int = 1_048_576,
        kv_cache_dtype: str = "fp8_e4m3",
        max_running_requests: int = 32,
    ) -> None:
        self.model_path = model_path
        self.fake_num_kv_blocks = fake_num_kv_blocks
        self.page_size = page_size
        self.context_length = context_length
        self.kv_cache_dtype = kv_cache_dtype
        self.max_running_requests = max_running_requests


class ContractTests(unittest.TestCase):
    def test_fake_sglang_effective_profile(self) -> None:
        sglang = types.ModuleType("sglang")
        sglang.__is_fake_sglang__ = True
        srt = types.ModuleType("sglang.srt")
        server_args = types.ModuleType("sglang.srt.server_args")
        server_args.ServerArgs = ServerArgs
        modules = {
            "sglang": sglang,
            "sglang.srt": srt,
            "sglang.srt.server_args": server_args,
        }
        contract = load_contract_module()
        with patch.dict(sys.modules, modules):
            result = contract.validate()
        self.assertTrue(result["profile_inherited"])
        self.assertEqual(result["profile"]["token_slots_per_rank"], 3_970_304)


if __name__ == "__main__":
    unittest.main()

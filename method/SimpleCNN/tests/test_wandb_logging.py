"""W&B 直连传输配置的回归测试。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from utils.logging import _WANDB_PROXY_ENV_NAMES, _configure_direct_wandb_transport


class DirectWandbTransportTests(unittest.TestCase):
    def test_clears_inherited_proxy_and_forces_direct_connection(self) -> None:
        environment = {
            "HTTP_PROXY": "http://127.0.0.1:47890",
            "HTTPS_PROXY": "http://127.0.0.1:47890",
            "ALL_PROXY": "socks5://127.0.0.1:47890",
            "http_proxy": "http://127.0.0.1:47890",
            "https_proxy": "http://127.0.0.1:47890",
            "all_proxy": "socks5://127.0.0.1:47890",
            "NO_PROXY": "internal.example",
            "no_proxy": "internal.example",
            "WANDB__PROXIES": '{"https": "http://127.0.0.1:47890"}',
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            _configure_direct_wandb_transport()
            self.assertTrue(all(name not in os.environ for name in _WANDB_PROXY_ENV_NAMES))
            self.assertEqual(os.environ["NO_PROXY"], "*")
            self.assertEqual(os.environ["no_proxy"], "*")
            self.assertNotIn("WANDB__PROXIES", os.environ)


if __name__ == "__main__":
    unittest.main()

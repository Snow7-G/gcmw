"""#87 复核回归：演示服务的库层边界——host 只允许 loopback。

demo_showcase.py 位于 scripts/（非 app 包），通过 sys.path 直接导入其
纯校验逻辑；不启动任何服务器。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from demo_showcase import _LOOPBACK_HOSTS, start_server


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "10.0.0.5", "example.com"])
def test_start_server_rejects_non_loopback_hosts(host):
    """直接调用（绕过 CLI）也必须在库层拒绝：固定演示凭据不得离开本机。"""
    with pytest.raises(ValueError, match="loopback"):
        start_server(host=host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_the_only_allowed_set(host):
    assert host in _LOOPBACK_HOSTS

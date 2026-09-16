"""#87 复核回归：演示服务的库层边界——host 只允许 loopback。

demo_showcase.py 位于 scripts/（非 app 包），通过 sys.path 直接导入其
纯校验逻辑；不启动任何服务器。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from demo_showcase import start_server


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.10", "10.0.0.5", "example.com", "localhost", "::1"],
)
def test_start_server_rejects_any_host_but_127_0_0_1(host):
    """直接调用（绕过 CLI）也必须在库层拒绝：唯一合法地址是 127.0.0.1。"""
    with pytest.raises(ValueError, match="exactly '127.0.0.1'"):
        start_server(host=host)

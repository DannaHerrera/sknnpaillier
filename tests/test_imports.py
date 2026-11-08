"""Import the library independently of pytest's already loaded modules."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("file_logging", ["0", "1"])
def test_import_without_side_effects(tmp_path, file_logging):
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("SKNN_LOG_"):
            del environment[name]
    environment.update({
        "SKNN_LOG_TO_FILE": file_logging,
        "SKNN_LOG_ERROR_FILE": file_logging,
        "SKNN_LOG_PATH": str(tmp_path / "logs"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    code = '''
import logging
from unittest.mock import patch

with patch("secrets.randbits", side_effect=AssertionError("key generation")), \\
     patch.object(logging.Logger, "addHandler", side_effect=AssertionError("handler")):
    import sknnpaillier
    import sknnpaillier.cloud_b
    import sknnpaillier.kd_tree
    import sknnpaillier.key_m
    import sknnpaillier.secure_data_t
    import sknnpaillier.log
    for name in sknnpaillier.__all__:
        assert getattr(sknnpaillier, name) is not None
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []

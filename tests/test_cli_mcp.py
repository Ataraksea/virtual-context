"""Tests for the `virtual-context mcp` CLI command."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest


def test_cli_mcp_invokes_serve():
    """Verify that 'virtual-context mcp' correctly calls server.serve()."""
    with patch("virtual_context.mcp.server.serve") as mock_serve:
        from virtual_context.cli.main import main
        with patch("sys.argv", ["virtual-context", "mcp"]):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0
        mock_serve.assert_called_once()


def test_cli_mcp_config_propagation():
    """Verify that 'virtual-context --config <path> mcp' sets the config env var."""
    config_path = "/path/to/virtual-context-test.yaml"
    with patch("virtual_context.mcp.server.serve") as mock_serve, \
         patch.dict(os.environ, {}):
        from virtual_context.cli.main import main
        with patch("sys.argv", ["virtual-context", "--config", config_path, "mcp"]):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0
        mock_serve.assert_called_once()
        assert os.environ.get("VIRTUAL_CONTEXT_CONFIG") == os.path.abspath(config_path)

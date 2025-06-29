"""Tests for the main module."""
from excpres.main import hello_world


def test_hello_world(capsys):
    """Test the hello_world function."""
    hello_world()
    captured = capsys.readouterr()
    assert "Hello, World! Welcome to Excpres!" in captured.out

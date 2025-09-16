"""Example test module."""

import pytest


def test_example_passes():
    """Example test that always passes."""
    assert True


def test_addition():
    """Test basic addition."""
    assert 1 + 1 == 2


def test_list_operations():
    """Test list operations."""
    my_list = [1, 2, 3]
    my_list.append(4)
    assert len(my_list) == 4
    assert my_list[-1] == 4


@pytest.mark.parametrize("input_value,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
])
def test_multiplication(input_value, expected):
    """Test multiplication with parametrization."""
    assert input_value * 2 == expected
"""
Unit tests for core variable types.

Tests variable classes used for type-safe storage of tracking data.
"""

import unittest
from typing import ClassVar
from unittest.mock import Mock

import pytest

from ethoscope.core.variables import (
    BaseBoolVariable,
    BaseIntVariable,
    BaseRelativeVariable,
    HeightVariable,
    IsInferredVariable,
    Label,
    PhiVariable,
    SQLDataType,
    WidthVariable,
    XPosVariable,
    YPosVariable,
    mLogLik,
)


class TestVariableTypes(unittest.TestCase):
    """Test suite for variable type classes."""

    def test_is_inferred_variable_creation(self):
        """Test IsInferredVariable can be created with valid values."""
        var = IsInferredVariable(1)
        assert int(var) == 1
        assert var.header_name == "is_inferred"
        assert var.functional_type == "bool"

    def test_phi_variable_creation(self):
        """Test PhiVariable can be created."""
        expected = 180
        var = PhiVariable(expected)
        assert int(var) == expected
        assert var.header_name == "phi"
        assert var.functional_type == "angle"

    def test_label_variable_creation(self):
        """Test Label variable can be created."""
        expected = 5
        var = Label(expected)
        assert int(var) == expected
        assert var.functional_type == "label"

    def test_width_height_variables(self):
        """Test width and height variables."""
        expected_width = 50
        expected_height = 30
        width = WidthVariable(expected_width)
        height = HeightVariable(expected_height)

        assert int(width) == expected_width
        assert int(height) == expected_height
        assert width.header_name == "w"
        assert height.header_name == "h"
        assert width.functional_type == "distance"

    def test_mloglik_variable(self):
        """Test mLogLik variable for probability storage."""
        expected = 1000
        var = mLogLik(expected)
        assert int(var) == expected
        assert var.functional_type == "proba"


class TestVariableValidation(unittest.TestCase):
    """Test class-definition-time validation of variable definitions."""

    def test_missing_functional_type_raises_at_definition(self):
        """Defining a variable without functional_type raises at class definition."""
        with pytest.raises(NotImplementedError):
            _ = type(
                "BadVariable",
                (BaseIntVariable,),
                {"sql_data_type": SQLDataType.INT, "header_name": "test"},
            )

    def test_missing_sql_data_type_raises_at_definition(self):
        """Defining a variable without a sql_data_type raises at class definition."""
        with pytest.raises(NotImplementedError):
            _ = type(
                "BadVariable",
                (BaseIntVariable,),
                {
                    "functional_type": "test",
                    "header_name": "test",
                    "sql_data_type": None,
                },
            )

    def test_missing_header_name_raises_at_definition(self):
        """Defining a variable without header_name raises at class definition."""
        with pytest.raises(NotImplementedError):
            _ = type(
                "BadVariable",
                (BaseIntVariable,),
                {"functional_type": "test", "sql_data_type": SQLDataType.INT},
            )

    def test_abstract_class_definition_skips_validation(self):
        """Classes explicitly marked abstract skip validation."""
        class AbstractTemplate(BaseIntVariable):
            _abstract: ClassVar[bool] = True

        assert AbstractTemplate.__dict__["_abstract"] is True

    def test_concrete_subclass_of_abstract_template_is_validated(self):
        """Concrete subclasses of abstract templates are validated."""
        with pytest.raises(NotImplementedError):
            _ = type("BadVariable", (BaseBoolVariable,), {})

    def test_abstract_class_cannot_be_instantiated(self):
        """Instantiating an abstract class raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _ = BaseBoolVariable(1)


class TestRelativeVariables(unittest.TestCase):
    """Test suite for relative position variables."""

    def test_x_pos_variable_to_absolute(self):
        """Test XPosVariable converts to absolute coordinates (lines 140-144)."""
        # Create mock ROI with offset
        x_offset = 100
        y_offset = 50
        mock_roi = Mock()
        mock_roi.offset = (x_offset, y_offset)

        # Create relative X position
        x_rel = XPosVariable(25)

        # Convert to absolute
        x_abs = x_rel.to_absolute(mock_roi)

        # Should be 25 + 100 = 125
        assert isinstance(x_abs, XPosVariable)
        assert int(x_abs) == 25 + x_offset

    def test_y_pos_variable_to_absolute(self):
        """Test YPosVariable converts to absolute coordinates (lines 155-158)."""
        # Create mock ROI with offset
        x_offset = 100
        y_offset = 50
        mock_roi = Mock()
        mock_roi.offset = (x_offset, y_offset)

        # Create relative Y position
        y_rel = YPosVariable(30)

        # Convert to absolute
        y_abs = y_rel.to_absolute(mock_roi)

        # Should be 30 + 50 = 80
        assert isinstance(y_abs, YPosVariable)
        assert int(y_abs) == 30 + y_offset

    def test_to_absolute_calls_get_absolute_value(self):
        """Test to_absolute method calls _get_absolute_value (line 124)."""
        mock_roi = Mock()
        mock_roi.offset = (10, 20)

        x_var = XPosVariable(5)
        result = x_var.to_absolute(mock_roi)

        # Result should be from _get_absolute_value
        assert int(result) == 5 + 10

    def test_base_relative_variable_not_implemented(self):
        """Test BaseRelativeVariable requires _get_absolute_value implementation."""
        class IncompleteVariable(BaseRelativeVariable):
            header_name = "test"
            # Missing _get_absolute_value implementation

        var = IncompleteVariable(10)
        mock_roi = Mock()

        with pytest.raises(NotImplementedError):
            var.to_absolute(mock_roi)


if __name__ == "__main__":
    unittest.main()

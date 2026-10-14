# tests/dataset/test_data_processor.py
import pytest

pytestmark = pytest.mark.cpu_test


class TestDataProcessorImport:
    def test_data_processor_importable(self):
        from psrl.utils.dataset.data_processor import DataProcessor

        assert DataProcessor is not None

    def test_dataset_utils_importable(self):
        from psrl.utils.dataset import utils

        assert utils is not None

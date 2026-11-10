import logging
from uuid import uuid4

import pytest
from psrl.utils.logger import DualOutputHandler, FileOnlyHandler

pytestmark = pytest.mark.cpu_test


def _attach_logger(handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(f"psrl.test.ray_logger.{uuid4().hex}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return logger


def test_file_only_handler_keeps_writing_after_close(tmp_path):
    handler = FileOnlyHandler(str(tmp_path), "FileOnly")
    logger = _attach_logger(handler)
    try:
        logger.info("Before close.")
        handler.close()
        logger.info("After close.")
        text = (tmp_path / "FileOnly.log").read_text()
        assert "Before close." in text
        assert "After close." in text
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_dual_output_handler_keeps_writing_after_close(tmp_path, capsys):
    handler = DualOutputHandler(str(tmp_path), "Dual")
    logger = _attach_logger(handler)
    try:
        logger.info("Before close.")
        handler.close()
        logger.info("After close.")
        text = (tmp_path / "Dual.log").read_text()
        assert "Before close." in text
        assert "After close." in text
        stderr = capsys.readouterr().err
        assert "<Dual> - Before close." in stderr
        assert "<Dual> - After close." in stderr
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_file_only_handler_truncates_on_construct(tmp_path):
    first = FileOnlyHandler(str(tmp_path), "Truncate")
    logger = _attach_logger(first)
    try:
        logger.info("Stale record.")
    finally:
        logger.removeHandler(first)
        first.close()

    second = FileOnlyHandler(str(tmp_path), "Truncate")
    logger = _attach_logger(second)
    try:
        logger.info("Fresh record.")
        text = (tmp_path / "Truncate.log").read_text()
        assert "Stale record." not in text
        assert "Fresh record." in text
    finally:
        logger.removeHandler(second)
        second.close()

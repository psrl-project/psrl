"""
XML Function-Calling Model for standalone eval.

Extends ``LitellmTextbasedModel`` to handle SWE-agent-LM's XML function calling
format (``<function=NAME>...</function>``) by translating tool calls into bash
commands via ``psrl.tools.tool_parser.xml_fc_tool_parser.parse_xml_fc_to_bash``.
"""

from __future__ import annotations

import logging
import os

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_textbased_model import (
    LitellmTextbasedModel,
    LitellmTextbasedModelConfig,
)
from psrl.utils.agent.overflow import PromptOverflowError, handle_prompt_overflow  # noqa: F401

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class XmlFcModelConfig(LitellmTextbasedModelConfig):
    """
    Configure XML function call defaults.
    """

    action_regex: str = "__XML_FUNCTION_CALLING__"
    format_error_template: str = (
        "Your last output did not include a valid function call. "
        "Please make sure your output includes exactly ONE function call.\n"
        "If you think you have already resolved the issue, please submit your changes "
        "by calling the `submit` function.\n"
        "If you think you cannot solve the problem, please call `submit`.\n"
        "Otherwise, please continue with a new tool call using the correct format:\n\n"
        "<function=function_name>\n"
        "<parameter=parameter_name>value</parameter>\n"
        "</function>"
    )


@handle_prompt_overflow
class XmlFcModel(LitellmTextbasedModel):
    """
    LitellmTextbasedModel subclass that parses SWE-agent-LM XML function calls.

    Instead of regex-extracting a bash code block, it uses
    ``parse_xml_fc_to_bash()`` to translate ``<function=bash>``,
    ``<function=str_replace_editor>``, and ``<function=submit>`` calls into
    executable bash commands.
    """

    def __init__(self, **kwargs):
        """Initialize with XmlFcModelConfig."""
        super(LitellmTextbasedModel, self).__init__(config_class=XmlFcModelConfig, **kwargs)

    def _parse_actions(self, response) -> list[dict]:
        """
        Parse XML function calls into executable shell commands.

        Raises:
            FormatError: When no valid action is found.
        """
        from psrl.tools.tool_parser.xml_fc_tool_parser import parse_xml_fc_to_bash

        content = response.choices[0].message.content or ""

        cmd = parse_xml_fc_to_bash(content)
        if cmd is not None:
            return [{"command": cmd}]

        # Raise `FormatError` when the parser finds no action.
        raise FormatError(
            {
                "role": "user",
                "content": self.config.format_error_template,
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": 0,
                    "model_response": content,
                },
            }
        )

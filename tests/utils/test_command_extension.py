import asyncio

import pytest
from psrl.utils.server.command import Command, CommandExtension, CommandType

pytestmark = pytest.mark.cpu_test


def test_timed_out_command_remains_synchronizable_without_requeue():
    async def scenario():
        extension = CommandExtension()
        command = Command(type=CommandType.WAKE_UP, instance_ids=[1])

        pending = await extension.exec_command(command, timeout=0.001, blocking=True)

        assert pending == {"status": "QUEUED", "command_id": 0}
        assert extension.command_queue.qsize() == 1
        queued = extension.command_queue.get_nowait()
        extension._start_command(0)
        extension._complete_command(0, True)

        assert queued is command
        assert extension.command_queue.empty()
        assert await extension.synchronize_command(0, timeout=0.1) is True

    asyncio.run(scenario())

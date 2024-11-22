import pytest
import asyncio
import aiohttp
from typing import List


@pytest.fixture(autouse=True)
async def cleanup_sessions():
    """Cleanup any remaining sessions after each test."""
    sessions: List[aiohttp.ClientSession] = []

    try:
        yield
        # Allow pending tasks to complete
        await asyncio.sleep(0.2)  # Increased sleep time

        # Get all tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        # Find any aiohttp sessions in the tasks
        for task in tasks:
            if hasattr(task, "get_coro"):
                coro = task.get_coro()
                if hasattr(coro, "cr_frame"):
                    frame = coro.cr_frame
                    if frame and "self" in frame.f_locals:
                        obj = frame.f_locals["self"]
                        if isinstance(obj, aiohttp.ClientSession):
                            sessions.append(obj)

        # Cancel tasks
        for task in tasks:
            task.cancel()

        # Wait for cancellation
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Close sessions
        for session in sessions:
            if not session.closed:
                await session.close()

    except Exception as e:
        print(f"Error in cleanup: {e}")
        raise
    finally:
        # Force close any remaining sessions
        for session in sessions:
            if not session.closed:
                await session.close()

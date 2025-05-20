"""Configuration utilities for Runnables."""

import asyncio
from concurrent.futures import Executor
from contextvars import copy_context
from functools import partial
from typing import Callable, Optional, TypeVar, cast, Any

T = TypeVar("T")


async def run_in_executor(
    executor: Optional[Executor],
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a function in an executor.

    Args:
        executor: The executor to run in.
        func (Callable[P, T]): The function.
        *args (Any): The positional arguments to the function.
        **kwargs (Any): The keyword arguments to the function.

    Returns:
        T: The output of the function.

    Raises:
        RuntimeError: If the function raises a StopIteration.
    """

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration can't be set on an asyncio.Future
            # it raises a TypeError and leaves the Future pending forever
            # so we need to convert it to a RuntimeError
            raise RuntimeError from exc

    if executor is None:
        # Use default executor with context copied from current context
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast("Callable[..., T]", partial(copy_context().run, wrapper)),
        )

    return await asyncio.get_running_loop().run_in_executor(executor, wrapper)

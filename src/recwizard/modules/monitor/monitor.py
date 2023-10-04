import json
import time
import logging
from contextlib import contextmanager
logger = logging.getLogger(__name__)

@contextmanager
def monitoring(mode="info"):
    r"""Monitoring context manager that takes care of controlling the monitor's internal state
    like resetting the call chain graphs.

        Example:
            >>> with monitoring(mode="debug") as m:
                    output = model.response(input)
                    print(m.graph)

        Args:
            mode: The logging mode can be info/debug. Monitor is activated in debug mode (default: :str:`info`)
        
        Returns:
            RecwizardMonitor
    """
    monitor = RecwizardMonitor()
    if mode == "debug":
        monitor.activate()
    try:
        yield monitor
    finally:
        # Restore original states
        monitor.deactivate()
        monitor.reset()


def serializable(x, max_size=3000):
    try:
        obj = json.dumps(x)
        if len(obj) > max_size:
            return False
        return True
    except (TypeError, OverflowError):
        return False

class RecwizardMonitor:
    r"""RecwizardMonitor static class to wrap module/model response method for debugging purposes.
        The intermediate outputs can then be accessed using the `monitoring` context
        Example:

            @monitor
            def response(input, **kwargs):
                ...

    """
    graph = []
    active = False

    @classmethod
    def monitor(cls, function):
        def function_wrapper(*args, **kwargs):
            if not cls.active:
                return function(*args, **kwargs)
            logger.info(f"Called {function}")
            start_time = time.time()
            result = function(*args, **kwargs)
            end_time = time.time()
            output = {k: v for k, v in result.items() if serializable(v)} if isinstance(result, dict) else result
            if not serializable(output):
                output = str(output)
            cls.graph.append({
                "node": function.__qualname__,
                "method": function.__name__,
                "input": [arg for arg in args[1:] if serializable(arg)],
                "kwargs": {k: v for k, v in kwargs.items() if serializable(v)},
                "output": output,
                "start_time": start_time,
                "end_time": end_time,
                "time": end_time - start_time
            })
            return result

        function_wrapper.__doc__ = function.__doc__
        return function_wrapper

    @classmethod
    def activate(cls):
        cls.active = True

    @classmethod
    def deactivate(cls):
        cls.active = False

    @classmethod
    def reset(cls):
        cls.graph = []



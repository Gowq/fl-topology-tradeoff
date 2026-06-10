"""
Timeout utilities for Grid experiments.

Provides timeout mechanism to prevent experiments from hanging indefinitely.
Uses signal.alarm() for Unix/Linux systems (compatible with GridUNESP).
"""

import signal
import functools
import time
from typing import Callable, Any, Optional


class TimeoutError(Exception):
    """Raised when a function execution exceeds the timeout limit."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout events."""
    raise TimeoutError("Function execution exceeded timeout limit")


def with_timeout(seconds: int = 7200):
    """
    Decorator to add timeout protection to a function.
    
    Args:
        seconds: Timeout duration in seconds (default: 7200 = 2 hours)
    
    Returns:
        Decorated function that raises TimeoutError if execution exceeds timeout
    
    Example:
        @with_timeout(3600)  # 1 hour timeout
        def long_running_experiment():
            # ... experiment code ...
            pass
    
    Note:
        - Only works on Unix/Linux systems (uses signal.alarm)
        - Not compatible with Windows
        - Cannot be nested (signal.alarm limitation)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Set up the timeout handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm and restore old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    return decorator


def run_with_timeout(func: Callable, timeout_seconds: int = 7200, 
                     on_timeout: Optional[Callable] = None, 
                     *args, **kwargs) -> tuple[bool, Any]:
    """
    Run a function with timeout protection and return success status.
    
    Args:
        func: Function to execute
        timeout_seconds: Timeout duration in seconds (default: 7200 = 2 hours)
        on_timeout: Optional callback function to execute on timeout
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Tuple of (success: bool, result: Any)
        - If successful: (True, function_result)
        - If timeout: (False, None)
    
    Example:
        success, result = run_with_timeout(
            train_model, 
            timeout_seconds=3600,
            on_timeout=lambda: print("Training timed out!"),
            model=model, 
            data=data
        )
        
        if success:
            print(f"Training completed: {result}")
        else:
            print("Training timed out, skipping...")
    """
    # Set up the timeout handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel alarm on success
        signal.signal(signal.SIGALRM, old_handler)
        return True, result
        
    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
        
        if on_timeout:
            on_timeout()
        
        return False, None
    
    except Exception as e:
        # Cancel alarm on any other exception
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        raise e


def format_timeout_duration(seconds: int) -> str:
    """
    Format timeout duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "2h 30m", "45m", "90s")
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("Testing timeout utilities...")
    
    # Test 1: Function that completes within timeout
    @with_timeout(5)
    def fast_function():
        time.sleep(1)
        return "Success!"
    
    try:
        result = fast_function()
        print(f"✅ Test 1 passed: {result}")
    except TimeoutError:
        print("❌ Test 1 failed: Unexpected timeout")
        sys.exit(1)
    
    # Test 2: Function that exceeds timeout
    @with_timeout(2)
    def slow_function():
        time.sleep(5)
        return "Should not reach here"
    
    try:
        result = slow_function()
        print("❌ Test 2 failed: Should have timed out")
        sys.exit(1)
    except TimeoutError:
        print("✅ Test 2 passed: Timeout triggered correctly")
    
    # Test 3: run_with_timeout helper
    def another_slow_function():
        time.sleep(5)
        return "Done"
    
    success, result = run_with_timeout(
        another_slow_function,
        timeout_seconds=2,
        on_timeout=lambda: print("  → Timeout callback executed")
    )
    
    if not success and result is None:
        print("✅ Test 3 passed: run_with_timeout handled timeout correctly")
    else:
        print("❌ Test 3 failed: Should have timed out")
        sys.exit(1)
    
    # Test 4: Format duration
    assert format_timeout_duration(7200) == "2h"
    assert format_timeout_duration(3665) == "1h 1m 5s"
    assert format_timeout_duration(90) == "1m 30s"
    print("✅ Test 4 passed: Duration formatting works")
    
    print("\n🎉 All tests passed!")

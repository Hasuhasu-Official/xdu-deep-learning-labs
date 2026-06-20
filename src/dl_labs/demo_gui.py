"""Compatibility entry point for the final demo GUI."""

from .apps.demo_gui import *  # noqa: F401,F403
from .apps.demo_gui import main


if __name__ == "__main__":
    main()

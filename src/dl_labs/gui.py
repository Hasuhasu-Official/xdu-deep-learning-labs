"""Compatibility entry point for the training control GUI."""

from .apps.gui import *  # noqa: F401,F403
from .apps.gui import main


if __name__ == "__main__":
    main()

"""Compatibility entry point for experiment 2 vision code."""

from .exp2_vision.segmentation_sr import *  # noqa: F401,F403
from .exp2_vision.segmentation_sr import main


if __name__ == "__main__":
    main()

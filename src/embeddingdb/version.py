# -*- coding: utf-8 -*-

"""A single-source version for ``embeddingdb``."""

__all__ = [
    'VERSION',
    'get_version',
]

VERSION = '0.0.2-dev'


def get_version() -> str:
    """Get the current software version string."""
    return VERSION

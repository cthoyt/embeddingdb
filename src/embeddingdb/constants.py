# -*- coding: utf-8 -*-

"""Constants for ``embeddingdb``."""

import os

import click
from easy_config import EasyConfig

__all__ = [
    'Config',
    'config',
]

HOME = os.path.expanduser('~')


class Config(EasyConfig):
    """Configuration for ``embeddingdb``."""

    NAME = 'embeddingdb'
    FILES = [
        os.path.join(HOME, '.config', 'embeddingdb', 'config.ini'),
    ]

    #: The SQLAlchemy connection string
    connection: str

    def get_connection_option(self) -> click.Option:
        """Get a click Option for the connection string."""
        return click.option('-c', '--connection', default=self.connection, show_default=True)


config = Config.load()

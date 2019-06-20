# -*- coding: utf-8 -*-

"""The command line interface for ``embeddingdb``.

Why does this file exist, and why not put this in ``__main__``?
You might be tempted to import things from ``__main__`` later, but that will cause
problems--the code will get executed twice:

- When you run `python3 -m embeddingdb` python will execute
  ``__main__.py`` as a script. That means there won't be any
  ``embeddingdb.__main__`` in ``sys.modules``.
- When you import ``__main__`` it will get executed again (as a module) because
  there's no ``embeddingdb.__main__`` in ``sys.modules``.

Also see https://click.pocoo.org/latest/setuptools/
"""

import json
from typing import Optional

import click

from embeddingdb.constants import config
from embeddingdb.sql.analysis import main as analyze
from embeddingdb.sql.io import main as upload
from embeddingdb.sql.models import Collection, get_session


@click.command()
@click.option('--limit', type=int)
@config.get_connection_option()
def ls(limit: Optional[int], connection: str):
    """List the collections in the database."""
    session = get_session(connection)
    collections = session.query(Collection)
    if limit is not None:
        collections = collections.limit(limit)
    click.echo('\t'.join((
        'collection_id',
        'dimensions',
        'package_name',
        'package_version',
        'extras',
    )))
    for collection in collections:
        click.echo('\t'.join((
            str(collection.id),
            str(collection.dimensions),
            collection.package_name,
            collection.package_version,
            json.dumps(collection.extras) if collection.extras else '{}'
        )))


commands = {
    'ls': ls,
    'analyze': analyze,
    'upload': upload,
}

try:
    from embeddingdb.web.wsgi import app


    @click.command()
    def web():
        """Run the web application."""
        app.run()


    commands['web'] = web
except ImportError:
    pass

main = click.Group(commands=commands)

# -*- coding: utf-8 -*-

"""I/O for the relational database for embeddings."""

import json
import os
import random
from typing import Any, Mapping, Optional

import click
from sqlalchemy.orm import Session
from tqdm import tqdm, trange

from .models import Collection, Embedding, get_session
from ..constants import config

__all__ = [
    'load_pykeen',
    'load_word2vec',
]


def load_word2vec(*, path: str, session: Session) -> Collection:
    """Load a word2vec file into the database."""
    with open(path) as file:
        rows, dimensions = map(int, next(file).strip().split())
        collection = Collection(
            dimensions=dimensions,
        )
        for line in tqdm(file, total=rows):
            curie, *vector = line.strip().split()
            embedding = Embedding(
                collection=collection,
                curie=curie,
                vector=[float(x) for x in vector],
            )
            session.add(embedding)
    session.add(collection)
    session.commit()
    return collection


def load_pykeen(*, directory, session: Session) -> Collection:
    """Load a PyKEEN output into the database."""
    config_path = os.path.join(directory, 'configuration.json')
    with open(config_path) as file:
        pykeen_config = json.load(file)

    embedding_path = os.path.join(directory, 'entities_to_embeddings.json')

    collection = Collection(
        dimensions=int(pykeen_config['embedding_dim']),
        package_name='pykeen',
        package_version=pykeen_config['pykeen-version'],
        extras=pykeen_config,
    )

    with open(embedding_path) as file:
        embeddings = json.load(file)

    for curie, vector in tqdm(embeddings.items()):
        embedding = Embedding(
            collection=collection,
            curie=curie,
            vector=vector,
        )
        session.add(embedding)
    session.add(collection)
    session.commit()
    return collection


def load_random(
        *,
        dimensions: Optional[int] = None,
        session: Session,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Collection:
    lamb = random.randint(1, 5)
    dimensions = dimensions if dimensions is not None else 12 * random.randint(3, 8)

    collection = Collection(
        dimensions=dimensions,
        package_name='test',
        package_version='0.0.0',
        extras={
            'lamb': lamb,
        },
    )

    for i in trange(500, **(tqdm_kwargs or {})):
        embedding = Embedding(
            collection=collection,
            curie=f'test:{i}',
            vector=[random.expovariate(lamb) for _ in range(dimensions)]
        )
        session.add(embedding)
    session.add(collection)
    session.commit()
    return collection


@click.command()
@click.option('-f', '--fmt', type=click.Choice(['keen', 'word2vec', 'random']))
@click.option('-p', '--path')
@config.get_connection_option()
def main(fmt: str, path: str, connection: str):
    """Upload embeddings."""
    session = get_session(connection=connection)

    if fmt == 'word2vec':
        load_word2vec(session=session, path=path)
    elif fmt == 'keen':
        load_pykeen(directory=path, session=session)
    else:
        n_random = 5
        for _ in trange(n_random, desc=f'Loading {n_random} random data sets'):
            load_random(
                session=session,
                dimensions=12 * random.randint(3, 8),
                tqdm_kwargs=dict(leave=False),
            )


if __name__ == '__main__':
    main()

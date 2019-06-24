# -*- coding: utf-8 -*-

"""I/O for the relational database for embeddings."""

import json
import os
import random
import sys
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import click
from gensim.models import Word2Vec
from sqlalchemy.orm import Session
from tqdm import tqdm, trange

from .models import Collection, Embedding, get_session
from ..constants import config

__all__ = [
    'upload_word2vec',
    'upload_pykeen_from_directory',
    'upload_word2vec_embedding_file',
    'main',
]


def upload_word2vec(
        model: Union[str, Word2Vec],
        *,
        package_name: str,
        package_version: str,
        extras: Optional[Mapping[str, Any]] = None,
        session: Optional[Session] = None,
        use_tqdm: bool = True,
) -> Collection:
    if session is None:
        session = get_session()

    if isinstance(model, str):
        model = Word2Vec.load(model)

    collection = Collection(
        dimensions=model.vector_size,
        package_name=package_name,
        package_version=package_version,
        extras=extras,
    )

    it = model.wv.vocab
    if use_tqdm:
        it = tqdm(it, desc='Building SQLAlchemy models')
    for curie in it:
        vector = model.wv[curie]
        embedding = Embedding(
            collection=collection,
            curie=curie,
            vector=[float(x) for x in vector],
        )
        session.add(embedding)
    session.add(collection)
    session.commit()
    return collection


def upload_word2vec_embedding_file(
        path: str,
        *,
        package_name: str,
        package_version: str,
        extras: Optional[Mapping[str, Any]] = None,
        session: Optional[Session] = None,
) -> Collection:
    """Load a word2vec file into the database."""
    if session is None:
        session = get_session()

    with open(path) as file:
        it = _spliterate(file)
        rows, dimensions = map(int, next(it))
        collection = Collection(
            dimensions=dimensions,
            package_name=package_name,
            package_version=package_version,
            extras=extras,
        )
        for curie, *vector in tqdm(it, total=rows, desc='Building SQLAlchemy models'):
            embedding = Embedding(
                collection=collection,
                curie=curie,
                vector=[float(x) for x in vector],
            )
            session.add(embedding)
    session.add(collection)
    session.commit()
    return collection


def _spliterate(it: Iterable[str]) -> Iterable[Tuple[str, ...]]:
    for line in it:
        yield line.strip().split()


def upload_pykeen_from_directory(
        directory,
        *,
        session: Optional[Session] = None,
) -> Collection:
    """Load a PyKEEN output into the database."""
    if session is None:
        session = get_session()

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
        session: Optional[Session] = None,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Collection:
    if session is None:
        session = get_session()

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
@click.option('-f', '--fmt', type=click.Choice(['keen', 'word2vec', 'word2vec-model', 'random']))
@click.option('-p', '--path', type=click.Path(file_okay=True, dir_okay=True, exists=True))
@click.option('-m', '--metadata', type=click.File())
@config.get_connection_option()
def main(fmt: str, path: str, metadata, connection: str):
    """Upload embeddings."""
    session = get_session(connection=connection)

    if fmt in {'word2vec', 'word2vec-model'}:
        if not metadata:
            raise ValueError('Must give --metadata for word2vec')
        metadata = json.load(metadata)

        if fmt == 'word2vec-model':
            upload_function = upload_word2vec
        else:
            upload_function = upload_word2vec_embedding_file

        collection = upload_function(
            path,
            session=session,
            package_name=metadata.pop('package_name'),
            package_version=metadata.pop('package_version'),
            extras=metadata,
        )

        click.echo(f'Uploaded collection {collection.id}')
        return sys.exit(0)

    elif fmt == 'keen':
        collection = upload_pykeen_from_directory(directory=path, session=session)
        click.echo(f'Uploaded collection {collection.id}')
        return sys.exit(0)

    else:
        n_random = 5
        it = trange(n_random, desc=f'Loading {n_random} random data sets')
        for _ in it:
            collection = load_random(
                session=session,
                dimensions=12 * random.randint(3, 8),
                tqdm_kwargs=dict(leave=False),
            )
            it.write(f'Uploaded collection {collection.id}')
        return sys.exit(0)


if __name__ == '__main__':
    main()

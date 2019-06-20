# -*- coding: utf-8 -*-

"""A blueprint for a RESTful API."""

from flask import Blueprint, jsonify
from sqlalchemy import and_

from embeddingdb.sql.io import load_random
from embeddingdb.sql.models import Base, Collection, Embedding
from embeddingdb.web.ext import db

__all__ = [
    'api',
]

api = Blueprint(
    'api',
    __name__,
    # url_prefix='/api',  # Maybe add this later if there's more frontend
)


def _collection_to_json(collection: Collection):
    return {
        'id': collection.id,
        'package': {
            'name': collection.package_name,
            'version': collection.package_version,
        },
        'dimensions': collection.dimensions,
        'extras': collection.extras or {},
    }


def _embedding_to_json(embedding: Embedding):
    return {
        'curie': embedding.curie,
        'vector': embedding.vector,
        'collection': _collection_to_json(embedding.collection),
    }


@api.route('/test')
def add_test_data():
    """Add test data."""
    Base.metadata.create_all(db.engine, checkfirst=True)
    collection = load_random(session=db.session, tqdm_kwargs=dict(desc='Adding test collection'))
    return jsonify(
        _collection_to_json(collection)
    )


@api.route('/collection')
def get_collections():
    """Return all collections.

    ---
    tags:
        - collection
    """
    return jsonify([
        _collection_to_json(collection)
        for collection in db.session.query(Collection)
    ])


@api.route('/collection/<int:collection_id>')
def get_collection(collection_id: int):
    """Return a collection.

    ---
    tags:
        - collection
    parameters:
      - name: collection_id
        in: path
        description: The database collection identifier
        required: true
        type: integer
    """
    collection = db.session.query(Collection).get(collection_id)

    return jsonify(
        _collection_to_json(collection)
    )


@api.route('/collection/<int:collection_id>/<curie>')
def get_collection_embedding(collection_id: int, curie: str):
    """Return an entity in a collection.

    ---
    tags:
        - collection
        - entity
    parameters:
      - name: collection_id
        in: path
        description: The database collection identifier
        required: true
        type: integer
      - name: curie
        in: path
        description: The entity's CURIE
        required: true
        type: string
    """
    conditions = and_(
        Embedding.curie == curie,
        Embedding.collection_id == collection_id
    )

    embedding = db.session.query(Embedding).filter(conditions).one_or_none()

    return jsonify(
        _embedding_to_json(embedding)
    )


@api.route('/entity/<curie>')
def get_entity(curie: str):
    """Return an entity in all collections.

    ---
    tags:
        - entity
    parameters:
      - name: curie
        in: path
        description: The entity's CURIE
        required: true
        type: string
    """
    return jsonify([
        _embedding_to_json(embedding)
        for embedding in db.session.query(Embedding).filter(Embedding.curie == curie)
    ])

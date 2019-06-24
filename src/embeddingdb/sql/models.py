# -*- coding: utf-8 -*-

"""SQLAlchemy models for storing embeddings."""

from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import ARRAY, Column, Float, ForeignKey, Integer, JSON, String, UniqueConstraint, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, backref, relationship, scoped_session, sessionmaker

from ..constants import config

__all__ = [
    'Base',
    'Collection',
    'Embedding',
    'get_session',
]

Base = declarative_base()


def get_session(connection: Optional[str] = None) -> Session:
    """Get a scoped session at the given connection."""
    if connection is None:
        connection = config.connection
    engine = create_engine(connection)
    Base.metadata.create_all(bind=engine, checkfirst=True)
    session_maker = sessionmaker(bind=engine)
    session: Session = scoped_session(session_maker)  # override type annotations
    return session


EMBEDDING_TABLE_NAME = 'embeddingdb_embedding'
COLLECTION_TABLE_NAME = 'embeddingdb_collection'


class Collection(Base):
    """Represents a group of embeddings calculated together."""

    __tablename__ = COLLECTION_TABLE_NAME
    id = Column(Integer, primary_key=True)

    dimensions = Column(Integer, index=True, unique=False, nullable=False,
                        doc='Dimensionality of the embeddings in this collection')
    package_name = Column(String, index=True, unique=False, nullable=True,
                          doc='The package used to generate the entity embeddings')
    package_version = Column(String, index=True, unique=False, nullable=True,
                             doc='The version of the package used to generate the entity embeddings')
    extras = Column(JSON, index=False, unique=False, nullable=True,
                    doc='Extra information associated with the collection')

    def as_ndarray(self) -> np.ndarray:
        """Get this collection as a numpy array (with no labels)."""
        return np.array([
            embedding.vector
            for embedding in self.embeddings.order_by(Embedding.curie)
        ])

    def as_dataframe(self) -> pd.DataFrame:
        """Get this collection as a pandas DataFrame."""
        return pd.DataFrame.from_dict(
            {
                embedding.curie: embedding.vector
                for embedding in self.embeddings.order_by(Embedding.curie)
            },
            orient='index',
        )


class Embedding(Base):
    """Represents the embedding for an entity."""

    __tablename__ = EMBEDDING_TABLE_NAME
    id = Column(Integer, primary_key=True)

    # Consider normalizing out entity to new table
    curie = Column(String(1023), index=True, unique=False, nullable=False, doc='CURIE for the entity')
    vector = Column(ARRAY(Float), nullable=False, doc='Embedding for entity')

    collection_id = Column(Integer, ForeignKey(f'{Collection.__tablename__}.id'), nullable=False, index=True)
    collection = relationship(Collection, backref=backref('embeddings', lazy='dynamic', cascade="all, delete-orphan"))

    __table_args__ = (
        UniqueConstraint(collection_id, curie),
    )

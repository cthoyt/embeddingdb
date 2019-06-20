Embedding Database
==================
This package provides a database schema and Python wrapper
for storing the embeddings generated through various representation
learning packages.

Currently, this package focuses on using a SQL database with SQLAlchemy,
but might be extended to use a NoSQL database as an alternative.

Installation
------------
Install ``embeddingdb`` directly from GitHub with:

.. code-block:: sh

   $ pip install git+https://github.com/cthoyt/embeddingdb

Set the environment variable ``EMBEDDINGDB_CONNECTION`` to a valid
SQLAlchemy connection string for a PostgreSQL instance, as this package uses
the PostgreSQL-specific ``ARRAY`` type.

Running with Docker
-------------------
After installing Docker, the entire web application can be instantiated with:

.. code-block:: sh

   $ docker-compose up

Get the endpoint ``/test`` to instantiate the database and add a test collection.

# -*- coding: utf-8 -*-

"""A wrapper around creation of the web application."""

from flask import Flask

from embeddingdb.constants import config
from embeddingdb.web.api import api
from embeddingdb.web.ext import db, swagger

__all__ = [
    'get_app',
]


def get_app() -> Flask:
    """Build a Flask instance."""
    app = Flask(__name__)

    # Set configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = config.connection
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions
    db.init_app(app)
    swagger.init_app(app)

    # Register blueprints
    app.register_blueprint(api)

    return app

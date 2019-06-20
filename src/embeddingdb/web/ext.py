# -*- coding: utf-8 -*-

"""Flask extensions that can be lazily accessed before instantiation of the web application."""

from flasgger import Swagger
from flask_sqlalchemy import SQLAlchemy

from embeddingdb.version import VERSION

__all__ = [
    'db',
    'swagger',
]

db = SQLAlchemy()

swagger_config = Swagger.DEFAULT_CONFIG.copy()
swagger_config.update({
    'title': 'Embedding Database API',
    'description': 'This exposes the functions of embeddingdb as a RESTful API',
    'contact': {
        'responsibleOrganization': 'Fraunhofer SCAI',
        'responsibleDeveloper': 'Charles Tapley Hoyt',
        'email': 'charles.hoyt@scai.fraunhofer.de',
        'url': 'https://www.scai.fraunhofer.de/de/geschaeftsfelder/bioinformatik.html',
    },
    'version': VERSION,
    'specs_route': '/'
})
swagger = Swagger(config=swagger_config)

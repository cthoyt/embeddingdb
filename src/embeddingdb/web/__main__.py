# -*- coding: utf-8 -*-

"""Run the web application with ``python -m embeddingdb.web``."""

from .wsgi import app

if __name__ == '__main__':
    app.run()

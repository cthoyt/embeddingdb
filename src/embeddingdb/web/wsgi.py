# -*- coding: utf-8 -*-

"""A WSGI formulation of the web application.

Also allows the web application to be run with ``python -m embeddingdb.web.wsgi``.
"""

from embeddingdb.web.app import get_app

app = get_app()

if __name__ == '__main__':
    app.run()

# -*- coding: utf-8 -*-

"""Compute cross-correlations in embedding collections."""

from typing import BinaryIO, Mapping, Optional, Type, Union

import click
import joblib
from sklearn.base import RegressorMixin
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.linear_model import (
    LinearRegression, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso,
    MultiTaskLassoCV,
)
from sklearn.metrics import r2_score
from sqlalchemy.orm import Session

from embeddingdb.constants import config
from embeddingdb.sql.models import Collection, get_session

__all__ = [
    'perform_regression',
    'main',
]

_REGRESSIONS = {
    'linear': LinearRegression,
    'pls': PLSRegression,
    'cca': CCA,
    'elastic': MultiTaskElasticNet,
    'elastic-cv': MultiTaskElasticNetCV,
    'lasso': MultiTaskLasso,
    'lasso-cv': MultiTaskLassoCV,
    # 'svr': sklearn.svm.SVR,
}


def calculate_overlap():
    """Calculate the pairwise overlap between all collections."""
    raise NotImplementedError


def perform_regression(
        collection_1: Collection,
        collection_2: Collection,
        regression_cls: Union[None, str, Type[RegressorMixin]] = None,
        regression_kwargs: Optional[Mapping] = None,
        output: Union[None, str, BinaryIO] = None,
):
    """Perform a regression between two collections of embeddings and evaluate the results.

    :param collection_1: The first collection
    :param collection_2: The second collection
    :param regression_cls: Class or shortcut name to class that is a ``RegressorMixin``. Valid shortcuts are 'lienar'
     and 'pls'.
    :param regression_kwargs: Keyword arguments to pass to the regressor class on instantiation
    :param output: Optional path to output the regressor model using ``joblib``
    :return:
    """
    if regression_cls is None:
        regression_cls = LinearRegression
    elif isinstance(regression_cls, str):
        regression_cls = _REGRESSIONS[regression_cls]
    elif not issubclass(regression_cls, RegressorMixin):
        raise TypeError(f'regression_cls had invalid type: {regression_cls}')

    clf = regression_cls(**(regression_kwargs or {}))
    x = collection_1.as_dataframe()
    y = collection_2.as_dataframe()

    index = x.index & y.index
    x = x.loc[index]
    y = y.loc[index]

    clf.fit(x, y)

    if output is not None:
        joblib.dump(clf, output)

    y_pred = clf.predict(x)
    r2 = r2_score(y, y_pred)

    return clf, r2, len(index), len(index) / min(len(x.index), len(y.index))


def _get_collection(session: Session, collection_id: int) -> Collection:
    """Get a collection by its identifier."""
    return session.query(Collection).get(collection_id)


@click.command()
@click.argument('id_1', type=int)
@click.argument('id_2', type=int)
@click.option('-m', '--model', type=click.Choice(list(_REGRESSIONS)), default='linear')
@click.option('-o', '--output', type=click.File('wb'))
@config.get_connection_option()
def main(id_1: int, id_2: int, model: Optional[str], output: Optional[BinaryIO], connection: str):
    """Perform a regression between two collections."""
    session = get_session(connection=connection)
    collection_1 = _get_collection(session, id_1)
    collection_2 = _get_collection(session, id_2)

    clf, r2, intersect, intersect_percent = perform_regression(
        collection_1,
        collection_2,
        regression_cls=model,
        output=output,
    )
    click.echo(f'Model: {clf}')
    click.echo(f'Dimensions: {clf.coef_.shape}')
    click.echo(f'R^2: {r2:.3f}')
    click.echo(f'Intersection: {intersect} ({intersect_percent:.1%})')


if __name__ == '__main__':
    main()

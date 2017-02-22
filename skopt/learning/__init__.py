"""Machine learning extensions for model-based optimization."""

from .forest import RandomForestRegressor
from .forest import RandomForestQuantileRegressor
from .forest import ExtraTreesRegressor
from .gbrt import GradientBoostingQuantileRegressor
from .gaussian_process import GaussianProcessRegressor


__all__ = ("RandomForestRegressor",
           "ExtraTreesRegressor",
           "RandomForestQuantileRegressor",
           "GradientBoostingQuantileRegressor",
           "GaussianProcessRegressor")

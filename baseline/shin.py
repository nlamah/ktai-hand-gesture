"""Shared public interface to the unchanged Shin implementation.

The implementation stays in research/shin.py because recorded source hashes
and saved models require that path. These imports reuse the same functions.
"""
from research.shin import (
    VERSION, ALL_PAIRS, DISTANCE_PAIRS, TABLE2_OMISSIONS,
    raw_distances, distance_features, angle_features, combined_features,
    distance_for_sample, angle_for_sample, combined_for_sample,
)

__all__ = [
    'VERSION', 'ALL_PAIRS', 'DISTANCE_PAIRS', 'TABLE2_OMISSIONS',
    'raw_distances', 'distance_features', 'angle_features', 'combined_features',
    'distance_for_sample', 'angle_for_sample', 'combined_for_sample',
]

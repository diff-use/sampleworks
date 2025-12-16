"""Metrics for model evaluation.

This module provides the base metric framework.
This file originated in RosettaCommons/foundry and is licensed under BSD-3-Clause.
"""

from sampleworks.metrics.metric import Metric, MetricInputError, MetricManager


__all__ = [
    "Metric",
    "MetricManager",
    "MetricInputError",
]

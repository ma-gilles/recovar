"""Per-step training metrics logging, with an optional Comet ML backend.

Ported from the standalone SOLVAR repo's ``solvar/logger.py``. Trimmed to the
Comet backend only: this repo's training loops (e.g.
:class:`recovar.solvar.solvar.Trainer`) already persist every step's metrics
dict to disk (``solvar_iteration_data.pkl``), so a CSV backend would just
duplicate that.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

try:
    import comet_ml

    COMET_ML_AVAILABLE = True
except ImportError:
    COMET_ML_AVAILABLE = False
    comet_ml = None

if TYPE_CHECKING:
    from comet_ml import Experiment as CometExperiment

logger = logging.getLogger(__name__)


class MetricsLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        raise NotImplementedError


class NullMetricsLogger(MetricsLogger):
    def log_metrics(self, metrics: dict, step: int):
        pass


class CometMLMetricsLogger(MetricsLogger):
    def __init__(self, experiment: "CometExperiment | None" = None, **comet_kwargs):
        """Initialize CometML logger.

        Args:
            experiment: Optional existing CometML Experiment object. If None, creates a
                new one.
            **comet_kwargs: Additional arguments passed to comet_ml.start() if
                experiment is None.
                Common kwargs include: project_name, workspace, api_key, etc.
        """

        self.experiment: "CometExperiment | None"
        if not COMET_ML_AVAILABLE:
            logger.debug("CometML is not available. %s will not log metrics.", type(self).__name__)
            self.experiment = None
            return

        if experiment is not None:
            self.experiment = experiment
        elif self._should_start_new_experiment(comet_kwargs):
            self.experiment = comet_ml.start(**comet_kwargs)
        elif (running := comet_ml.get_running_experiment()) is not None:
            self.experiment = running
        else:
            logger.debug(
                "No existing CometML experiment found and no new experiment "
                "parameters provided. %s will not log metrics.",
                type(self).__name__,
            )
            self.experiment = None

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to CometML when available."""
        if self.experiment is not None:
            self.experiment.log_metrics(metrics, step=step)

    def _should_start_new_experiment(self, comet_kwargs) -> bool:
        """Determine whether to start a new CometML experiment based on provided kwargs."""

        if os.environ.get("COMET_EXPERIMENT_KEY"):
            # If COMET_EXPERIMENT_KEY is set externally, we assume the user wants to use an existing experiment.
            # in this case comet_ml.start() will automatically use the existing experiment.
            return True

        if "experiment_key" in comet_kwargs.keys() or "project_name" in comet_kwargs.keys():
            # If experiment_key or project_name is explicitly provided in comet_kwargs
            # we should start a new experiment.
            return True

        return False

# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""
import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union
import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set, Union

from torch import Tensor

from lightning_fabric.loggers.logger import Logger, rank_zero_experiment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.logger import _add_prefix
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.types import _PATH


from lightning_fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger
from lightning_fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning_fabric.utilities.logger import _convert_params
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

log = logging.getLogger(__name__)

import glob


class CSVLogger(Logger, FabricCSVLogger):
    r"""Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'lightning_logs'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: _PATH,
        name: str = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__(
            root_dir=save_dir,
            name=name,
            version=version,
            prefix=prefix,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )
        self._save_dir = os.fspath(save_dir)

    @property
    def root_dir(self) -> str:
        """Parent directory for all checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will
        be saved in "save_dir/version"

        """
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, version)

    @property
    def save_dir(self) -> str:
        """The current directory where logs are saved.

        Returns:
            The path to current directory where logs are saved.

        """
        return self._save_dir

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # type: ignore[override]
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @property
    @rank_zero_experiment
    def experiment(self) -> _FabricExperimentWriter:
        r"""Actual _ExperimentWriter object. To use _ExperimentWriter features in your
        :class:`~pytorch_lightning.core.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir)
        return self._experiment




def get_latest_ckpt(log_dir):
    ckpt_files = glob.glob(os.path.join(log_dir, '*.ckpt'))  
    if ckpt_files:
        latest_file = max(ckpt_files, key=os.path.getctime)  
    else:  
        latest_file = None
    return latest_file

class _ExperimentWriter:
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """

    NAME_METRICS_FILE = "metrics.csv"
    NAME_TXT_FILE = "log.txt"
    def __init__(self, log_dir: str) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []

        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        self._fs.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)
        self.txt_path = os.path.join(self.log_dir, self.NAME_TXT_FILE)
        
    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams."""
        pass
        
    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)

        with open(self.txt_path, "a") as f:
            if "step" in self.metrics[-1]:
                f.write("step: " + str(self.metrics[-1]["step"]) + "\n")
            for key, value in self.metrics[-1].items():
                if key != "step":
                    f.write(key + ": " + str(value) + ";")
            f.write("\n")
            
    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        try:
            new_keys = self._record_new_keys()
            file_exists = self._fs.isfile(self.metrics_file_path)
            if new_keys and file_exists:
                # we need to re-write the file if the keys (header) change
                self._rewrite_with_new_header(self.metrics_keys)
                with self._fs.open(self.metrics_file_path, mode=("a" if file_exists else "w"), newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
                    if not file_exists:
                        # only write the header if we're writing a fresh file
                        writer.writeheader()
                    writer.writerows(self.metrics)
        except Exception as e:
            print(e)
        
        with open(self.txt_path, "a") as f:
            if "step" in self.metrics[-1]:
                f.write("step: " + str(self.metrics[-1]["step"]) + "\n")
            for key, value in self.metrics[-1].items():
                if key != "step":
                    f.write(key + ": " + str(value) + ";")
            f.write("\n")
        
        
        self.metrics = []  # reset

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: List[str]) -> None:
        with self._fs.open(self.metrics_file_path, "r", newline="") as file:
            metrics = list(csv.DictReader(file))

        with self._fs.open(self.metrics_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

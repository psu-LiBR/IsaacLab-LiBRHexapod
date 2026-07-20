import csv

import numpy as np

from sim2real.logging_utils import CsvRunLogger
from sim2real.profiles import NUM_JOINTS


def test_csv_run_logger_writes_header_and_rows(tmp_path):
    path = tmp_path / "run.csv"
    obs_dim, action_dim = 33, 8

    with CsvRunLogger(str(path), obs_dim=obs_dim, action_dim=action_dim) as logger:
        logger.log(
            step=0, t=0.0, obs=np.zeros(obs_dim), raw_action=np.zeros(action_dim),
            target_sim=np.zeros(NUM_JOINTS), target_ticks=np.zeros(NUM_JOINTS),
            measured_pos_sim=np.zeros(NUM_JOINTS), measured_vel_sim=np.zeros(NUM_JOINTS),
            loop_time_ms=5.0,
        )
        logger.log(
            step=1, t=0.02, obs=np.ones(obs_dim), raw_action=np.ones(action_dim),
            target_sim=np.ones(NUM_JOINTS), target_ticks=np.full(NUM_JOINTS, 2048),
            measured_pos_sim=np.ones(NUM_JOINTS), measured_vel_sim=np.ones(NUM_JOINTS),
            loop_time_ms=6.0,
        )

    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    expected_cols = 3 + obs_dim + action_dim + 4 * NUM_JOINTS
    assert len(rows[0]) == expected_cols
    assert len(rows) == 3  # header + 2 data rows
    assert rows[0][:3] == ["step", "t", "loop_time_ms"]
    assert rows[0][3] == "obs_0"
    assert rows[1][0] == "0"

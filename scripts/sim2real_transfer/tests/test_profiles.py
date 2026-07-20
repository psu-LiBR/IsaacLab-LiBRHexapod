import numpy as np
import pytest

from sim2real.profiles import PROFILES, GoalObsBuilder, VelocityObsBuilder, make_obs_builder


def test_velocity_profile_dims():
    spec = PROFILES["velocity"]
    assert spec.obs_dim == 33
    assert spec.action_dim == 8


def test_goal_profile_dims():
    spec = PROFILES["goal"]
    assert spec.obs_dim == 34
    assert spec.action_dim == 8


def test_velocity_obs_order():
    builder = VelocityObsBuilder()
    gyro = np.array([1.0, 2.0, 3.0])
    gravity = np.array([4.0, 5.0, 6.0])
    joint_pos = np.arange(10.0, 18.0)  # 8 values
    joint_vel = np.arange(20.0, 28.0)
    last_action = np.arange(30.0, 38.0)
    q_default = np.zeros(8)
    command = np.array([0.2, 0.0, 0.0])

    obs = builder.build(gyro, gravity, joint_pos, joint_vel, last_action, q_default, command)

    assert obs.shape == (33,)
    assert np.allclose(obs[0:3], gyro)
    assert np.allclose(obs[3:6], gravity)
    assert np.allclose(obs[6:9], command)
    assert np.allclose(obs[9:17], joint_pos - q_default)
    assert np.allclose(obs[17:25], joint_vel)
    assert np.allclose(obs[25:33], last_action)


def test_goal_obs_order_command_is_4dim():
    builder = GoalObsBuilder()
    gyro = np.zeros(3)
    gravity = np.array([0.0, 0.0, -1.0])
    joint_pos = np.zeros(8)
    joint_vel = np.zeros(8)
    last_action = np.zeros(8)
    q_default = np.zeros(8)
    command = np.array([5.0, 0.0, 0.0, 0.0])

    obs = builder.build(gyro, gravity, joint_pos, joint_vel, last_action, q_default, command)

    assert obs.shape == (34,)
    assert np.allclose(obs[6:10], command)


def test_joint_pos_rel_subtracts_default():
    builder = VelocityObsBuilder()
    joint_pos = np.full(8, -0.47)
    q_default = np.full(8, -0.47)
    obs = builder.build(
        np.zeros(3), np.array([0, 0, -1.0]), joint_pos, np.zeros(8), np.zeros(8),
        q_default, np.zeros(3),
    )
    assert np.allclose(obs[9:17], 0.0)


def test_wrong_command_dim_raises():
    builder = VelocityObsBuilder()
    with pytest.raises(ValueError):
        builder.build(
            np.zeros(3), np.zeros(3), np.zeros(8), np.zeros(8), np.zeros(8),
            np.zeros(8), np.zeros(4),  # wrong dim for velocity profile
        )


def test_make_obs_builder_factory():
    assert isinstance(make_obs_builder("velocity"), VelocityObsBuilder)
    assert isinstance(make_obs_builder("goal"), GoalObsBuilder)
    with pytest.raises(ValueError):
        make_obs_builder("bogus")

import contextlib
import importlib
import sys
import types

import numpy as np
import pytest

import robomimic.envs as RobomimicEnvs
import robomimic.utils.obs_utils as ObsUtils

_MISSING = object()


@pytest.fixture
def load_env_robosuite_module(monkeypatch):
    @contextlib.contextmanager
    def _load(image_convention):
        original_module = sys.modules.get("robomimic.envs.env_robosuite", _MISSING)
        original_pkg_attr = getattr(RobomimicEnvs, "env_robosuite", _MISSING)

        fake_robosuite = types.ModuleType("robosuite")
        fake_robosuite.__version__ = "1.4.1"

        fake_transform_utils = types.ModuleType("robosuite.utils.transform_utils")
        fake_transform_utils.make_pose = lambda pos, rot: np.eye(4)

        fake_utils = types.ModuleType("robosuite.utils")
        fake_utils.transform_utils = fake_transform_utils

        fake_macros = types.ModuleType("robosuite.macros")
        fake_macros.IMAGE_CONVENTION = image_convention

        monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
        monkeypatch.setitem(sys.modules, "robosuite.utils", fake_utils)
        monkeypatch.setitem(sys.modules, "robosuite.utils.transform_utils", fake_transform_utils)
        monkeypatch.setitem(sys.modules, "robosuite.macros", fake_macros)
        fake_robosuite.utils = fake_utils
        fake_robosuite.macros = fake_macros

        sys.modules.pop("robomimic.envs.env_robosuite", None)
        if hasattr(RobomimicEnvs, "env_robosuite"):
            delattr(RobomimicEnvs, "env_robosuite")

        try:
            module = importlib.import_module("robomimic.envs.env_robosuite")
            yield module
        finally:
            sys.modules.pop("robomimic.envs.env_robosuite", None)
            if original_module is _MISSING:
                sys.modules.pop("robomimic.envs.env_robosuite", None)
            else:
                sys.modules["robomimic.envs.env_robosuite"] = original_module

            if original_pkg_attr is _MISSING:
                if hasattr(RobomimicEnvs, "env_robosuite"):
                    delattr(RobomimicEnvs, "env_robosuite")
            else:
                setattr(RobomimicEnvs, "env_robosuite", original_pkg_attr)

    return _load


def _make_env(module):
    env = module.EnvRobosuite.__new__(module.EnvRobosuite)
    env._is_v1 = True
    env._lang_emb = None
    env._init_kwargs = {}
    env.get_real_depth_map = lambda depth: depth + 10.0
    env.env = types.SimpleNamespace(
        sim=types.SimpleNamespace(
            render=lambda **kwargs: np.array(
                [
                    [[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                ],
                dtype=np.uint8,
            )
        ),
        robots=[],
    )
    return env


def test_get_observation_respects_opencv_image_convention(load_env_robosuite_module):
    with load_env_robosuite_module("opencv") as module:
        ObsUtils.initialize_obs_modality_mapping_from_dict(
            {"rgb": ["agentview_image"], "depth": ["agentview_depth"]}
        )

        env = _make_env(module)
        rgb = np.array(
            [
                [[1, 0, 0], [2, 0, 0]],
                [[3, 0, 0], [4, 0, 0]],
            ],
            dtype=np.uint8,
        )
        depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        obs = env.get_observation({"agentview_image": rgb, "agentview_depth": depth})

        np.testing.assert_array_equal(obs["agentview_image"], rgb)
        np.testing.assert_allclose(obs["agentview_depth"], depth[..., None] + 10.0)


def test_get_observation_flips_opengl_image_convention(load_env_robosuite_module):
    with load_env_robosuite_module("opengl") as module:
        ObsUtils.initialize_obs_modality_mapping_from_dict(
            {"rgb": ["agentview_image"], "depth": ["agentview_depth"]}
        )

        env = _make_env(module)
        rgb = np.array(
            [
                [[1, 0, 0], [2, 0, 0]],
                [[3, 0, 0], [4, 0, 0]],
            ],
            dtype=np.uint8,
        )
        depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        obs = env.get_observation({"agentview_image": rgb, "agentview_depth": depth})

        np.testing.assert_array_equal(obs["agentview_image"], rgb[::-1])
        np.testing.assert_allclose(obs["agentview_depth"], depth[::-1][..., None] + 10.0)


def test_render_respects_image_convention(load_env_robosuite_module):
    with load_env_robosuite_module("opencv") as opencv_module:
        opencv_env = _make_env(opencv_module)
        np.testing.assert_array_equal(
            opencv_env.render(mode="rgb_array"),
            np.array(
                [
                    [[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                ],
                dtype=np.uint8,
            ),
        )

    with load_env_robosuite_module("opengl") as opengl_module:
        opengl_env = _make_env(opengl_module)
        np.testing.assert_array_equal(
            opengl_env.render(mode="rgb_array"),
            np.array(
                [
                    [[3, 3, 3], [4, 4, 4]],
                    [[1, 1, 1], [2, 2, 2]],
                ],
                dtype=np.uint8,
            ),
        )


def test_load_env_robosuite_module_restores_import_state(load_env_robosuite_module):
    original_module = sys.modules.get("robomimic.envs.env_robosuite", _MISSING)
    original_pkg_attr = getattr(RobomimicEnvs, "env_robosuite", _MISSING)

    with load_env_robosuite_module("opencv") as module:
        assert sys.modules["robomimic.envs.env_robosuite"] is module
        assert getattr(RobomimicEnvs, "env_robosuite") is module

    restored_module = sys.modules.get("robomimic.envs.env_robosuite", _MISSING)
    restored_pkg_attr = getattr(RobomimicEnvs, "env_robosuite", _MISSING)

    assert restored_module is original_module
    assert restored_pkg_attr is original_pkg_attr

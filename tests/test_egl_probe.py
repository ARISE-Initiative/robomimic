"""
Test that the optional egl_probe dependency is handled gracefully: robomimic
environment wrappers must work without it, and use it to pick the rendering
GPU when it is installed.

Runs without MuJoCo or GPUs: robosuite and egl_probe are replaced by stubs.
Run with `python test_egl_probe.py` or `pytest test_egl_probe.py`.
"""
import contextlib
import sys
import types
import unittest

import robomimic.utils.env_utils as EnvUtils


_MISSING = object()


@contextlib.contextmanager
def patched_modules(modules, reload=()):
    """
    Temporarily set entries in sys.modules (a None entry makes the import fail) and
    drop the modules in @reload so that they get imported again against the patched
    entries. Only these entries are restored afterwards, so modules imported for the
    first time inside the block (e.g. torch) stay loaded.
    """
    names = list(modules) + list(reload)
    saved = {name: sys.modules.get(name, _MISSING) for name in names}
    sys.modules.update(modules)
    for name in reload:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _fake_robosuite():
    """Minimal stand-in for robosuite that records kwargs passed to robosuite.make."""
    robosuite = types.ModuleType("robosuite")
    robosuite.__version__ = "1.5.1"
    robosuite.make_kwargs = []

    def make(env_name, **kwargs):
        robosuite.make_kwargs.append(kwargs)
        return types.SimpleNamespace(observation_names=[])

    robosuite.make = make
    utils = types.ModuleType("robosuite.utils")
    transform_utils = types.ModuleType("robosuite.utils.transform_utils")
    robosuite.utils = utils
    utils.transform_utils = transform_utils
    return {
        "robosuite": robosuite,
        "robosuite.utils": utils,
        "robosuite.utils.transform_utils": transform_utils,
    }


def _fake_egl_probe(devices):
    egl_probe = types.ModuleType("egl_probe")
    egl_probe.get_available_devices = lambda: list(devices)
    return egl_probe


class TestOptionalEglProbe(unittest.TestCase):

    def setUp(self):
        EnvUtils._EGL_PROBE_WARNING_SHOWN = False

    def test_device_ids_without_egl_probe(self):
        with patched_modules({"egl_probe": None}):
            self.assertEqual(EnvUtils.get_egl_device_ids(), [])
            self.assertTrue(EnvUtils._EGL_PROBE_WARNING_SHOWN)

    def test_device_ids_with_egl_probe(self):
        with patched_modules({"egl_probe": _fake_egl_probe([2, 3])}):
            self.assertEqual(EnvUtils.get_egl_device_ids(), [2, 3])

    def _make_offscreen_env(self, egl_probe_module):
        fake_robosuite = _fake_robosuite()
        modules = dict(fake_robosuite, egl_probe=egl_probe_module)
        with patched_modules(modules, reload=["robomimic.envs.env_robosuite"]):
            from robomimic.envs.env_robosuite import EnvRobosuite
            EnvRobosuite(env_name="Lift", render=False, render_offscreen=True)
        return fake_robosuite["robosuite"].make_kwargs[-1]

    def test_robosuite_env_without_egl_probe(self):
        kwargs = self._make_offscreen_env(None)
        self.assertTrue(kwargs["has_offscreen_renderer"])
        # no device picked, so the simulator uses its default rendering device
        self.assertNotIn("render_gpu_device_id", kwargs)

    def test_robosuite_env_with_egl_probe(self):
        kwargs = self._make_offscreen_env(_fake_egl_probe([1]))
        self.assertEqual(kwargs["render_gpu_device_id"], 1)


if __name__ == "__main__":
    unittest.main()

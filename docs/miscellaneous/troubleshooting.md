# Troubleshooting

## Common Mistakes

This section will be populated with common mistakes and links to issues and fixes as we are made aware of them.

If you're still having problems, please submit an issue on [the github page](https://github.com/ARISE-Initiative/robomimic/issues).

## Known Issues

This section contains known issues that are either minor, or that will be patched later.

- `PrintLogger` breaks if you use `embed()` with a new-ish IPython installation. The current workaround is to use an old version. Known working version is `ipython==5.8.0`

- On robomimic v0.2, the `test_scripts` tests will fail if `robosuite` is not on the `offline_study` branch with the following error: `No "site" with name gripper0_ee_x exists.`. This is because the test hdf5 was collected on that branch -- switching to that branch should make the test pass.
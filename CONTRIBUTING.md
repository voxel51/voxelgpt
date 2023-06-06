# Contributing to VoxelGPT

Thanks for your interest in contributing to VoxelGPT!

## Pre-commit hooks

If you plan to contribute a PR, please check install the pre-commit hooks
before commiting:

```shell
pre-commit install
```

You can manually lint a file if necessary like so:

```shell
# Manually run linting configured in the pre-commit hook
pre-commit run --files <file>
```

## Using the plugin in the FiftyOne App

When developing locally, you must make your source install of VoxelGPT
available as a plugin in order to access it in the FiftyOne App.

A convenient way to do that is to symlink your `voxelgpt` directory into your
FiftyOne plugins directory:

```shell
# Symlinks your clone of voxelgpt into your FiftyOne plugins directory
ln -s "$(pwd)" "$(fiftyone config plugins_dir)/voxelgpt"
```

## Developing and building the plugin JS bundle

To build the Fiftyone plugin you must:

-   Install `fiftyone` from source.
    [See here](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md)
    for instructions
-   Set the `FIFTYONE_DIR` environment variable to point to your `fiftyone`
    source directory
-   Have `yarn@3.5.x` installed
-   Run `yarn install` to install the `voxelgpt` dependencies

To create a build, run:

```shell
# production build of the plugin js bundle
yarn build

# for rebuilding the bundle automatically during development
yarn dev
```

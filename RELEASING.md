# Releasing a new VoxelGPT version

> You must have the [contributor steps](CONTRIBUTING.md) completed before
> running the commands below

## Release script

The simplest way to release a new version of VoxelGPT is to run the following
script, which automates the steps described below.

```shell
yarn release $VERSION
```

## Manually build the archive

Alternatively, you can follow these steps to manually create a release.

### Build the plugin

```shell
yarn build
```

### Bump the version

Bump the version of the plugin by running:

```shell
# bumps the patch version (for bug fixes only)
yarn bump

# manually set the version
yarn bump $VERSION
```

### Commit all files

Only files committed locally will be included in the plugin archive.

This is also a good time to tag the new version.

```shell
VERSION=1.2.3

git checkout -b release/$VERSION
git add .  # files you want included
git commit -m 'release version $VERSION'
git tag $VERSION

git push origin --follow-tags  # push the commit and tags
```

### Create the plugin archive

```shell
yarn archive
```

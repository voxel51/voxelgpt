## How to deploy VoxelGPT to FiftyOne Teams

> You must have the [contributor steps](CONTRIBUTING.md) completed before
> running the commands below

## Release Script

To create a release run the following.

```shell
yarn release <version>
```

You can also follow these steps to manually create a release.

### Build the latest

```shell
yarn build
```

### Bump the version

Bump the version of the plugin by running:

```shell
# bumps the patch version (for bug fixes only)
yarn bump

# manually set the version
yarn bump <version>
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

### Upload to Teams

Go to
[https://YOUR_FIFTYONE_TEAMS/settings/plugins](https://YOUR_FIFTYONE_TEAMS/settings/plugins).

To install a new plugin, click "Install plugin". To upgrade an existing plugin,
find it in the list and click the 3 dots and choose "Upgrade plugin".

Upload the newly created archive.

### Set your permissions

Find the plugin in the list and click on "X operators". Select the appropriate
permissions for your plugin.

### That's it!

At this point you should have a newly installed/upgraded plugin. Users will see
this change immediately.

### Troubleshooting

If you are seeing issues with a plugin not updating:

-   check the logs for any additional information
-   restart the appropriate pods (if you have `teams-plugins` pods, those
    should be the only ones restarted, otherwise restart the `fiftyone-app`
    pods.)

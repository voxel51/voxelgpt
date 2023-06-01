## How to deploy voxelgpt to Fiftyone Teams

> **You must have the [contributor steps](README.md#contributing) completed before running the commands below**

**Bump the version**

Bump the version of the plugin by running:

```
# bumps the patch version (for bug fixes only)
yarn bump

# sets the version
yarn bump 1.2.3
```

**Commit all Files**

Only files committed locally will be included in the plugin archive.

This is also a good time to tag the new version.

```
VERSION=1.2.3
git checkout -b release/$VERSION
git add . # files you want included
git commit -m 'release version $VERSION' # this will be in the output from the command above
git tag $VERSION
git push origin --follow-tags # push the commit and tags
```

**Create the Plugin Archive**

```
yarn archive
```

**Upload Archive to Teams**

Goto [https://MY_FIFTYONE_TEAMS/settings/plugins](https://MY_FIFTYONE_TEAMS/settings/plugins).

To install a new plugin, click "Install plugin". To upgrade an existing plugin, find it in the list and click the 3 dots and choose "Upgrade plugin".

Upload the newly created archive.

**Set your Permissions**

Find the plugin in the list and click on "X operators". Select the appropriate permissions for your plugin.

**That's it!**

At this point you should have a newly installed/upgraded plugin. Users will see this change immediately.

**Troubleshooting Tips**

If you are seeing issues with a plugin not updating:

 - check the logs for any additional information
 - restart the appropriate pods (if you have `teams-plugins` pods, those should be the only ones restarted, otherwise restart the `fiftyone-app` pods.)
# Deploying VoxelGPT to FiftyOne Teams

This guide explains how to upload the latest version of VoxelGPT to your
[FiftyOne Teams](https://voxel51.com/fiftyone-teams) deployment.

## Uploading the code

You can upload VoxelGPT manually or via the
[Management SDK](https://docs.voxel51.com/teams/management_sdk.html).

### Manual upload

First clone and ZIP the repository:

```shell
git clone https://github.com/voxel51/voxelgpt
cd voxelgpt

git archive --format=zip --output=archives/voxelgpt.zip
```

Then follow
[these instructions](https://docs.voxel51.com/teams/teams_plugins.html) to
upload the ZIP via the UI.

### Management SDK

You can also programmatically upload the plugin via the
[Management SDK](https://docs.voxel51.com/teams/management_sdk.html):

```shell
git clone https://github.com/voxel51/voxelgpt
```

```py
import fiftyone.management as fom

fom.upload_plugin("/path/to/voxelgpt", optimize=True, overwrite=True)
```

### Troubleshooting

If you are seeing issues with the plugin not updating in the App:

-   Check the logs for any additional information
-   Restart your `teams-plugins` pods

## Updating your `teams-plugins` container

VoxelGPT requires certain Python packages to be installed. You can see the
current requirements by running:

```shell
cat requirements.txt
```

If any of these Python packages are not available in your `teams-plugins`
containers, you'll need to add them.

You must also ensure that a valid OpenAI API key
([create one](https://platform.openai.com/account/api-keys)) is available to
the containers via the `OPENAI_API_KEY` environment variable.

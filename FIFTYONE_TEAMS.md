# Deploying VoxelGPT to FiftyOne Teams

This guide explains how to upload the latest version of VoxelGPT to your
[FiftyOne Teams](https://voxel51.com/fiftyone-teams) deployment.

## Enable Plugins for FiftyOne Teams

In order to run the VoxelGPT plugin you must have enabled Plugins for
FiftyOne Teams.

Instructions for enabling  FiftyOne Teams Plugins with Docker Compose
are [here](https://github.com/voxel51/fiftyone-teams-app-deploy/tree/main/docker#enabling-fiftyone-teams-plugins)

Instructions for enabling FiftyOne Teams Plugins with Helm are
[here](https://helm.fiftyone.ai/#enabling-fiftyone-teams-plugins)

## Creating your `teams-plugins` GPT container

VoxelGPT requires certain Python packages to be installed. You can see the
current requirements by running:

```shell
cat requirements.txt
```

If any of these Python packages are not available in your `teams-plugins`
containers, you'll need to add them or you can use the `fiftyone-app-gpt`
image included in the `voxel51` Docker Hub repository.

You must also ensure that a valid OpenAI API key
([create one](https://platform.openai.com/account/api-keys)) is available to
the containers via the `OPENAI_API_KEY` environment variable.

To use the Voxel51 image for Docker Compose, edit your
 `compose.override.yaml` to include:

```
services:
  env:
    OPENAI_API_KEY: Your OpenAI API Key here
  teams-plugins:
    image: voxel51/fiftyone-app-gpt:v1.3.0
```

For Helm, edit your `values.yaml` to include:

```
pluginsSettings:
  environment:
    OPENAI_API_KEY: Your OpenAI API Key here
  image:
    repository: voxel51/fiftyone-app-gpt
	tag: v1.3.0
```


## Uploading the code

You can upload VoxelGPT manually or via the
[Management SDK](https://docs.voxel51.com/teams/management_sdk.html).

### Manual upload

1.  Download a ZIP of the `main` branch
    [from this link](https://github.com/voxel51/voxelgpt/archive/refs/heads/main.zip)
2.  Follow
    [these instructions](https://docs.voxel51.com/teams/teams_plugins.html) to
    upload the ZIP via the UI

### Management SDK

You can also programmatically upload the plugin via the
[Management SDK](https://docs.voxel51.com/teams/management_sdk.html) if you
have exposed your `teams-api` service.

Instructions for exposing your `teams-api` service using docker compose
are [here](https://github.com/voxel51/fiftyone-teams-app-deploy/blob/main/docker/docs/expose-teams-api.md)

Instructions for exposing your API service using helm are
[here](https://helm.fiftyone.ai/docs/expose-teams-api.html)

```shell
wget https://github.com/voxel51/voxelgpt/archive/refs/heads/main.zip
```

```py
import fiftyone.management as fom

fom.upload_plugin("main.zip", overwrite=True)
```

Or, if you already have the repository cloned locally, you can use:

```py
fom.upload_plugin("/path/to/voxelgpt", optimize=True, overwrite=True)
```

### Troubleshooting

If you are seeing issues with the plugin not updating in the App:

-   Check the logs for any additional information
-   Restart your `teams-plugins` pods

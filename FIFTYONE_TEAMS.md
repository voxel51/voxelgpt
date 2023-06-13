# Deploying VoxelGPT to FiftyOne Teams

This guide explains how to upload the latest version of VoxelGPT to your
[FiftyOne Teams](https://voxel51.com/fiftyone-teams) deployment.

## Uploading the code

You can upload VoxelGPT manually or via the Management SDK.

### Manual upload

1.  Download a ZIP of the `main` branch
    [from this link](https://github.com/voxel51/voxelgpt/archive/refs/heads/main.zip)
2.  Follow
    [these instructions](https://docs.voxel51.com/teams/teams_plugins.html) to
    upload the ZIP via the UI

### Management SDK

You can also programmatically upload the plugin via the
[Management SDK](https://docs.voxel51.com/teams/management_sdk.html).

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

## Updating your plugin containers

VoxelGPT requires certain Python packages to be available in your
`teams-plugins` containers. You can see the current requirements by running:

```shell
cat requirements.txt
```

If any of these Python packages are not available in your `teams-plugins`
containers, you'll need to add them (or just use the `fiftyone-app-gpt` image
included in the `voxel51` Docker Hub repository).

You must also ensure that a valid OpenAI API key
([create one](https://platform.openai.com/account/api-keys)) is available to
the containers via the `OPENAI_API_KEY` environment variable.

### Docker Compose

If your Teams deployment uses Docker Compose, edit your `compose.override.yaml`
to include:

```
services:
  teams-plugins:
    env:
      OPENAI_API_KEY: Your OpenAI API Key here
    image: voxel51/fiftyone-app-gpt:v1.3.0
```

and then redeploy your `teams-plugins` service:

```
docker compose up -d
```

### Helm

If your Teams deployment uses Helm, edit your `values.yaml` to include:

```
pluginsSettings:
  environment:
    OPENAI_API_KEY: XXXXXXXX
  image:
    repository: voxel51/fiftyone-app-gpt
  tag: v1.3.0
```

and then redeploy your `teams-plugins` service:

```
helm upgrade fiftyone-teams-app voxel51/fiftyone-teams-app -f values.yaml
```

## Troubleshooting

### Enabling plugins

In order to use VoxelGPT (or any other plugin), you must have enabled plugins
for your FiftyOne Teams deployment:

-   Instructions for
    [Docker Compose](https://github.com/voxel51/fiftyone-teams-app-deploy/tree/main/docker#enabling-fiftyone-teams-plugins)
-   Instructions for
    [Helm](https://helm.fiftyone.ai/#enabling-fiftyone-teams-plugins)

### Exposing your `teams-api` service

In order to use the
[Management SDK](https://docs.voxel51.com/teams/management_sdk.html), you must
have exposed your `teams-api` service:

-   Instructions for
    [Docker Compose](https://github.com/voxel51/fiftyone-teams-app-deploy/blob/main/docker/docs/expose-teams-api.md)
-   Instructions for
    [Helm](https://helm.fiftyone.ai/docs/expose-teams-api.html)

### Plugin not updating

If you are seeing issues with the plugin not updating in the App after you
upload a new version:

-   Check the logs for any additional information
-   Try restarting your `teams-plugins` pods

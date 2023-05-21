"""
GPT plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import random
import sys
import traceback

import fiftyone.operators as foo
import fiftyone.operators.types as types


class add_sys_path(object):
    """Context manager that temporarily inserts a path to ``sys.path``."""

    def __init__(self, path, index=0):
        self.path = path
        self.index = index

    def __enter__(self):
        sys.path.insert(self.index, self.path)

    def __exit__(self, *args):
        try:
            sys.path.remove(self.path)
        except:
            pass


class CreateViewWithGPT(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="create_view_with_gpt",
            label="Create View with GPT",
            execute_as_generator=True,
            unlisted=True,
        )

    @property
    def resolve_inputs(self):
        inputs = types.Object()
        inputs.str("query", label="Query", required=True)
        inputs.define_property("history", types.List(types.Object()))
        return types.Property(inputs)

    async def execute(self, ctx):
        # ideal it should overwrite anything added to the view after the session
        # started
        sample_collection = ctx.dataset
        # if ctx.view is not None:
        #     sample_collection = ctx.view
        # else:
        #     sample_collection = ctx.dataset

        query = ctx.params["query"]
        message_history = ctx.params["history"]
        chat_history = [item["content"] for item in message_history]

        # should this be needed?
        chat_history.append(query)

        try:
            with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
                from gpt_view_generator import ask_gpt_generator

                for response in ask_gpt_generator(
                    sample_collection, query, chat_history=chat_history
                ):
                    type = response["type"]
                    data = response["data"]

                    if type == "view":
                        yield self.view(ctx, data)
                    elif type == "log":
                        yield self.log(ctx, data)
        except Exception as e:
            yield self.error(ctx, dict(exception=e))
        finally:
            yield self.done(ctx)

    def view(self, ctx, data):
        view = data["view"]
        return ctx.trigger("set_view", params=dict(view=view._serialize()))

    def show_message(self, ctx, content, viewType):
        outputs = types.Object()
        outputs.str("message", view=viewType)

        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(
                outputs=types.Property(outputs).to_json(),
                data=dict(message=content),
                content=content,
            ),
        )

    def markdown(self, ctx, src):
        return self.show_message(ctx, src, types.MarkdownView())

    def log(self, ctx, data):
        message = data["message"]
        return self.markdown(ctx, message)

    def error(self, ctx, data):
        exception = data["exception"]

        message = str(exception)
        trace = traceback.format_exc()

        return self.show_message(ctx, message, types.ErrorView(label=message, description=trace))

    def done(self, ctx):
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(done=True),
        )


class OpenVoxelGPTPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_voxel_gpt_panel",
            label="Open VoxelGPT Panel",
            unlisted=True
        )

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(label="Open VoxelGPT",
                         icon="/assets/chatgpt.svg", prompt=False)
        )

    def execute(self, ctx):
        return ctx.trigger(
            "open_panel",
            params=dict(name="gpt_search", isActive=True),
        )


def register(p):
    p.register(CreateViewWithGPT)
    p.register(OpenVoxelGPTPanel)

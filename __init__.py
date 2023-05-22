"""
VoxelGPT plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import sys
import traceback

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types


# @todo replace with `fou.add_sys_path`
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


class AskVoxelGPT(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="ask_voxelgpt",
            label="Ask VoxelGPT",
            execute_as_generator=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "query",
            label="query",
            required=True,
            description="What would you like to view?",
        )
        return types.Property(inputs)

    async def execute(self, ctx):
        query = ctx.params["query"]
        if ctx.view is not None:
            sample_collection = ctx.view
        else:
            sample_collection = ctx.dataset

        messages = []

        try:
            with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
                from voxelgpt import ask_voxelgpt_generator

                for response in ask_voxelgpt_generator(
                    query, sample_collection, dialect="string"
                ):
                    type = response["type"]
                    data = response["data"]

                    if type == "view":
                        yield self.view(ctx, data["view"])
                    elif type == "message":
                        yield self.message(ctx, data["message"], messages)
        except Exception as e:
            yield self.error(ctx, e)

    def view(self, ctx, view):
        if view != ctx.view:
            return ctx.trigger("set_view", params=dict(view=view._serialize()))

    def message(self, ctx, message, messages):
        messages.append(message)

        outputs = types.Object()
        outputs.str("query", label="You")
        results = dict(query=ctx.params["query"])
        for i, msg in enumerate(messages, 1):
            field = "message" + str(i)
            outputs.str(field, label="VoxelGPT")
            results[field] = msg

        return ctx.trigger(
            "show_output",
            params=dict(
                outputs=types.Property(outputs).to_json(),
                results=results,
            ),
        )

    def error(self, ctx, exception):
        message = str(exception)
        trace = traceback.format_exc()
        view = types.Error(label=message, description=trace)
        outputs = types.Object()
        outputs.view("message", view)
        return ctx.trigger(
            "show_output",
            params=dict(
                outputs=types.Property(outputs).to_json()
            ),
        )


class AskVoxelGPTPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="ask_voxelgpt_panel",
            label="Ask VoxelGPT Panel",
            execute_as_generator=True,
            unlisted=True,
        )

    async def execute(self, ctx):
        query = ctx.params["query"]
        history = ctx.params.get("history", [])
        chat_history, sample_collection, orig_view = self._parse_history(
            ctx, history
        )

        try:
            with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
                from voxelgpt import ask_voxelgpt_generator

                for response in ask_voxelgpt_generator(
                    query,
                    sample_collection,
                    chat_history=chat_history,
                    dialect="markdown",
                ):
                    type = response["type"]
                    data = response["data"]

                    if type == "view":
                        if orig_view is not None:
                            message = (
                                "I'm remembering your previous view. Any "
                                "follow-up questions in this session will be "
                                "posed with respect to it"
                            )
                            yield self.message(
                                ctx, message, orig_view=orig_view
                            )

                        yield self.view(ctx, data["view"])
                    elif type == "message":
                        yield self.message(
                            ctx, data["message"], history=data["history"]
                        )
        except Exception as e:
            yield self.error(ctx, e)
        finally:
            yield self.done(ctx)

    def view(self, ctx, view):
        if view != ctx.view:
            return ctx.trigger("set_view", params=dict(view=view._serialize()))

    def message(self, ctx, message, **kwargs):
        return self.show_message(ctx, message, types.MarkdownView(), **kwargs)

    def error(self, ctx, exception):
        message = str(exception)
        trace = traceback.format_exc()
        view = types.Error(label=message, description=trace)
        return self.show_message(ctx, message, view)

    def done(self, ctx):
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(done=True),
        )

    def show_message(self, ctx, message, view_type, **kwargs):
        outputs = types.Object()
        outputs.str("message", view=view_type)
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(
                outputs=types.Property(outputs).to_json(),
                data=dict(message=message, **kwargs),
            ),
        )

    def _parse_history(self, ctx, history):
        if history is None:
            history = []

        # Parse chat history
        chat_history = []
        orig_view = None
        for item in history:
            if item["type"] == "outgoing":
                history = item.get("content", None)
            else:
                history = item.get("data", {}).get("history", None)
                _orig_view = item.get("data", {}).get("orig_view", None)
                if _orig_view is not None:
                    orig_view = _orig_view

            if history:
                chat_history.append(history)

        # If we found an `orig_view`, start from that instead
        if orig_view is not None:
            try:
                sample_collection = fo.DatasetView._build(
                    ctx.dataset, orig_view
                )
                return chat_history, sample_collection, None
            except:
                pass

        return chat_history, ctx.view, ctx.view._serialize()


class OpenVoxelGPTPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_voxelgpt_panel",
            label="Open VoxelGPT Panel",
            on_startup=True,
            unlisted=True,
        )

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Open VoxelGPT",
                icon="/assets/chatgpt.svg",
                prompt=False,
            ),
        )

    def execute(self, ctx):
        if ctx.dataset_name == "quickstart":
            return ctx.trigger(
                "open_panel",
                params=dict(name="voxelgpt", isActive=True, layout="horizontal")
            )


def register(p):
    p.register(AskVoxelGPT)
    p.register(AskVoxelGPTPanel)
    p.register(OpenVoxelGPTPanel)

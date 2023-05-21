"""
VoxelGPT plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
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
        view = types.ErrorView(label=message, description=trace)
        outputs = types.Object()
        outputs.str("message", view=view)
        return ctx.trigger(
            "show_output",
            params=dict(
                outputs=types.Property(outputs).to_json(),
                data=dict(message=message),
                # content=message,
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
        sample_collection = ctx.dataset
        chat_history = ctx.params.get("history", None)

        # if query == "Hello!":
        #     yield self.message(ctx, "Nice to meet you.")
        #     yield self.done(ctx)
        #     return
        # elif query == "Goodbye!":
        #     yield self.message(ctx, "See you soon!")
        #     yield self.done(ctx)
        #     return
        # elif query == "options":
        #     yield self.prompt_for_choices(ctx)
        #     yield self.done(ctx)
        #     return

        if chat_history:
            chat_history = [item["content"] for item in chat_history]

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
                        yield self.view(ctx, data["view"])
                    elif type == "message":
                        yield self.message(ctx, data["message"])
        except Exception as e:
            yield self.error(ctx, e)
        finally:
            yield self.done(ctx)

    def view(self, ctx, view):
        if view != ctx.view:
            return ctx.trigger("set_view", params=dict(view=view._serialize()))

    def message(self, ctx, message):
        return self.show_message(ctx, message, types.MarkdownView())

    def error(self, ctx, exception):
        message = str(exception)
        trace = traceback.format_exc()
        view = types.Error(label=message, description=trace)
        return self.show_message(ctx, message, view)

    def prompt_for_choices(self, ctx):
        outputs = types.Object()
        outputs.view(
            "hello",
            types.Button(
                label="Say Hello",
                space=1,
                operator=f"{self.plugin_name}/send_message_to_voxelgpt",
                params=dict(message="Hello!"),
            ),
        )
        outputs.view(
            "goodbye",
            types.Button(
                label="Say Goodbye",
                space=1,
                operator=f"{self.plugin_name}/send_message_to_voxelgpt",
                params=dict(message="Goodbye!"),
            ),
        )
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(
                outputs=types.Property(outputs).to_json(),
                data=dict(message=""),
                content="",
            ),
        )

    def done(self, ctx):
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(done=True),
        )

    def show_message(self, ctx, message, view_type):
        outputs = types.Object()
        outputs.str("message", view=view_type)
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(
                outputs=types.Property(outputs).to_json(),
                data=dict(message=message),
                content=message,
            ),
        )


class OpenVoxelGPTPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_voxelgpt_panel",
            label="Open VoxelGPT Panel",
            # unlisted=True,
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
        return ctx.trigger(
            "open_panel",
            params=dict(name="voxelgpt", isActive=True),
        )


def register(p):
    p.register(AskVoxelGPT)
    p.register(AskVoxelGPTPanel)
    p.register(OpenVoxelGPTPanel)

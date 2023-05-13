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


class AskGPT(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="ask_gpt",
            label="Ask GPT",
            execute_as_generator=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str(
            "query",
            label="query",
            required=True,
            description="Tell ChatGPT what you'd like to do",
        )

        inputs.str(
            "context",
            label="context",
            description="Context for this conversation",
            required=False,
        )

        return types.Property(inputs)

    async def execute(self, ctx):
        if ctx.view is not None:
            sample_collection = ctx.view
        else:
            sample_collection = ctx.dataset

        query = ctx.params["query"]
        context = ctx.params.get("context", None)
        if context:
            chat_history = chat_history.split("\n")
        else:
            chat_history = None

        logs = []

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
                        yield self.log(ctx, data, logs)
        except Exception as e:
            yield self.error(ctx, dict(exception=e))

    def view(self, ctx, data):
        view = data["view"]
        return ctx.trigger("set_view", params=dict(view=view._serialize()))

    def log(self, ctx, data, logs):
        message = data["message"]
        logs.append(message)

        outputs = types.Object()
        outputs.str("query", label="You")
        results = dict(query=ctx.params["query"])
        for i, msg in enumerate(logs, 1):
            field = "message" + str(i)
            outputs.str(field, label="ChatGPT")
            results[field] = msg

        return ctx.trigger(
            "show_output",
            params=dict(
                outputs=types.Property(outputs).to_json(),
                results=results,
            ),
        )

    def error(self, ctx, data):
        exception = data["exception"]

        outputs = types.Object()
        outputs.str("message", view=types.Error(label=str(exception)))
        outputs.str("trace", view=types.Error(label=traceback.format_exc()))

        return ctx.trigger(
            "show_output",
            params=dict(outputs=types.Property(outputs).to_json()),
        )


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
        return types.Property(inputs)

    async def execute(self, ctx):
        if ctx.view is not None:
            sample_collection = ctx.view
        else:
            sample_collection = ctx.dataset

        # @todo feed these as input
        query = "show me 10 random samples"
        chat_history = None

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

    def log(self, ctx, data):
        message = data["message"]

        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(message=message),
        )

    def error(self, ctx, data):
        exception = data["exception"]

        message = str(exception)
        trace = traceback.format_exc()
        msg = "%s\n\nTraceback\n%s" % (message, trace)

        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(message=msg),
        )

    def done(self, ctx):
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params=dict(done=True),
        )


def register(p):
    p.register(AskGPT)
    p.register(CreateViewWithGPT)

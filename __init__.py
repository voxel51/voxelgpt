"""
VoxelGPT plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import json
import os
import sys
import traceback

from bson import json_util

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .voxelgpt import ask_voxelgpt_generator
import db
class AskVoxelGPT(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="ask_voxelgpt",
            label="Ask VoxelGPT",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
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

    def execute(self, ctx):
        query = ctx.params["query"]
        sample_collection = ctx.view if ctx.view is not None else ctx.dataset
        messages = []

        inject_voxelgpt_secrets(ctx)

        try:
            streaming_message = None

            for response in ask_voxelgpt_generator(
                query,
                sample_collection=sample_collection,
                dialect="string",
                allow_streaming=True,
            ):
                type = response["type"]
                data = response["data"]

                if type == "view":
                    yield self.view(ctx, data["view"])
                elif type == "message":
                    kwargs = {}

                    if data["overwrite"]:
                        kwargs["overwrite_last"] = True

                    yield self.message(
                        ctx, data["message"], messages, **kwargs
                    )
                elif type == "streaming":
                    kwargs = {}

                    if streaming_message is None:
                        streaming_message = data["content"]
                    else:
                        streaming_message += data["content"]
                        kwargs["overwrite_last"] = True

                    yield self.message(
                        ctx, streaming_message, messages, **kwargs
                    )

                    if data["last"]:
                        streaming_message = None
        except Exception as e:
            yield self.error(ctx, e)

    def view(self, ctx, view):
        if view != ctx.view:
            return ctx.trigger(
                "set_view",
                params=dict(view=serialize_view(view)),
            )

    def message(self, ctx, message, messages, overwrite_last=False):
        if overwrite_last:
            messages[-1] = message
        else:
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
            params=dict(outputs=types.Property(outputs).to_json()),
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

    def execute(self, ctx):
        query = ctx.params["query"]
        history = ctx.params.get("history", [])
        chat_history, sample_collection, orig_view = self._parse_history(
            ctx, history
        )

        inject_voxelgpt_secrets(ctx)

        try:

            # Log user query
            table = db.table(db.UserQueryTable)
            ctx.params["query_id"] = table.insert_query(query)

            streaming_message = None

            for response in ask_voxelgpt_generator(
                query,
                sample_collection=sample_collection,
                chat_history=chat_history,
                dialect="markdown",
                allow_streaming=True,
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
                    kwargs = {}

                    if data["overwrite"]:
                        kwargs["overwrite_last"] = True

                    kwargs["history"] = data["history"]
                    yield self.message(ctx, data["message"], **kwargs)
                elif type == "streaming":
                    kwargs = {}

                    if streaming_message is None:
                        streaming_message = data["content"]
                    else:
                        streaming_message += data["content"]
                        kwargs["overwrite_last"] = True

                    if data["last"]:
                        kwargs["history"] = streaming_message

                    yield self.message(ctx, streaming_message, **kwargs)

                    if data["last"]:
                        streaming_message = None
                elif type == "warning":
                    yield self.warning(ctx, data["message"])
        except Exception as e:
            yield self.error(ctx, e)
        finally:
            yield self.done(ctx)

    def view(self, ctx, view):
        if view != ctx.view:
            return ctx.trigger(
                "set_view",
                params=dict(view=serialize_view(view)),
            )

    def message(self, ctx, message, **kwargs):
        return self.show_message(ctx, message, types.MarkdownView(), **kwargs)

    def warning(self, ctx, message):
        view = types.Warning(label=message)
        return self.show_message(ctx, message, view)

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
                query_id=ctx.params.get("query_id"),
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

        # If we have an `orig_view` into the same dataset, start from it
        if orig_view is not None and orig_view["dataset"] == ctx.dataset.name:
            try:
                view = deserialize_view(ctx.dataset, orig_view["stages"])
                return chat_history, view, None
            except:
                pass

        orig_view = dict(
            dataset=ctx.dataset.name,
            stages=serialize_view(ctx.view),
        )

        return chat_history, ctx.view, orig_view


class OpenVoxelGPTPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_voxelgpt_panel",
            label="Open VoxelGPT Panel",
            unlisted=True,
        )

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Open VoxelGPT",
                icon="/assets/icon-dark.svg",
                prompt=False,
            ),
        )

    def execute(self, ctx):
        ctx.trigger(
            "open_panel",
            params=dict(name="voxelgpt", isActive=True, layout="horizontal"),
        )


class OpenVoxelGPTPanelOnStartup(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_voxelgpt_panel_on_startup",
            label="Open VoxelGPT Panel",
            on_startup=True,
            unlisted=True,
        )

    def execute(self, ctx):
        open_on_startup = get_plugin_setting(
            ctx.dataset, self.plugin_name, "open_on_startup", default=False
        )
        if open_on_startup:
            ctx.trigger(
                "open_panel",
                params=dict(
                    name="voxelgpt", isActive=True, layout="horizontal"
                ),
            )


class VoteForQuery(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="vote_for_query",
            label="Vote For Query",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "query_id",
            label="query_id",
            required=True,
            description="User Query to Vote For",
        )
        inputs.enum(
            "vote",
            ["upvote", "downvote"],
            label="Vote",
            required=True,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        query_id = ctx.params["query_id"]
        vote = ctx.params["vote"]


        table = db.table(db.UserQueryTable)
        if vote == "upvote":
            table.upvote_query(query_id)
        elif vote == "downvote":
            table.downvote_query(query_id)
        else:
            raise ValueError(f"Invalid vote '{vote}'")


def get_plugin_setting(dataset, plugin_name, key, default=None):
    value = dataset.app_config.plugins.get(plugin_name, {}).get(key, None)

    if value is None:
        value = fo.app_config.plugins.get(plugin_name, {}).get(key, None)

    if value is None:
        value = default

    return value


def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def deserialize_view(dataset, stages):
    return fo.DatasetView._build(dataset, json_util.loads(json.dumps(stages)))


def inject_voxelgpt_secrets(ctx):
    try:
        api_key = ctx.secrets["OPENAI_API_KEY"]
    except:
        api_key = None

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key


def register(p):
    p.register(AskVoxelGPT)
    p.register(AskVoxelGPTPanel)
    p.register(OpenVoxelGPTPanel)
    p.register(OpenVoxelGPTPanelOnStartup)
    p.register(VoteForQuery)

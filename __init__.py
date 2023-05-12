import os
import sys

import fiftyone.operators as foo
import fiftyone.operators.types as types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_view_generator import ask_gpt_generator


class ChatGPTViewBuilder(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="chatgpt_view_builder",
            label="ChatGPT view builder",
            execute_as_generator=True,
            # unlisted=True,
        )

    ## testing only
    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str(
            "query",
            label="query",
            # required=True,
            description="Tell ChatGPT what you'd like to do",
        )
        inputs.str(
            "chat_history",
            label="chat_history",
            description="Chat history for this conversation",
            # required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        self._logs = []

        if ctx.view is not None:
            sample_collection = ctx.view
        else:
            sample_collection = ctx.dataset

        query = ctx.params["query"]

        chat_history = ctx.params.get("chat_history", None)
        if chat_history:
            chat_history = chat_history.split("\n")
        else:
            chat_history = None

        for response in ask_gpt_generator(
            sample_collection, query, chat_history=chat_history
        ):
            type = response["type"]
            data = response["data"]

            if type == "view":
                yield self.view(ctx, data)
            elif type == "log":
                yield self.log(ctx, data)
            elif type == "error":
                yield self.error(ctx, data)

    def view(self, ctx, data):
        view = data["view"]
        return ctx.trigger("set_view", {"view": view._serialize()})

    ## testing only
    def log(self, ctx, data):
        message = data["message"]

        self._logs.append(message)
        msg = "\n".join(self._logs)

        outputs = types.Object()
        return ctx.trigger(
            "show_output",
            dict(
                outputs=types.Property(outputs).to_json(),
                results={"message": msg},
            ),
        )

    ## testing only
    def error(self, ctx, data):
        message = data["message"]
        trace = data["trace"]

        msg = "%s\n\nTraceback\n%s" % (message, trace)

        outputs = types.Object()
        outputs.str("msg", view=types.Error(label=msg))
        return ctx.trigger(
            "show_output",
            dict(outputs=types.Property(outputs).to_json()),
        )

    """
    ## production
    def log(self, ctx, data):
        message = data["message"]
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params={"message": message},
        )

    ## production
    def error(self, ctx, data):
        message = data["message"]
        trace = data["trace"]  # @todo use

        # @todo format as error?
        return ctx.trigger(
            f"{self.plugin_name}/show_message",
            params={"message": message},
        )
    """


def register(p):
    p.register(ChatGPTViewBuilder)

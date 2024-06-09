"""
Workspace Inspection Agent

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
from typing import List, Dict, Any

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import fiftyone as fo

# pylint: disable=relative-beyond-top-level
from .utils import get_cache, PROMPTS_DIR, _build_agent_executor_chain, gpt_4o

WORKSPACE_INSPECTION_PATH = os.path.join(
    PROMPTS_DIR, "workspace_inspection.txt"
)


def _create_workspace_agent_executor():
    tools = make_workspace_inspection_tools()
    return _build_agent_executor_chain(
        gpt_4o, tools, WORKSPACE_INSPECTION_PATH
    )


def run_workspace_inspection_query(query):
    def workspace_inspection_func(info):
        query = info["query"]
        response = _create_workspace_agent_executor().invoke({"input": query})
        return response

    workspace_runnable = RunnableLambda(workspace_inspection_func)
    return workspace_runnable.invoke({"query": query})["output"]


def make_workspace_inspection_tools():
    import fiftyone.operators as foo
    import fiftyone.plugins as fop

    @tool
    def list_datasets() -> List[str]:
        """Lists the names of the datasets in the workspace."""
        return fo.list_datasets()

    @tool
    def get_fiftyone_config() -> Dict[str, Any]:
        """Returns the current configuration of the FiftyOne library."""
        return fo.config

    @tool
    def get_fiftyone_app_config() -> Dict[str, Any]:
        """Returns the current configuration of the FiftyOne App."""
        return fo.app_config

    @tool
    def list_plugins() -> List[str]:
        """Lists the names of the plugins in the workspace."""
        return [p.name for p in fop.list_plugins()]

    @tool
    def list_enabled_plugins() -> List[str]:
        """Lists the names of the enabled plugins in the workspace."""
        return [p.name for p in fop.list_enabled_plugins()]

    @tool
    def list_disabled_plugins() -> List[str]:
        """Lists the names of the disabled plugins in the workspace."""
        return [p.name for p in fop.list_disabled_plugins()]

    @tool
    def find_plugin(plugin_name: str) -> str:
        """Finds the plugin on disk with the specified name."""
        path = fop.find_plugin(plugin_name)
        return path if path is not None else "Not found"

    @tool
    def list_operators_in_plugin(plugin_name: str) -> List[str]:
        """Lists the names of the operators in the specified plugin."""
        return fop.get_plugin(plugin_name).operators

    @tool
    def get_plugin_description(plugin_name: str) -> str:
        """Returns the description of the specified plugin."""
        return fop.get_plugin(plugin_name).description

    @tool
    def get_plugin_version(plugin_name: str) -> str:
        """Returns the version of the specified plugin."""
        version = fop.get_plugin(plugin_name).version
        return version if version is not None else "Not available"

    @tool
    def get_plugin_author(plugin_name: str) -> str:
        """Returns the author of the specified plugin."""
        author = fop.get_plugin(plugin_name).author
        return author if author is not None else "Not available"

    @tool
    def get_plugin_url(plugin_name: str) -> str:
        """Returns the URL of the specified plugin."""
        url = fop.get_plugin(plugin_name).url
        return url if url is not None else "Not available"

    @tool
    def get_plugin_license(plugin_name: str) -> str:
        """Returns the license of the specified plugin."""
        license = fop.get_plugin(plugin_name).license
        return license if license is not None else "Not available"

    @tool
    def get_plugin_fiftyone_compatibility(plugin_name: str) -> str:
        """Returns the FiftyOne compatibility version of the specified plugin."""
        compatibility = fop.get_plugin(plugin_name).fiftyone_compatibility
        return compatibility if compatibility is not None else "Not available"

    @tool
    def list_operators() -> List[str]:
        """Lists the names of the operators in the workspace."""
        return foo.list_operators()

    @tool
    def list_builtin_operators() -> List[str]:
        """Lists the names of the built-in operators in the workspace."""
        return [o.config.name for o in foo.builtin.BUILTIN_OPERATORS]

    @tool
    def get_operator_description(operator_name: str) -> str:
        """Returns the description of the specified operator."""
        try:
            label = foo.get_operator(operator_name).config.label
            return label
        except:
            if operator_name in list_builtin_operators():
                for o in foo.builtin.BUILTIN_OPERATORS:
                    if o.config.name == operator_name:
                        return o.config.label

        return "Not found"

    @tool
    def get_operator_info(operator_name: str) -> Dict[str, Any]:
        """Returns the info of the specified operator."""
        try:
            return foo.get_operator(operator_name).config
        except:
            if operator_name in list_builtin_operators():
                for o in foo.builtin.BUILTIN_OPERATORS:
                    if o.config.name == operator_name:
                        return o.config

        return {}

    return [
        list_datasets,
        get_fiftyone_config,
        get_fiftyone_app_config,
        list_plugins,
        list_enabled_plugins,
        list_disabled_plugins,
        find_plugin,
        list_operators_in_plugin,
        get_plugin_description,
        get_plugin_version,
        get_plugin_author,
        get_plugin_url,
        get_plugin_license,
        get_plugin_fiftyone_compatibility,
        list_operators,
        list_builtin_operators,
        get_operator_description,
        get_operator_info,
    ]

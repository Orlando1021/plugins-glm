# -*- coding: utf-8 -*-


from plugin_service.plugins.tool import Tool


def normalize_plugin_name(name: str) -> str:
    if name in ("AMiner", 'aminer'):
        return 'aminer'
    raise NotImplementedError()


PLUGIN_REGISTRY = {}

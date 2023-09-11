# -*- coding: utf-8 -*-


import json
import random
from collections.abc import Mapping
import logging
import requests

#from plugin_service.logging import logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _compose_tool_usage(name: str, parameters: Mapping) -> str:
    js_obj = {'工具': name, '参数': parameters}
    return f"使用工具：{json.dumps(js_obj, ensure_ascii=False)}"


class Tool(object):
    def __init__(self, url: str) -> None:
        if url.endswith('/'):
            url = url[:-1]
        response = requests.get(url + '/.well-known/ai-plugin.json')
        self.config = response.json()
        self.url = url

        self.name = self.config['name']
        self.description = self.config['description']
        self.return_description = self.config['return']
        self.parameters = self.config['parameters']
        self.examples = self.config['examples']
        self.full_example = self.config.get('one_shot_example', None)

    def __call__(self, parameters: Mapping) -> str:
        logger.info(self.url)
        logger.info(parameters)
        json_body = {
            '函数': parameters['函数'],
            '函数参数': {'parameters': parameters['函数参数']},
        }
        print('json_body:{}'.format(json_body))
        response = requests.post(self.url, json=json_body)
        print("L40 tool:{}".format(response))
        if response.status_code != 200:
            return f"调用工具发生错误：{response.text}\n"
        return response.json()

    @property
    def full_documentation(self) -> str:
        parameter_str = '\n'.join(
            [
                "{}{}: {}".format(
                    item['name'],
                    ' (Optional)' if 'optional' in item else '',
                    item['description'],
                )
                for item in self.parameters
            ]
        )
        example = random.choice(self.examples)
        example_str = "[示例输入]:{} ".format(_compose_tool_usage(self.name, example['input']))
        
        return "工具描述：{}\n工具返回：{}\n[参数]\n{}\n{}请返回json格式的结果".format(
            self.description,
            self.return_description,
            parameter_str,
            example_str,
        )

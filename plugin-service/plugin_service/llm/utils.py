# -*- coding: utf-8 -*-


import json
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterator, List, Optional, Tuple

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from plugin_service.llm import prompts
# from plugin_service.logging import llm_logger as logger
from plugin_service.plugins import PLUGIN_REGISTRY, normalize_plugin_name
import logging

logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)

def extract_json(s: str) -> Optional[Dict[str, Any]]:
    end_idx = -1
    for idx in range(len(s)):
        if s[idx] == '}':
            try:
                obj = json.loads(s[: idx + 1])
                end_idx = idx
            except json.JSONDecodeError:
                pass
    if end_idx >= 0:
        obj = json.loads(s[: end_idx + 1])
        return obj
    return None


def parse_reply(reply: str) -> Dict[str, Any]:
    # normalizes reply
    reply = reply.replace('：', ':')
    lines = reply.splitlines()
    reply = '\n'.join(filter(len, map(lambda l: l.strip(), lines)))

    reply_json = {'action': None, 'thought': None}
    if '思考:' in reply:
        sub_reply = reply[reply.find('思考:') + 3 :]
        if '思考:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('思考:')]
        if '使用工具:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('使用工具:')]
        if '提交回复:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('提交回复:')]
        reply_json['thought'] = sub_reply.strip()

    if '使用工具:' in reply:
        sub_reply = reply[reply.find('使用工具:') + 5 :]
        if '思考:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('思考:')]
        if '使用工具:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('使用工具:')]
        if '提交回复:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('提交回复:')]
        tool_json = extract_json(sub_reply)
        if tool_json is not None:
            reply_json['action'] = {
                'type': 'tool',
                'content': {
                    '工具': tool_json['工具'],
                    '参数': tool_json['参数'],
                },
            }
    elif '提交回复:' in reply:
        sub_reply = reply[reply.find('提交回复:') + 5 :]
        if '思考:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('思考:')]
        if '使用工具:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('使用工具:')]
        if '提交回复:' in sub_reply:
            sub_reply = sub_reply[: sub_reply.find('提交回复:')]
        reply_json['action'] = {
            'type': 'response',
            'content': sub_reply.strip(),
        }
    return reply_json



def query(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    plugin_names: Sequence[str],
    query: str,
    history: Optional[Sequence[Tuple[str, str]]] = None,
    context: Optional[Mapping[str, str]] = None,
    **kwargs: Mapping[str, Any],
) -> List[Tuple[str, str]]:
    if history is None:
        history = []
    if context is None:
        context = {}

    plugin_docs = []
    plugin_idx = 1
    
    # 获取插件文档
    for name in plugin_names:
        print("PLUGIN_REGISTRY:{}\n".format(PLUGIN_REGISTRY))
        if name in PLUGIN_REGISTRY:
            plugin = PLUGIN_REGISTRY[name]
            plugin_docs.append(
                f'{plugin_idx}. {plugin.name}\n{plugin.full_documentation}'
            )
            
            
    # 构建指令和上下文信息
    full_doc = '\n'.join(plugin_docs)
    print("full_doc:{} from L113".format(full_doc))
    instruction = prompts.INSTRUCTION.format(tools_str=full_doc) + '\n'
    location = context.get('location', None)
    time = context.get('time', None)
    if location and time:
        instruction += f'用户地点：{location}，当前时间：{time}\n'

    # 构建消息历史
    round_idx = 0
    flatten_history = []
    for query, reply in history:
        flatten_history.append(query.strip())
        flatten_history.append(reply.strip())
        if query.startswith('[Round '):
            round_idx += 1
    message = instruction
    message += '\n\n'.join(flatten_history)
    message += f'\n[Round {round_idx + 1}]\n问：\n{query}\n答：\n'

    cur_history = history + [(query, None)]
    cnt = 0
    while True:
        inputs = tokenizer([message], return_tensors='pt').to(model.device)
        if inputs.input_ids.shape[1] > 2048:
            cur_history[-1] = (
                cur_history[-1][0],
                "已经超过模型能处理的最大长度，请停止当前对话，清空重新开始",
            )
            return cur_history[-1][1]
        kwargs.update(**inputs)
        #outputs = model.generate
        
        for outputs in model.stream_generate(
            tokenizer=tokenizer, max_length=2048, **kwargs
        ):
            outputs = outputs.tolist()[0][len(inputs['input_ids'][0]) :]
        response = tokenizer.decode(outputs)
        
        # 这里为什么只是思考？
        print("L152 response=====================\n{} \n========================".format(response))
        cur_history[-1] = (cur_history[-1][0], response)
        # yield cur_history
        message += response + '\n'
        # 根据respnse提取action
        action = parse_reply(response)
        # 这里就没往下执行了
        print("L160 action:{}".format(action))
        if 'action' in action and 'type' in action['action']:
            if action['action']['type'] == 'response':
                break
            elif action['action']['type'] == 'tool':
                # tool_json = json.dumps(
                #     action['action']['content'], ensure_ascii=False
                # )
                tool = PLUGIN_REGISTRY[
                    normalize_plugin_name(action['action']['content']['工具'])
                ]
                ret = tool(action['action']['content']['参数'])
                message += f"下面是工具返回的结果：\n{ret}\n"
                print('L173 下面是工具返回的结果：:{}'.format(message))
                cur_history.append((ret, None))
                print("L175 cur_history:{}".format(cur_history))
        cnt += 1
        if cnt == 10:
            break
    logger.info(cur_history)
    return cur_history[-1][1]


def stream_query(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    plugin_names: Sequence[str],
    query: str,
    history: Optional[Sequence[Tuple[str, str]]] = None,
    context: Optional[Mapping[str, str]] = None,
    **kwargs: Mapping[str, Any],
) -> Iterator[str]:
    if history is None:
        history = []
    if context is None:
        context = {}

    plugin_docs = []
    plugin_idx = 1
    for name in plugin_names:
        if name in PLUGIN_REGISTRY:
            plugin = PLUGIN_REGISTRY[name]
            plugin_docs.append(
                f'{plugin_idx}. {plugin.name}\n{plugin.full_documentation}'
            )
    full_doc = '\n'.join(plugin_docs)
    instruction = prompts.INSTRUCTION.format(tools_str=full_doc) + '\n'
    location = context.get('location', None)
    time = context.get('time', None)
    if location and time:
        instruction += f'用户地点：{location}，当前时间：{time}\n'

    round_idx = 0
    flatten_history = []
    for query, reply in history:
        flatten_history.append(query.strip())
        flatten_history.append(reply.strip())
        if query.startswith('[Round '):
            round_idx += 1
    message = instruction
    message += '\n\n'.join(flatten_history)
    message += f'\n[Round {round_idx + 1}]\n问：\n{query}\n答：\n'

    cur_history = history + [(query, None)]
    cnt = 0
    while True:
        inputs = tokenizer([message], return_tensors='pt').to(model.device)
        if inputs.input_ids.shape[1] > 2048:
            cur_history[-1] = (
                cur_history[-1][0],
                "已经超过模型能处理的最大长度，请停止当前对话，清空重新开始",
            )
            yield cur_history[-1][1]
        kwargs.update(**inputs)
        response = ''
        for outputs in model.stream_generate(
            tokenizer=tokenizer, max_length=2048, **kwargs
        ):
            outputs = outputs.tolist()[0][len(inputs['input_ids'][0]) :]
            response = tokenizer.decode(outputs)
            if response.startswith('提交回复') and len(response) > 5:
                yield response[5:]

        cur_history[-1] = (cur_history[-1][0], response)
        # yield cur_history
        message += response + '\n'
        # extracts action from reply
        action = parse_reply(response)
        logger.info('message:\n%s', message)
        logger.info('response:\n%s', response)
        logger.info('action:\n%s', action)
        if 'action' in action and 'type' in action['action']:
            if action['action']['type'] == 'response':
                break
            elif action['action']['type'] == 'tool':
                # tool_json = json.dumps(
                #     action['action']['content'], ensure_ascii=False
                # )
                tool = PLUGIN_REGISTRY[
                    normalize_plugin_name(action['action']['content']['工具'])
                ]
                ret = tool(action['action']['content']['参数'])
                if ret.startswith('调用工具发生错误'):
                    cur_history.append((ret, "抱歉，工具调用异常，无法获取相关信息。"))
                    break
                yield ret
                message += f"下面是工具返回的结果：\n{ret}\n"
                cur_history.append((ret, None))
        cnt += 1
        if cnt == 10:
            break
    logger.info(cur_history)
    response = cur_history[-1][1]
    if response.startswith('提交回复'):
        response = response[5:]
    yield response

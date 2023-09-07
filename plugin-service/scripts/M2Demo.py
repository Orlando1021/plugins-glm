mport json, os, sys, re, requests, random
import os, sys
import platform
import signal
import argparse
import gradio as gr
import random
from datetime import datetime, timedelta
import time
from chatglm6b_v10.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm6b_v10.tokenization_chatglm import ChatGLMTokenizer
import concurrent.futures

chkp_path = './output'
tags = ['function_v2']
lrs = ['2e-5']
tool_name2id = {
    '搜索引擎': 'google_search',
    'AMiner': 'aminer',
    '高考信息': 'gaokao',
    '天气预报': 'xj_weather',
    'CogView': 'cogview',
    'Stock': 'stock',
    '天眼查': 'tian_yan_cha'
}
tool_site = 'http://localhost:9003/'
LOAD_PROC_NUM = 32

# tuple2path = {}
# name2model = {}
# for tag in tags:
#     for lr in lrs:
#         if os.path.exists(os.path.join(chkp_path, tag + '_' + lr)):
#             dirs = os.listdir(os.path.join(chkp_path, tag + '_' + lr))
#             for directory in dirs:
#                 if directory.startswith('checkpoint-'):
#                     step = int(directory.split('-')[-1])
#                     tuple2path[(tag, lr, str(step))] = os.path.join(chkp_path, tag + '_' + lr, directory)
# x = 0
# for tp in tuple2path:
#     checkpoint_path = tuple2path[tp]
#     model = ChatGLMForConditionalGeneration.from_pretrained(
#         checkpoint_path
#     ).half().to('cuda:{}'.format(x % 8))
#     model = model.eval()
#     s = '标签：{}，学习率：{}，步数：{}'.format(tp[0], tp[1], tp[2])
#     name2model[s] = model
#     x += 1


def load_model(tag, lr, step, x):
    checkpoint_path = tuple2path[(tag, lr, str(step))]
    model = ChatGLMForConditionalGeneration.from_pretrained(checkpoint_path).half().to(f'cuda:{(x + 6) % 8}')
    model = model.eval()
    s = f'标签：{tag}，学习率：{lr}，步数：{step}'
    return s, model

tuple2path = {}
name2model = {}
for tag in tags:
    for lr in lrs:
        if os.path.exists(os.path.join(chkp_path, tag + '_' + lr)):
            dirs = os.listdir(os.path.join(chkp_path, tag + '_' + lr))
            for directory in dirs:
                if directory.startswith('checkpoint-'):
                    step = int(directory.split('-')[-1])
                    tuple2path[(tag, lr, str(step))] = os.path.join(chkp_path, tag + '_' + lr, directory)

# Your existing code...

# Create a list of model parameters to load
models_to_load = []
for idx, tp in enumerate(tuple2path):
    tag, lr, step = tp
    models_to_load.append((tag, lr, step, idx))

# Define a function to load the models in parallel
def load_models_parallel(models):
    loaded_models = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=LOAD_PROC_NUM) as executor:
        results = executor.map(load_model, *zip(*models))
        for result in results:
            loaded_models.append(result)
    return loaded_models

# Load the models in parallel
loaded_models = load_models_parallel(models_to_load)

# Store the loaded models in the name2model dictionary
for s, model in loaded_models:
    name2model[s] = model



lines = open('../../3_answer_generation/Plugin-Annotation/cities.txt').readlines()
cities = []
for line in lines:
    cities.append(line.strip())

def get_additional_info():
    return '用户地点：{}，'.format(random.choice(cities)) + '当前时间：{}，{}'.format((datetime.now() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'), ''.join(['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][datetime.now().weekday()]))

def compose_tool_usage(name: str, parameters: dict):
    js = {
        '工具': name,
        '参数': parameters
    }
    return "使用工具：%s"%(json.dumps(js, ensure_ascii=False))

class Tool:
    # TODO: adapt to more APIs in the Internet, now is used for local toy APIs
    def __init__(self, url) -> None:
        if url.endswith("/"):
            url = url[:-1]
        res = requests.get(url + "/.well-known/ai-plugin.json")
        if res.status_code != 200:
            raise Exception("Invalid tool server")
        self.config = res.json()
        self.url = url

        self.name = self.config["name"]
        self.description = self.config["description"]
        self.return_description = self.config["return"]
        self.parameters = self.config["parameters"]
        self.examples = self.config["examples"]
        self.full_example = self.config["one_shot_example"] if ("one_shot_example" in self.config) else None

    def __call__(self, parameters) -> str:
        res = requests.post(self.url, json=parameters)
        if res.status_code != 200:
            try:
                return "调用工具发生错误：\n" + res.json()
            except:
                return "调用工具发生错误：\n" + res.text
        return res.json()

    @property
    def full_documentation(self):
        parameter_str = "\n".join(["%s%s: %s"%(item["name"], " (Optional)" if ("optional" in item and item["optional"]) else "", item["description"]) for item in self.parameters])
        item = random.choice(self.examples)
        example_str = "[示例输入]\n%s" % (
            compose_tool_usage(self.name, item["input"])
        )
        return "\n工具描述：%s\n工具返回：%s\n[参数]\n%s\n%s"%(self.description, self.return_description, parameter_str, example_str)

tools = {}
for tool_id in tool_name2id.values():
    tools[tool_id] = Tool(tool_site + tool_id)

def get_tool(name):
    for tool in tools:
        if tools[tool].name == name:
            return tools[tool]

INSTRUCTION  = """你是一个能够调用各种外部工具的助理，可以通过各种工具来得到一些额外信息，比如天气，新闻，搜索，等等。每一轮操作中，你需要在下面两个操作中选择一个：

- 使用工具：你使用工具来解决问题。在这种情况下，你应该输入工具所需的参数，然后你将从工具接收输出。
- 提交回复：当你认为有可能回答这个问题时，立即将你的回答提交给用户。

下面是这两个操作的使用示意：

[工具使用]
思考：你关于调用工具的思考
使用工具：{{"工具":"工具名", "参数": {{"参数名": "参数值"}}}}

[提交回复]
思考：你关于提交回复的思考
提交回复：你最终提交给用户的回复

以下是你能够调用的工具：
{tools_str}
"""

def extract_json(s):
    end_index = -1
    for i in range(len(s)):
        if s[i] == '}':
            try:
                json_dict = json.loads(s[:i+1])
                end_index = i
                break
            except json.JSONDecodeError:
                pass

    if end_index >= 0:
        json_dict = json.loads(s[:end_index+1])
        return json_dict
    else:
        return None

def decode_assistant(ret:str):
    s = ret.strip().split('\n')
    sl = []
    for i in range(len(s)):
        if s[i] == '':
            continue
        sl.append(s[i])
    s = '\n'.join(sl)
    dic = {}
    if '思考：' in s:
        new_s = s[s.find('思考：') + 3:]
        if '思考：' in new_s:
            new_s = new_s[:new_s.find('思考：')]
        if '使用工具：' in new_s:
            new_s = new_s[:new_s.find('使用工具：')]
        if '提交回复：' in new_s:
            new_s = new_s[:new_s.find('提交回复：')]
        dic['thought'] = new_s.strip()
    if '使用工具：' in s:
        new_s = s[s.find('使用工具：') + 5:]
        if '思考：' in new_s:
            new_s = new_s[:new_s.find('思考：')]
        if '使用工具：' in new_s:
            new_s = new_s[:new_s.find('使用工具：')]
        if '提交回复：' in new_s:
            new_s = new_s[:new_s.find('提交回复：')]
        js = extract_json(new_s)
        if js is not None:
            dic['tool_usage'] = {
                '工具': js['工具'],
                '参数': js['参数']
            }
    elif '提交回复：' in s:
        new_s = s[s.find('提交回复：') + 5:]
        if '思考：' in new_s:
            new_s = new_s[:new_s.find('思考：')]
        if '使用工具：' in new_s:
            new_s = new_s[:new_s.find('使用工具：')]
        if '提交回复：' in new_s:
            new_s = new_s[:new_s.find('提交回复：')]
        dic['commit_response'] = new_s.strip()

    r = {
        "action": None,
        "thought": dic.get('thought', None)
    }
    if 'tool_usage' in dic:
        r['action'] = {
            'type': 'tool',
            'content': dic['tool_usage']
        }
    elif 'commit_response' in dic:
        r['action'] = {
            'type': 'response',
            'content': dic['commit_response']
        }
    return r

def check_round(history):
    num = 0
    for item in history:
        if item[0].startswith("[Round "):
            num += 1
    return num

def stream_query(query, tool_names, additional_info, model_selection, history):
    if history is None:
        history = []
    global name2model, tokenizer
    model = name2model[model_selection]
    used_tools = []
    for tool_name in tool_names:
        if tool_name in tool_name2id:
            used_tools.append(tools[tool_name2id[tool_name]])
    docs = ['{}. {}'.format(idx + 1, tool.name) + tool.full_documentation for idx, tool in enumerate(used_tools)]
    docs = '\n'.join(docs)
    instruction_this = INSTRUCTION.format(tools_str=docs) + '\n' + additional_info + '\n'
    tmp = []
    for item in history:
        tmp.append(item[0].strip())
        tmp.append(item[1].strip())

    round_num = check_round(history)
    this_query = '[Round {}]\n问：\n{}\n答：\n'.format(round_num + 1, query)

    record = history + [(this_query, None)]
    msg = instruction_this + '\n\n'.join(tmp) + '\n' + this_query
    while True:
        inputs = tokenizer([msg], return_tensors="pt").to(model.device)
        if inputs.input_ids.shape[1] > 2048:
            record[-1] = (record[-1][0], '已经超过模型能处理的最大长度，请停止当前对话，清空重新开始')
            yield record
            return
        for outputs in model.stream_generate(**inputs, max_length=2048, tokenizer=tokenizer):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            response = model.process_response(response)
            record[-1] = (record[-1][0], response)
            yield record
        print(response)
        msg += response
        action = decode_assistant(response.replace('“', '"').replace('”', '"').replace('""', '"'))
        if "action" in action and "type" in action["action"]:
            if action["action"]["type"] == "response":
                break
            elif action["action"]["type"] == "tool":
                record.append(("使用工具中...\n%s"%(json.dumps(action["action"]["content"], ensure_ascii=False)), None))
                yield record
                use_tool = get_tool(action["action"]["content"]["工具"])
                ret = use_tool(action["action"]["content"]["参数"])
        msg += "下面是工具返回的结果：\n{}".format(ret)
        record[-1] = ((ret, None))
        yield record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9918)
    args = parser.parse_args()

    tokenizer = ChatGLMTokenizer.from_pretrained(
        'chatglm6b_v10'
    )

    with gr.Blocks() as demo:
        model_list = sorted(list(name2model.keys()))
        gr.HTML("<h1> Tool Augmented ChatGLM-6B </h1><p>为避免出现长度过长的问题，请最多选择两个工具。对话轮次一般可持续2-5轮。</p>")
        chatbot = gr.Chatbot()
        tool_choice = gr.CheckboxGroup(list(tool_name2id.keys()), value=list(tool_name2id.keys())[0], label="Tool", interactive=True)
        model_selection = gr.Radio(model_list, label="模型选择", value=model_list[0])
        additional_info = gr.Textbox(label="additional infomation", value=get_additional_info())
        msg = gr.Textbox(label="Input")
        clear = gr.Button("清空对话")
        refresh = gr.Button("刷新用户信息")

        def update_info():
            return gr.Textbox.update(value=get_additional_info(), interactive=True)

        msg.submit(stream_query, [msg, tool_choice, additional_info, model_selection, chatbot], [chatbot], show_progress=True)

        clear.click(lambda: None, None, chatbot, queue=False)
        refresh.click(fn=update_info, inputs=None, outputs=[additional_info], show_progress=False, queue=False)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=True, auth=("zphz", "plugin_test"))

# -*- coding: utf-8 -*-


import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModel, AutoTokenizer

from plugin_service import llm
from plugin_service.logging import logger
from plugin_service.plugins import PLUGIN_REGISTRY, Tool

load_dotenv(verbose=True)
MODEL_DIR = Path(os.environ['MODEL_DIR']).expanduser().resolve()
CITY_FILE = Path(os.environ['CITY_FILE']).expanduser().resolve()
WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
TIME_FMT = '%Y-%m-%d %H:%M:%S'
TOOL_HOME = 'http://localhost:9002'
TOOL_URLS = {'aminer': TOOL_HOME + '/aminer'}
model = None
tokenizer = None
cities = None
plugin_registry = {}


if torch.cuda.is_available():
    cuda_device_id = int(os.environ.get('GPU_ID', 0))
    torch_device = f'cuda:{cuda_device_id}'
else:
    torch_device = 'cpu'


class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]] = []
    temperature: float = 0.8
    top_p: float = 0.8


class ChatResponse(BaseModel):
    response: str
    duration: str


class MetaResponse(BaseModel):
    model_version: str
    generated_token_num: int


class ChatStreamResponse(BaseModel):
    event: str
    id: str
    data: str
    meta: Optional[MetaResponse]


def empty_cuda_cache() -> None:
    if torch_device.startswith("cuda:"):
        with torch.cuda.device(torch_device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _get_location() -> str:
    return random.choice(cities)


def _get_time() -> str:
    now = datetime.now()
    return f'{WEEKDAYS[now.weekday()]}，{now.strftime(TIME_FMT)}'


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not MODEL_DIR.is_dir():
        raise RuntimeError()
    global tokenizer, model, cities

    logger.info("Loading model from '%s'...", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True
    ).half()
    if torch_device.startswith('cuda:'):
        with torch.cuda.device(torch_device):
            model = model.cuda()
    else:
        raise NotImplementedError()
    model = model.eval()

    for name, url in TOOL_URLS.items():
        PLUGIN_REGISTRY[name] = Tool(url)
    logger.info("Loaded %d plugins", len(PLUGIN_REGISTRY))

    cities = []
    with open(CITY_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            cities.append(line.strip())
    logger.info("Loaded %d cities.", len(cities))

    yield
    empty_cuda_cache()
    tokenizer = None
    model = None
    cities = None


app = FastAPI(lifespan=lifespan)


@app.get('/')
async def home() -> str:
    return "chatglm-6b (fp16, plugin enhanced)"


@app.post('/chat/')
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("request:\n%s\n", request)
    start = time.perf_counter()
    try:
        context = {'location': _get_location(), 'time': _get_time()}

        response = llm.utils.query(
            model,
            tokenizer,
            ['aminer'],
            request.message,
            history=request.history,
            context=context,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as e:
        response = str(e)
    duration = time.perf_counter() - start
    empty_cuda_cache()
    return ChatResponse(response=response, duration=f'{duration:.3f}s')


@app.post('/stream_chat/')
async def stream_chat(
    request: ChatRequest,
) -> EventSourceResponse:
    logger.info("request:\n%s\n", request)

    async def event_generator(
        request: ChatRequest, location: str, time: str, msg_uuid: str
    ):
        message = None
        for reply in llm.utils.stream_query(
            model,
            tokenizer,
            ['aminer'],
            request.message,
            history=request.history,
            context={'location': location, 'time': time},
            temperature=request.temperature,
            top_p=request.top_p,
        ):
            yield {'event': 'add', 'id': msg_uuid, 'data': reply}
            message = reply
        yield {
            'event': 'finish',
            'id': msg_uuid,
            'data': message,
            # 'meta': {'model_version': 'chatglm-6b'},
        }

    return EventSourceResponse(
        event_generator(request, _get_location(), _get_time(), uuid.uuid4())
    )

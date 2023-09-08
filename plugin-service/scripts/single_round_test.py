#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path

import torch
import typer
from transformers import AutoConfig, AutoModel, AutoTokenizer

app = typer.Typer()


@app.command()
def main(model_dir: str, gpu_id: int):
    model_dir = str(Path(model_dir).expanduser().resolve())
    if torch.cuda.is_available():
        torch_device = f'cuda:{gpu_id}'
    else:
        torch_device = 'cpu'

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, config=config, trust_remote_code=True
    ).half()
    if torch_device.startswith('cuda:'):
        with torch.cuda.device(torch_device):
            model = model.cuda()
    model = model.eval()

    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)


if __name__ == '__main__':
    app()

# plugins_glm: 为了接入线上插件，微调各种参数量的glm

## 📖 Introduction
为了接入线上插件，微调各种参数量的glm(glm-12b,glm2-6b)

[**晨晖的飞书文档**](https://lslfd0slxc.feishu.cn/docx/GgZYdKAuWoD6p6xxdVVcoumoncg)

# 😊 数据集
[**v1数据集**](https://huggingface.co/datasets/orlando1021/toolglm_v1)

v1数据集包含两个状态：
1. 请求体生成，包含思考和使用工具两个键值
    思考：思维链过程   
    使用工具：抽取出api call调用参数
2. 回答，将function calling的结果拼接到input最后
    思考：思维链过程   
    提交回复：根据fuction calling的结果回答


# 启动服务
```
# 启动工具server，启动端口默认为9002
cd /share/AgentGLM/Pipeline/1_server/Tool-Learning/src
python start_local_server.py


# 启动action_executer,该服务会访问9002端口。启动端口默认为30016
cd /share/AgentGLM/plugin-service/plugin_service
python main.py 

# 服务都启动后，尝试curl命令 （目前只开了aminer这个工具）
curl -X 'POST' 'http://localhost:30016/chat/' \
-H 'Content-Type: application/json' \
-d '{
    "message": "用aminer工具和pubInfo函数告诉我attention is all you need的作者是谁",
    "history": [],
    "temperature": 0.1,
    "top_p": 0.0
}'

```

# 搭建的测试环境
* 现有pipeline已在30016端口启动，随时根据聚合api的数据集微调6b/12b

* lagent的测试
```
# lagent是 InternLM实现的轻量级智能体项目，实现了rewoo/react智能体。
# 可以将现有执行请求的pipeline修改成lagent样式
python /share/lagent/lagent_test.py
```

* Toolbench
```
# Toolbench是刘知远团队实现的data generation和测评的pipeline，并提供了一版微调后的lamma-7b，已下载到开发机上
# lamma-7b的推理结果
cd /share/ToolBench
python ./toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path /share/models/ToolLLaMA-2-7b \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file toolllama_dfs_inference_result/0911 \
    --toolbench_key $TOOLBENCH_KEY
    
```

# sft脚本
bash run.sh
| glm-6b | glm-13b | 表头3 |
|-------|-------|-------|
| bash run.sh| 单元格2 | 单元格3 |
| 单元格1 | 单元格2 | 单元格3 |

已跑通6b，本周内13b跑通


# 💻 todo：
1. action_executor
根据晨晖推理代码和lagent重构action_executor

2. 如何衡量sft之后的llm
根据toolbench搭建api测试环境，评估pass rate
感觉sft之后的效果好像并不太好


# 🔥相关项目
[**lagent:llm based agents**](https://github.com/InternLM/lagent/blob/main/README_zh-CN.md)

[**Toolbench:pipeline和对api的测试环境**](https://github.com/OpenBMB/ToolBench)
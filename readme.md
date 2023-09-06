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


# sft脚本
bash run.sh
| glm-6b | glm-13b | 表头3 |
|-------|-------|-------|
| bash run.sh| 单元格2 | 单元格3 |
| 单元格1 | 单元格2 | 单元格3 |

已跑通6b，本周内13b跑通

# 💻 todo：
1. 如何执行生成的请求体
2. 如何衡量function calling的pass rate

# 🔥相关项目
[**lagent:llm based agents**](https://github.com/InternLM/lagent/blob/main/README_zh-CN.md)
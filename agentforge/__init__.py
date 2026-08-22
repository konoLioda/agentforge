"""
AgentForge 主包

基于 LangChain + LangGraph 的多智能体协作框架，主要能力：
- ReAct 推理智能体（Thought → Action → Observation → Final Answer）
- Planner → Worker → Reviewer 多智能体协作工作流
- 9 个工具链（搜索 / 计算 / 代码解释 / 文件 / 抓取 / API / 图片生成等）
- 短期记忆（对话历史）+ 长期记忆（Chroma 向量库）
- 多用户系统、FastAPI HTTP 接口、Streamlit Web 界面
"""

__version__ = "1.0.0"

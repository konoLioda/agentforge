"""
LangGraph 工作流编排

将 Planner → Worker → Reviewer 串联为完整的多智能体协作流程。
"""

import operator
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agentforge.agents.planner_agent import PlannerAgent, PlannerOutput
from agentforge.agents.reviewer_agent import ReviewerAgent, ReviewResult
from agentforge.agents.worker_agent import WorkerAgent
from config.logging_config import get_logger

logger = get_logger("workflow")


class WorkflowState(TypedDict):
    """工作流状态，在各节点间传递"""

    original_task: str  # 原始用户任务
    subtasks: List[dict]  # Planner 分解的子任务列表
    current_task_id: int  # 当前执行的子任务编号
    completed_outputs: Annotated[List[str], operator.add]  # 已完成的输出（累加）
    review_result: Optional[dict]  # 审查结果
    iteration: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数
    final_answer: str  # 最终答案


def create_workflow(max_iterations: int = 3) -> StateGraph:
    """
    创建多智能体协作工作流。

    流程：
    1. Planner 分解任务
    2. Worker 执行每个子任务（按依赖顺序）
    3. Reviewer 审查最终成果
    4. 如未通过，重新执行（最多 max_iterations 次）

    Args:
        max_iterations: 审查未通过时的最大重试次数

    Returns:
        LangGraph StateGraph 对象
    """
    planner = PlannerAgent()
    worker = WorkerAgent()
    reviewer = ReviewerAgent()

    def plan_node(state: WorkflowState) -> dict:
        """Planner 节点：分解任务"""
        logger.info("[workflow] 进入 planner 节点")
        task = state["original_task"]
        result: PlannerOutput = planner.think(task)
        logger.info(f"[workflow] planner 完成，分解出 {len(result.subtasks)} 个子任务")
        return {
            "subtasks": [t.model_dump() for t in result.subtasks],
            "current_task_id": 1,
            "iteration": state.get("iteration", 0) + 1,
        }

    def worker_node(state: WorkflowState) -> dict:
        """Worker 节点：执行当前子任务"""
        logger.info(f"[workflow] 进入 worker 节点，当前任务 id={state['current_task_id']}")
        subtasks = state["subtasks"]
        current_id = state["current_task_id"]
        completed = state.get("completed_outputs", [])

        # 找到当前任务
        current_task = None
        for t in subtasks:
            if t["id"] == current_id:
                current_task = t
                break

        if not current_task:
            return {"current_task_id": current_id + 1}

        # 获取依赖任务的输出作为上下文
        context_parts = []
        for t in subtasks:
            if t["id"] in current_task.get("depends_on", []):
                # 依赖任务的输出索引 = 任务 id - 1
                idx = t["id"] - 1
                if idx < len(completed):
                    context_parts.append(f"[任务{t['id']}结果]: {completed[idx]}")

        context = "\n".join(context_parts) if context_parts else None

        # 执行任务
        response = worker.think(current_task["description"], context)
        return {
            "completed_outputs": [response.content],
            "current_task_id": current_id + 1,
        }

    def review_node(state: WorkflowState) -> dict:
        """Reviewer 节点：审查最终成果"""
        logger.info("[workflow] 进入 reviewer 节点")
        all_outputs = state.get("completed_outputs", [])
        final_output = "\n\n".join(all_outputs)

        result: ReviewResult = reviewer.think(
            task=state["original_task"],
            output=final_output,
        )

        return {
            "review_result": result.model_dump(),
            "final_answer": final_output if result.approved else "",
        }

    def should_continue(state: WorkflowState) -> str:
        """判断是否需要继续执行下一个子任务"""
        subtasks = state["subtasks"]
        current_id = state["current_task_id"]

        if current_id <= len(subtasks):
            return "worker"

        return "review"

    def should_retry(state: WorkflowState) -> str:
        """判断审查未通过是否需要重试"""
        review = state.get("review_result", {})
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", max_iterations)

        if review.get("approved", False):
            return "end"

        if iteration < max_iter:
            return "plan"

        return "end"

    # 构建图
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", plan_node)
    graph.add_node("worker", worker_node)
    graph.add_node("reviewer", review_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", should_continue, {
        "worker": "worker",
        "review": "reviewer",
    })
    graph.add_conditional_edges("worker", should_continue, {
        "worker": "worker",
        "review": "reviewer",
    })
    graph.add_conditional_edges("reviewer", should_retry, {
        "plan": "planner",
        "end": END,
    })

    return graph


def run_workflow(task: str, max_iterations: int = 3) -> dict:
    """
    运行完整工作流。

    Args:
        task: 用户任务
        max_iterations: 最大重试次数

    Returns:
        工作流最终状态
    """
    graph = create_workflow(max_iterations)
    app = graph.compile()

    initial_state: WorkflowState = {
        "original_task": task,
        "subtasks": [],
        "current_task_id": 0,
        "completed_outputs": [],
        "review_result": None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_answer": "",
    }

    return app.invoke(initial_state)

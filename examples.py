import os
import logging
import time

# --- 配置日志，以便观察MNEMON内部的活动 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- 导入 LangChain 和 MNEMON 的核心组件 ---
# 在运行此脚本前，请确保您已设置DASHSCOPE_API_KEY环境变量
# export DASHSCOPE_API_KEY="your_api_key_here"
if "DASHSCOPE_API_KEY" not in os.environ:
    print("错误：请先设置您的 DASHSCOPE_API_KEY 环境变量。")
    exit()

from langchain_community.vectorstores import FAISS
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

from evermind import MemoryManager, MnemonConfig, InProcessTaskQueue, MemoryMetadata

API_KEY = "your-api-key"


def run_example():
    """运行一个完整的MNEMON SDK使用示例。"""
    print("--- 欢迎来到 MNEMON SDK 示例 (适配通义千问和Neo4j) ---")

    # 1. 初始化依赖组件
    #    - LLM: 切换为通义千问大模型
    #    - Embedding Model: 切换为通义千问文本向量模型
    #    - Vector Store: 依旧使用内存中的FAISS
    #    - Graph Store: 新增Neo4j知识图谱实例
    #    - Task Queue: 依旧使用简单的同步实现
    print("\n[步骤 1] 正在初始化依赖组件...")
    # 注意：您提到的 qwen3-235-nonthinking fp8 是一个非常新的模型，
    # 此处我们使用官方文档中稳定且强大的 qwen-max 模型作为示例。
    llm = ChatTongyi(
        model="qwen3-235b-a22b-instruct-2507", temperature=0.7, api_key=API_KEY
    )
    embedding_model = DashScopeEmbeddings(
        model="text-embedding-v4", dashscope_api_key=API_KEY
    )
    placeholder_metadata = MemoryMetadata(source_type="system_placeholder").model_dump()
    vector_store = FAISS.from_texts(
        ["初始化占位符"], embedding_model, metadatas=[placeholder_metadata]
    )

    # 初始化Neo4j图谱，连接到您本地Docker容器
    graph_store = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="albert-cui"
    )

    task_queue = InProcessTaskQueue()
    print("组件初始化完成。")

    # 2. 配置并实例化 MemoryManager
    #    我们使用默认配置，它启用了所有高级功能，包括知识图谱。
    print("\n[步骤 2] 正在配置并实例化 MemoryManager...")
    config = MnemonConfig()
    memory_manager = MemoryManager(
        config=config,
        vector_store=vector_store,
        llm=llm,
        embedding_model=embedding_model,
        graph_store=graph_store,  # <--- 将图谱实例传入
        task_queue=task_queue,
        initial_instructions=[
            "你是一个名为'阿尔法'的AI研究助手。",
            "你的目标是帮助用户分析和总结信息。",
            "你总是以专业和客观的口吻回答问题。",
        ],
    )
    print("MemoryManager 实例化完成。")

    # 3. 写入记忆 (Ingestion)
    print("\n[步骤 3] 正在写入几条记忆...")
    memory_manager.ingest(
        "项目'凤凰'的目标是开发下一代AI认知架构。",
        metadata={"source_type": "project_brief"},
    )
    memory_manager.ingest(
        "项目'凤凰'的主要负责人是李博士。", metadata={"source_type": "project_brief"}
    )
    memory_manager.ingest(
        "用户张三对项目'凤凰'的成本非常关心。", metadata={"source_type": "user_chat"}
    )
    memory_manager.ingest("我今天感觉不错。", metadata={"source_type": "user_chat"})
    print("记忆写入请求已发送。")

    # 等待后台任务处理完成（因为我们用的是同步队列，所以会立即完成）
    print("等待2秒以确保记忆处理完毕...")
    time.sleep(2)

    # 4. 触发一次元认知反思，让MNEMON学习知识
    print("\n[步骤 4] 正在手动触发一次元认知反思...")
    memory_manager.run_maintenance(run_reflection=True, run_health_check=False)
    print("元认知反思完成。现在知识图谱中应该有新的知识了。")
    print("您可以访问 http://localhost:7474 查看图谱。")

    # 5. 进行查询 (Query)
    print("\n" + "=" * 50)
    print("[步骤 5.1] 进行向量检索查询...")
    query1 = "用户张三关心什么？"
    print(f"查询: {query1}")
    result1 = memory_manager.query(query1, synthesize_answer=True)
    print(f"\n【合成的答案】:\n{result1.synthesized_answer}")

    print("\n" + "=" * 50)
    print("[步骤 5.2] 进行知识图谱查询...")
    query2 = "李博士和项目'凤凰'之间是什么关系？"
    print(f"查询: {query2}")
    result2 = memory_manager.query(query2, synthesize_answer=True)
    print(f"\n【合成的答案】:\n{result2.synthesized_answer}")
    print("\n【用于生成答案的溯源记忆】:")
    for mem in result2.retrieved_memories:
        # 图谱查询返回的结果在page_content里
        print(f"  - [来源: {mem.type}] {mem.content}")
    print("=" * 50)


if __name__ == "__main__":
    run_example()

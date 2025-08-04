"""
Evermind 使用示例

展示渐进式智能记忆系统的核心功能：
- 记忆录入与多粒度索引
- RR权重检索 (相关性 + 时效性)
- 上下文引子生成
- 实体关联追踪与查询
"""

import asyncio
import os
from typing import Tuple

# 导入依赖
from qdrant_client import QdrantClient
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

import evermind

# --- 配置区域 ---
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL")

BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_BASE_URL = os.getenv("BAILIAN_BASE_URL")


def setup_components() -> Tuple[BaseLanguageModel, Embeddings, QdrantClient]:
    """
    集中初始化并返回所有必要的组件。
    """
    # 使用智谱 GLM-4.5 作为语言模型
    llm = ChatOpenAI(
        model="glm-4.5-air",
        openai_api_base=ZHIPU_BASE_URL,
        openai_api_key=ZHIPU_API_KEY,
        temperature=0.7,
    )

    # 使用阿里百炼作为 Embedding 模型
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=BAILIAN_API_KEY,
        model="text-embedding-v4",
    )

    # 使用内存模式的 Qdrant 客户端，方便演示
    client = QdrantClient(":memory:")

    return llm, embeddings, client


async def run_basic_demo():
    """基础使用演示：展示记忆的录入和基本查询。"""
    print("\n>> 🧠 场景一：基础使用演示 <<")
    print("=" * 60)

    llm, embeddings, client = setup_components()

    # 1. 创建一个简化配置的记忆管理器
    memory_manager = evermind.create_simple_memory_manager(
        qdrant_client=client, llm=llm, embeddings=embeddings, namespace="basic_demo"
    )
    await memory_manager.initialize()

    # 2. 录入一系列相关的记忆
    print("\n📝 步骤 1: 录入记忆...")
    memories = [
        "项目Phoenix的负责人是李博士，这是一个AI认知架构研发项目。",
        "张三对项目Phoenix的预算很关心，担心超支。",
        "今天开会讨论了Phoenix项目的技术路线，决定采用Transformer架构。",
        "李博士提到Phoenix项目需要在6个月内完成第一个原型。",
        "我们团队在人工智能和机器学习方面有丰富经验。",
    ]
    for content in memories:
        memory_id = await memory_manager.ingest(content, process_immediately=True)
        print(f"  ✅ 记忆已录入: {content[:30]}... (ID: {memory_id[:8]})")

    # 3. 对记忆进行查询
    print("\n🔍 步骤 2: 进行查询...")
    query = "谁负责Phoenix项目？什么时候需要完成？"
    print(f"\n❓ 查询: {query}")

    result = await memory_manager.query(query, task_type="factual_qa", limit=3)

    if not result.retrieved_memories:
        print("   ❌ 未找到相关记忆。")
        return

    print(
        f"   🎯 找到 {len(result.retrieved_memories)} 条相关记忆 (耗时: {result.processing_time_ms:.1f}ms)"
    )
    top_memory = result.retrieved_memories[0]
    print(f"   🥇 最相关: '{top_memory.memory_record.content}'")
    print(
        f"   📊 得分: {top_memory.final_score:.3f} (相关性: {top_memory.relevance_score:.3f}, 时效性: {top_memory.recency_score:.3f})"
    )


async def run_advanced_demo():
    """高级功能演示：展示实体关联查询和系统统计。"""
    print("\n\n>> 🚀 场景二：高级功能演示 <<")
    print("=" * 60)

    llm, embeddings, client = setup_components()

    # 1. 创建一个启用所有高级功能的记忆管理器
    config = evermind.EverMindConfig(
        enable_association_tracking=True,
        enable_context_injection=True,
    )
    memory_manager = evermind.create_memory_manager(
        qdrant_client=client,
        llm=llm,
        embeddings=embeddings,
        config=config,
        namespace="advanced_demo",
    )
    await memory_manager.initialize()

    # 2. 录入记忆，这些记忆会自动建立实体关联
    print("\n📝 步骤 1: 录入记忆以建立实体关联...")
    memories = [
        "OpenAI发布了GPT-4模型，在多项基准测试中表现优异。",
        "项目经理王五担心技术风险，建议先做小规模验证。",
        "李博士认为Transformer架构是目前最适合的选择。",
        "我们的AI助手项目计划集成最新的大语言模型技术，比如GPT-4。",
    ]
    for content in memories:
        await memory_manager.ingest(content, process_immediately=True)
        print(f"  ✅ 记忆已录入: {content[:30]}...")

    # 3. 基于实体进行关联查询
    print("\n🔗 步骤 2: 基于实体进行关联查询...")
    entity_to_query = "GPT-4"
    print(f"\n🏷️  查询与 '{entity_to_query}' 相关的所有记忆:")
    related_memories = await memory_manager.query_by_association(entity=entity_to_query)

    if not related_memories:
        print(f"   ❌ 未找到与 '{entity_to_query}' 相关的记忆。")
    else:
        for mem in related_memories:
            print(f"   📄 '{mem['content']}' (得分: {mem['score']:.3f})")

    # 4. 查看系统和命名空间的统计信息
    print("\n📊 步骤 3: 查看系统统计...")
    system_stats = memory_manager.get_system_stats()
    print(f"   - 系统命名空间: {system_stats['namespaces']}")
    print(f"   - 后台任务数: {system_stats['background_tasks']}")

    namespace_stats = await memory_manager.get_namespace_stats()
    print(f"\n📈 命名空间 '{namespace_stats['namespace']}' 统计:")
    print(f"   - 总记忆数: {namespace_stats['stats']['total_memories']}")
    print(f"   - 占用存储: {namespace_stats['stats']['storage_size_mb']:.4f} MB")


async def run_agent_context_demo():
    """上下文引子演示：模拟智能体如何利用记忆引子来增强对话能力。"""
    print("\n\n>> 🤖 场景三：智能体上下文引子演示 <<")
    print("=" * 60)

    llm, embeddings, client = setup_components()
    memory_manager = evermind.create_simple_memory_manager(
        client, llm, embeddings, "agent_demo"
    )
    await memory_manager.initialize()

    # 1. 模拟智能体在与用户交互过程中积累的记忆
    print("\n📝 步骤 1: 智能体积累记忆...")
    experiences = [
        "我擅长Python编程和机器学习算法。",
        "用户小明经常问我关于深度学习的问题。",
        "上次帮助小红解决了数据分析问题，她很满意。",
        "我最近学习了Transformer架构的原理。",
    ]
    for exp in experiences:
        await memory_manager.ingest(exp, process_immediately=True)
        print(f"  ✅ 经验已记录: {exp[:30]}...")

    # 2. 在对话开始前，智能体主动获取记忆引子来“预热”上下文
    print("\n💡 步骤 2: 智能体获取记忆引子...")
    hints = await memory_manager.get_context_hints(limit=3)

    # 3. 将引子构建成注入到 Prompt 中的上下文
    context_prompt = "你是一个AI助手，以下是你当前的一些记忆线索，请在回答时参考：\n"
    for hint in hints:
        context_prompt += f"- {hint['content']} (相关记忆: {hint['memory_count']}条)\n"

    print("\n📋 构建的上下文 Prompt:")
    print("```")
    print(context_prompt.strip())
    print("```")
    print("\n✨ 这个Prompt可以注入到发送给LLM的请求中，让AI的回答更具个性化和连续性。")


async def main():
    """运行所有演示"""
    try:
        await run_basic_demo()
        await run_advanced_demo()
        await run_agent_context_demo()
        print("\n\n🎉 全部演示完成！")
    except Exception as e:
        print(f"\n❌ 演示过程中出现严重错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    evermind.print_welcome()
    asyncio.run(main())

# 🧠 EverMind - 赋予智能体超凡记忆

**一个为大型语言模型（LLM）应用设计的、渐进式、可扩展的智能记忆系统。**

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/Apache-2.0)
[](https://github.com/pforge-ai/evermind)
[](https://github.com/pforge-ai/evermind)

-----

## 📖 项目简介

在构建高级AI智能体或复杂聊天机器人时，我们面临一个核心挑战：如何让它们拥有像人一样的记忆能力？它们需要能够：

  * **长期记住** 对话的关键信息和用户的偏好。
  * 在海量信息中**智能检索**出最相关的内容。
  * **理解上下文**，并根据记忆线索进行推理和联想。

**EverMind** 正是为此而生。它是一个即插即用的记忆中间件，旨在为任何LLM应用提供一个强大、灵活且具备类人特性的记忆核心。它不仅仅是一个向量数据库的封装，而是一个完整、智能的记忆管理解决方案。

无论您是在开发一个需要记住用户历史的AI伴侣，一个需要分析大量文档的知识库助手，还是一个需要进行多步推理的自主智能体，EverMind 都能成为您坚实的记忆基石。

## ✨ 核心特性

EverMind 的设计哲学是“渐进式智能”，您可以根据需求，从一个简单的向量存储开始，逐步启用更高级的认知功能。

  * 🧠 **渐进式多粒度索引 (Progressive Multi-Granular Indexing)**

      * **自动分析**: EverMind 不只是存储原始文本。它会自动评估记忆的“重要性”，并根据重要性阈值，智能地抽取**概念、可回答问题、实体、关键词和摘要**。
      * **丰富检索维度**: 这意味着您的AI不仅能进行语义相似度搜索，还能从不同认知粒度上理解和检索信息。

  * 🔍 **RR权重智能检索 (Relevance-Recency Weighted Retrieval)**

      * **超越余弦相似度**: 传统的向量检索只关心语义相关性。EverMind 引入了“时效性（Recency）”作为核心权重，结合“相关性（Relevance）”，动态计算出最符合当前情境的记忆。
      * **任务可配置**: 您可以为不同任务（如：事实问答、日常对话）配置不同的RR权重，实现更精细的检索策略。

  * 💡 **动态上下文引子 (Dynamic Context Hints)**

      * **记忆线索生成**: 在每次查询后，EverMind 会自动生成“上下文引子”，例如“相关实体：李博士”、“可回答问题：Phoenix项目何时完成？”。
      * **增强LLM性能**: 这些引子可以被注入到下一次的LLM提示词（Prompt）中，极大地增强了AI的对话连续性和上下文感知能力，使其看起来更“主动”、更“聪明”。

  * 🔗 **实体关联网络 (Entity Association Network)**

      * **自动构建关系**: 系统会自动追踪在记忆中共现的实体（如人名、项目名），并在后台构建一个实体关联网络。
      * **实现联想查询**: 您可以直接查询某个实体（如“李博士”），系统会返回所有与他相关的记忆，实现类似“知识图谱”的联想能力。

  * 🗂️ **IF内部管理策略 (Importance-Frequency Internal Strategy)**

      * **智能内部管理**: 除了用于对外检索的RR权重，系统内部还使用IF（重要性-频次）策略来管理记忆本身。
      * **支持高级策略**: 基于IF得分，系统未来可以实现自适应遗忘、记忆归档、智能缓存等高级管理功能，确保记忆库的健康和高效。

  * ⚙️ **面向协议，轻松扩展 (Protocol-Oriented & Extensible)**

      * **厂商无锁定**: EverMind 的核心组件（如 `MemoryManager`）完全面向 LangChain 的核心协议（`BaseLanguageModel`, `Embeddings`）编程。
      * **任意模型接入**: 这意味着您可以轻松接入并切换任何LLM和Embedding模型提供商（如智谱、百炼、月之暗面、OpenAI等），只需传入一个兼容的实例即可。

  * 🚀 **高性能存储后端 (High-Performance Storage Backend)**

      * **Qdrant 深度集成**: 内置基于 [Qdrant](https://qdrant.tech/) 的高性能、可扩展的向量存储后端。
      * **异步与流式处理**: 支持异步IO操作，并提供了流式处理队列，可以将高延迟的索引任务放到后台执行，保证了API的快速响应。

## 🏛️ 架构概览

EverMind 的架构清晰且模块化，主要由以下几个核心组件构成：

1.  **MemoryManager**: **系统主入口**。负责协调所有组件，并对外提供统一的 `ingest`（录入）和 `query`（查询）接口。
2.  **IndexingProcessor**: **记忆加工厂**。负责调用LLM对原始记忆进行分析，抽取多粒度索引。
3.  **RetrievalEngine**: **智能检索引擎**。负责执行RR权重检索、生成上下文引子和处理关联查询。
4.  **StorageBackend**: **存储后端**。一个可插拔的存储接口，目前提供了基于 Qdrant 的高性能实现（`QdrantStorageBackend`）。

*(架构图正在绘制中...)*

## 🚀 快速开始

只需三步，即可在您的项目中集成 EverMind。

### 1\. 安装

首先，请确保您已经安装了所有必要的依赖。

```bash
pip install evermind-ai langchain langchain-openai langchain-community qdrant-client
```

*(注意: `evermind-ai` 是一个假设的包名，请根据实际情况修改)*

### 2\. 环境配置

EverMind 需要访问LLM和Embedding服务。请在您的环境中配置好API Keys。

```python
import os

# 配置智谱AI (用于LLM)
os.environ["ZHIPU_API_KEY"] = "YOUR_ZHIPU_API_KEY"
os.environ["ZHIPU_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4/"

# 配置阿里百炼 DashScope (用于Embedding)
os.environ["DASHSCOPE_API_KEY"] = "YOUR_DASHSCOPE_API_KEY"
```

### 3\. 代码示例

下面的示例将演示从初始化、录入记忆到最终查询的全过程。

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.tongyi import TongyiEmbeddings
from qdrant_client import QdrantClient
import evermind

async def main():
    # 1. 初始化依赖组件
    llm = ChatOpenAI(
        model="glm-4.5-x",
        openai_api_base=os.getenv("ZHIPU_BASE_URL"),
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
    )
    embeddings = TongyiEmbeddings(model="text-embedding-v4")
    qdrant_client = QdrantClient(":memory:") # 使用内存数据库进行演示

    # 2. 创建 EverMind 记忆管理器
    # 使用 create_simple_memory_manager 可以快速开始，它内置了推荐的最佳实践配置
    memory_manager = evermind.create_simple_memory_manager(
        qdrant_client=qdrant_client,
        llm=llm,
        embeddings=embeddings,
        namespace="my_first_agent"
    )
    await memory_manager.initialize()
    print("✅ EverMind 初始化成功！")

    # 3. 录入记忆 (Ingest)
    print("\n📝 正在录入记忆...")
    await memory_manager.ingest(
        "项目Phoenix的负责人是李博士，这是一个AI认知架构研发项目。",
        process_immediately=True # 立即处理索引，便于演示
    )
    await memory_manager.ingest(
        "李博士提到Phoenix项目需要在6个月内完成第一个原型。",
        process_immediately=True
    )
    print("   记忆录入完成。")

    # 4. 查询记忆 (Query)
    print("\n🔍 正在查询记忆...")
    query = "谁负责Phoenix项目？什么时候需要完成？"
    print(f"   查询: {query}")
    
    result = await memory_manager.query(query, task_type="factual_qa")

    # 5. 查看结果
    if result.retrieved_memories:
        top_memory = result.retrieved_memories[0]
        print("\n🎯 查询结果:")
        print(f"   最相关的记忆: '{top_memory.memory_record.content}'")
        print(f"   综合得分: {top_memory.final_score:.3f}")
        print("\n💡 上下文引子:")
        for hint in result.context_hints:
            print(f"   - {hint.content}")
    else:
        print("   未找到相关记忆。")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 配置与使用

您可以通过 `EverMindConfig` 对象来精细化控制 EverMind 的行为。

```python
from evermind import EverMindConfig, create_memory_manager

# 创建一个自定义配置
my_config = EverMindConfig(
    # 启用关联追踪
    enable_association_tracking=True,
    # 禁用上下文引子生成以节省token
    enable_context_injection=False,
    # 调整索引抽取的难度
    indexing_config={
        "concept_extraction_threshold": 2.5, # 只有重要性高于2.5的记忆才抽取概念
    }
)

# 使用自定义配置创建管理器
memory_manager = create_memory_manager(
    config=my_config,
    # ... 其他参数 ...
)
```

## 🤝 贡献

我们热烈欢迎来自社区的任何贡献！无论您是修复Bug、添加新功能还是完善文档，我们都非常感谢。

请遵循以下步骤参与贡献：

1.  Fork 本仓库。
2.  创建一个新的分支 (`git checkout -b feature/your-feature-name`)。
3.  提交您的修改 (`git commit -m 'Add some feature'`)。
4.  将您的分支推送到远程仓库 (`git push origin feature/your-feature-name`)。
5.  创建一个 Pull Request。

## 📄 许可证

本项目采用 [Apache 2.0 许可证](https://opensource.org/licenses/Apache-2.0)。
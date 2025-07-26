# Evermind
**为智能体打造的“神级记忆装备”，一个可实现永久存活与自我进化的认知核心。**

## 📖 简介
Evermind 是一个专为高级AI智能体设计的、高度模块化、可配置的记忆系统。它的目标远不止于信息的被动存储与检索，而是构建一个能够模拟生物认知、实现主动学习、自我进化和长期成长的动态认知架构。

搭载 Evermind 的智能体将不再遗忘重要的上下文，更能从过去的经验中学习、反思、提炼出新的知识和策略，随着时间的推移而不断成长。

#### 核心设计原则
🧠 认知模拟: 架构模仿人类记忆的分层与分类机制（情节、语义、程序记忆）。

🧬 动态自组织: 记忆能够通过主动的“元认知反思”与“记忆维护”机制，持续优化自身结构和内容。

🔌 统一与可扩展: 所有后端依赖（LLM、向量数据库、图数据库）均面向协议设计，可被轻松替换和扩展。

📉 优雅降级: 系统功能可通过配置开关。在最简模式下，它是一个高性能的情节记忆库；在完全体模式下，它是一个能够自我进化的认知系统。

## ✨ 核心功能
+ 多维记忆评估 (RRIF模型): 通过时效性 (Recency)、相关性 (Relevance)、重要性 (Importance) 和 频率性 (Frequency) 四个维度智能评估每一条记忆的价值，实现前所未有的智能检索排序。

+ 元认知反思: 智能体可以在“空闲”时自动反思高价值的记忆，发现模式、提炼洞见，并将其固化为新的结构化知识。

+ 高级知识融合: 在反思过程中，通过两阶段的LLM调用，实现对新知识的实体归一和关系归一，有效避免知识冗余，构建高质量的知识图谱。

+ 混合检索: 智能地规划查询，能够无缝地结合向量检索（用于情节记忆）和图谱查询（用于语义记忆），提供更精准的答案。

+ 可配置的答案生成: 用户可以自由选择是获取由LLM合成的最终答案，还是直接获取经过RRIF模型排序后的原始记忆片段。

+ 记忆生命周期管理: 具备完整的记忆维护机制，包括固化、压缩、归档和删除，确保系统长期健康运行。

## 🚀 快速上手
### 1. 环境准备, 首先，确保您已安装 Python 3.9+。

#### 安装 Evermind SDK
```
pip install evermind
```

#### 设置您选择的大模型提供商的API密钥
例如，使用阿里巴巴通义千问
```
export DASHSCOPE_API_KEY="your_api_key_here"
```
(可选) 如果您想使用知识图谱功能，可通过Docker启动一个Neo4j实例
```
docker run --name evermind -neo4j -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/password neo4j:5
```

### 2. 使用示例
下面的示例将展示如何快速启动一个具备完整功能的 Evermind 实例。
```python
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.graphs import Neo4jGraph
from langchain_tongyi import ChatTongyi, TongyiEmbeddings
from evermind import MemoryManager, MnemonConfig, InProcessTaskQueue

# 配置日志，观察Evermind内部活动
logging.basicConfig(level=logging.INFO)

# 1. 初始化依赖组件
llm = ChatTongyi(model="qwen-max")
embedding_model = TongyiEmbeddings(model="text-embedding-v2")
vector_store = FAISS.from_texts(["init"], embedding_model) # 使用内存向量库
graph_store = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
task_queue = InProcessTaskQueue() # 使用简单的同步任务队列

# 2. 实例化 MemoryManager
memory = MemoryManager(
    config=MnemonConfig(), # 使用默认配置
    vector_store=vector_store,
    llm=llm,
    embedding_model=embedding_model,
    graph_store=graph_store,
    task_queue=task_queue,
    initial_instructions=["你是一个AI助手。"]
)

# 3. 写入记忆
memory.ingest("项目'凤凰'的主要负责人是李博士。")
memory.ingest("用户张三对项目'凤凰'的成本非常关心。")

# 4. 触发一次学习（元认知反思）
print("\n--- 正在进行元认知反思 ---")
memory.run_maintenance(run_reflection=True, run_health_check=False)
print("反思完成，知识已存入图谱。")

# 5. 进行查询
print("\n--- 正在进行查询 ---")
query = "谁是项目'凤凰'的负责人？"
result = memory.query(query)

print(f"\n查询: {query}")
print(f"回答: {result.synthesized_answer}")
```

## 🛠️ 架构与配置
Evermind 的核心是 MemoryManager 类，它通过一个统一的API (ingest, query, run_maintenance) 来协调所有内部工作流。

您可以通过实例化 MnemonConfig 并修改其属性，来精细地控制系统的每一个行为，例如关闭知识图谱功能、调整RRIF权重、配置记忆维护策略等。
```
from evermind import MnemonConfig

# 创建一个自定义配置
custom_config = MnemonConfig(
    enable_semantic_memory=False, # 禁用知识图谱
    recency_decay_rate=0.05      # 让记忆“遗忘”得更快
)

# ...在实例化MemoryManager时传入
# memory = MemoryManager(config=custom_config, ...)
```

## 🗺️ 路线图
我们对 Evermind 有着宏伟的规划，欢迎社区一同参与贡献：

[ ] 完善 _run_health_check_stage 的具体实现。
[ ] 提供更多开箱即用的数据库和任务队列适配器。
[ ] 引入更高级的记忆聚类算法。
[ ] 支持多模态记忆（图像、声音）。

## 🤝 贡献
我们热烈欢迎任何形式的贡献！如果您有任何想法、建议或发现了bug，请随时提交 Issues 或 Pull Requests。

## 📄 许可证
本项目采用 Apache 2.0 license 开源。
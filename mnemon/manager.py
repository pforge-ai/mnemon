import logging
import time
import uuid
import math
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.vectorstores import VectorStore

from .config import MnemonConfig, RRIFWeights
from .protocols import ITaskQueue
from .models import (
    ImportanceRating,
    MemoryMetadata,
    MemoryRecord,
    QuestionExtraction,
    QueryPlan,
    QueryResult,
    ReflectionResult,
    RetrievedMemory,
    FusedKnowledge,
)

# --- 日志记录器 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryManager:
    """
    MNEMON SDK 的核心管理器。
    这是与记忆系统交互的唯一入口点。
    """

    def __init__(
        self,
        config: MnemonConfig,
        vector_store: VectorStore,
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        graph_store: Optional[GraphStore] = None,
        task_queue: Optional[ITaskQueue] = None,
        initial_instructions: List[str] = [],
    ):
        self.config = config
        self.vector_store = vector_store
        self.llm = llm
        self.embedding_model = embedding_model
        self.graph_store = graph_store
        self.task_queue = task_queue

        if self.config.enable_semantic_memory and self.graph_store is None:
            raise ValueError(
                "GraphStore must be provided when 'enable_semantic_memory' is True."
            )
        if (
            self.config.enable_meta_reflection
            or self.config.maintenance.enable_archiving
        ) and self.task_queue is None:
            logger.warning(
                "TaskQueue is not provided. Meta-reflection and maintenance will run synchronously."
            )

        self._bootstrap(initial_instructions)
        logger.info("MemoryManager initialized successfully.")

    def ingest(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"Ingesting content: '{content[:50]}...'")
        base_metadata = MemoryMetadata(
            source_type=(
                metadata.get("source_type", "user_input") if metadata else "user_input"
            ),
            custom_data=metadata.get("custom_data", {}) if metadata else {},
        )
        record = MemoryRecord(content=content, metadata=base_metadata)
        if self.task_queue:
            self.task_queue.submit_task(self._process_ingestion_task, record)
        else:
            self._process_ingestion_task(record)
        return record.id

    def query(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None,
        synthesize_answer: bool = True,
    ) -> QueryResult:
        logger.info(
            f"Querying with text: '{query_text[:50]}...'. Synthesize answer: {synthesize_answer}"
        )
        plan = self._plan_query(query_text)
        retrieved_docs = []
        if plan.vector_search_query:
            retrieved_docs.extend(
                self.vector_store.similarity_search_with_score(
                    plan.vector_search_query, k=10
                )
            )

        if (
            self.config.enable_semantic_memory
            and plan.requires_knowledge_graph
            and self.graph_store
            and plan.graph_query
        ):
            try:
                graph_results = self.graph_store.query(plan.graph_query)
                for res in graph_results:
                    retrieved_docs.append(
                        (
                            Document(
                                page_content=str(res),
                                metadata={
                                    "source_type": "semantic_memory",
                                    "timestamp": time.time(),
                                },
                            ),
                            1.0,
                        )
                    )
            except Exception as e:
                logger.error(f"Error querying knowledge graph: {e}")

        if not retrieved_docs:
            return QueryResult(retrieved_memories=[])

        task_type = context.get("task_type", "default") if context else "default"
        weights = self.config.weights_by_task.get(
            task_type, self.config.weights_by_task["default"]
        )
        ranked_memories = self._rerank_memories(retrieved_docs, weights)
        self._update_memory_frequency(
            [mem.id for mem in ranked_memories if mem.type != "semantic_memory"]
        )

        final_answer = None
        if synthesize_answer:
            if not ranked_memories:
                final_answer = "I have no relevant memory to answer that."
            else:
                final_answer = self._synthesize_answer(query_text, ranked_memories)
        return QueryResult(
            synthesized_answer=final_answer, retrieved_memories=ranked_memories
        )

    def run_maintenance(
        self,
        run_reflection: bool = True,
        run_health_check: bool = True,
        force: bool = False,
        tasks: Optional[List[str]] = None,
    ) -> None:
        logger.info("Manual maintenance cycle triggered with custom parameters.")
        if run_reflection:
            if self.config.enable_meta_reflection:
                self._run_reflection_stage()
            else:
                logger.warning("Meta-reflection is disabled in config, skipping.")
        if run_health_check:
            self._run_health_check_stage(force=force, tasks=tasks)
        logger.info("Maintenance cycle finished.")

    def _process_ingestion_task(self, record: MemoryRecord):
        try:
            logger.info(f"Processing memory record {record.id} in background...")
            rating = self._rate_importance(record.content)
            record.metadata.importance_score = rating.score
            logger.info(
                f"Memory {record.id} rated with importance {rating.score}: '{rating.reason}'"
            )
            if rating.score >= self.config.importance_threshold_for_question_extraction:
                extracted = self._extract_questions(record.content)
                record.metadata.questions = extracted.questions
                logger.info(
                    f"Extracted {len(extracted.questions)} questions for memory {record.id}"
                )
            record.content_embedding = self.embedding_model.embed_query(record.content)
            lc_document = self._convert_to_langchain_document(record)
            self.vector_store.add_documents([lc_document])
            logger.info(f"Successfully processed and stored memory {record.id}.")
        except Exception as e:
            logger.error(
                f"Failed to process memory record {record.id}: {e}", exc_info=True
            )

    def _bootstrap(self, instructions: List[str]):
        if not instructions:
            return
        logger.info(f"Bootstrapping with {len(instructions)} initial instructions...")
        for instruction in instructions:
            base_metadata = MemoryMetadata(
                source_type="initial_instruction", importance_score=4.0
            )
            record = MemoryRecord(content=instruction, metadata=base_metadata)
            self._process_ingestion_task(record)

    def _run_reflection_stage(self):
        """执行元认知反思阶段，现在带有高级知识融合逻辑。"""
        logger.info("Starting meta-reflection stage...")
        try:
            high_value_docs = self.vector_store.similarity_search(
                "*", k=100, filter={"importance_score": {"$gte": 3.0}}
            )
        except TypeError:
            logger.warning(
                "Vector store does not support filtering or failed on it. Falling back to manual filtering."
            )
            high_value_docs = self.vector_store.similarity_search("*", k=100)
            high_value_docs = [
                doc
                for doc in high_value_docs
                if (doc.metadata.get("importance_score") or 0) >= 3.0
            ]
        if not high_value_docs:
            logger.info("No high-value memories found for reflection. Skipping.")
            return

        content_for_reflection = "\n\n---\n\n".join(
            [doc.page_content for doc in high_value_docs]
        )

        # 1. 初步提取
        reflection = self._reflect_on_memories(content_for_reflection)

        # --- 已修复 ---
        # 使用 getattr 安全地访问属性，防止因LLM未返回该字段而崩溃
        raw_triplets = getattr(reflection, "raw_triplets", None)
        insights = getattr(reflection, "insights", [])

        if not raw_triplets:
            logger.info(
                "No raw triplets were extracted from memories. Insights found: %s",
                insights,
            )
            # 即使没有三元组，洞见本身也值得被记录
            for insight in insights:
                self.ingest(
                    f"Insight from reflection: {insight}",
                    metadata={"source_type": "self_reflection"},
                )
            return

        # 2. 知识融合与归一
        fused_knowledge = self._fuse_knowledge(raw_triplets)
        if not fused_knowledge.fused_triplets:
            logger.info("Knowledge fusion resulted in no valid triplets.")
            return

        # 3. 写入融合后的知识
        if self.graph_store:
            logger.info(
                f"Fusing {len(fused_knowledge.fused_triplets)} new knowledge triplets into graph..."
            )
            for subj, pred, obj in fused_knowledge.fused_triplets:
                self.graph_store.query(
                    """
                    MERGE (a:Entity {name: $subj})
                    MERGE (b:Entity {name: $obj})
                    MERGE (a)-[r:`%s`]->(b)
                    ON CREATE SET r.confidence = 1, r.created_at = timestamp()
                    ON MATCH SET r.confidence = r.confidence + 1
                    """
                    % pred.upper().replace(" ", "_"),
                    params={"subj": subj, "obj": obj},
                )

        for insight in insights:
            self.ingest(
                f"Insight from reflection: {insight}",
                metadata={"source_type": "self_reflection"},
            )

    def _run_health_check_stage(self, force: bool, tasks: Optional[List[str]]):
        logger.info("Starting memory health check stage...")
        allowed_tasks = tasks or [
            "consolidation",
            "compression",
            "archiving",
            "deletion",
        ]
        logger.info(f"Executing health check tasks: {allowed_tasks}")
        pass

    def _plan_query(self, query_text: str) -> QueryPlan:
        parser = PydanticOutputParser(pydantic_object=QueryPlan)
        prompt_template = """
        Analyze the user's query and create a query plan.
        1. Rewrite the query to be optimal for vector database search.
        2. Determine if the query requires structured information from a knowledge graph.
        3. If so, generate a Cypher query. IMPORTANT: All nodes use the label `Entity` and have a `name` property.

        {format_instructions}

        User Query:
        "{query}"
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        return chain.invoke({"query": query_text})

    def _rerank_memories(
        self, documents: List[tuple[Document, float]], weights: RRIFWeights
    ) -> List[RetrievedMemory]:
        ranked_list = []
        for doc, relevance_score in documents:
            metadata = doc.metadata
            if not metadata:
                metadata = {}
            norm_relevance = relevance_score
            days_elapsed = (time.time() - metadata.get("timestamp", time.time())) / (
                60 * 60 * 24
            )
            norm_recency = math.exp(-self.config.recency_decay_rate * days_elapsed)
            importance = metadata.get("importance_score", 0)
            norm_importance = importance / 4.0
            access_count = metadata.get("access_count", 0)
            norm_frequency = math.log1p(access_count) / math.log1p(1000)
            final_score = (
                weights.relevance * norm_relevance
                + weights.recency * norm_recency
                + weights.importance * norm_importance
                + weights.frequency * norm_frequency
            )
            metadata.setdefault("source_type", "episodic")
            filtered_metadata = {
                k: v for k, v in metadata.items() if k in MemoryMetadata.model_fields
            }
            ranked_list.append(
                RetrievedMemory(
                    id=metadata.get("memory_id", str(uuid.uuid4())),
                    content=doc.page_content,
                    metadata=MemoryMetadata(**filtered_metadata),
                    score=final_score,
                    type=metadata.get("source_type", "episodic"),
                )
            )
        ranked_list.sort(key=lambda x: x.score, reverse=True)
        return ranked_list[:10]

    def _synthesize_answer(self, query: str, memories: List[RetrievedMemory]) -> str:
        context_str = "\n\n".join(
            [
                f"Memory (Score: {mem.score:.2f}, Source: {mem.type}):\n{mem.content}"
                for mem in memories
            ]
        )
        prompt_template = """
        Based on the following memories, provide a comprehensive, synthesized answer to the user's query.
        Do not just list the memories. Weave them into a coherent response.
        If the memories seem irrelevant, state that you couldn't find a relevant answer in your memory.

        Memories:
        ---
        {context}
        ---

        User Query: "{query}"

        Answer:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        result = chain.invoke({"context": context_str, "query": query})
        return result.content if hasattr(result, "content") else str(result)

    def _update_memory_frequency(self, memory_ids: List[str]):
        """
        为被成功用于回答问题的记忆增加访问计数。
        """
        if not memory_ids:
            return

        logger.info(f"Attempting to update access count for memories: {memory_ids}")

        if hasattr(self.vector_store, "docstore") and hasattr(
            self.vector_store.docstore, "_dict"
        ):
            memory_id_to_internal_id = {}
            for internal_id, doc in self.vector_store.docstore._dict.items():
                if (
                    "memory_id" in doc.metadata
                    and doc.metadata["memory_id"] in memory_ids
                ):
                    memory_id_to_internal_id[doc.metadata["memory_id"]] = internal_id

            updated_count = 0
            for mem_id in memory_ids:
                internal_id = memory_id_to_internal_id.get(mem_id)
                if internal_id:
                    doc_to_update = self.vector_store.docstore._dict[internal_id]
                    current_count = doc_to_update.metadata.get("access_count", 0)
                    doc_to_update.metadata["access_count"] = current_count + 1
                    updated_count += 1

            if updated_count > 0:
                logger.info(
                    f"Successfully updated access count for {updated_count} memories in FAISS docstore."
                )
        else:
            logger.warning(
                f"Vector store of type '{type(self.vector_store).__name__}' does not have a known "
                f"interface for metadata updates. Skipping frequency update."
            )

    def _rate_importance(self, content: str) -> ImportanceRating:
        parser = PydanticOutputParser(pydantic_object=ImportanceRating)
        prompt_template = """
        Rate the importance of the following piece of memory for a long-term AI agent.
        - Score 0: Trivial, conversational filler.
        - Score 1: Simple, common facts.
        - Score 2: Contains specific, useful information.
        - Score 3: A key insight, a core principle.
        - Score 4: A foundational, unchangeable instruction.

        {format_instructions}
        Memory Content: "{content}"
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        return chain.invoke({"content": content})

    def _extract_questions(self, content: str) -> QuestionExtraction:
        parser = PydanticOutputParser(pydantic_object=QuestionExtraction)
        prompt_template = """
        Based on the following text, generate a list of questions that this text can directly answer.
        {format_instructions}
        Text: "{content}"
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        return chain.invoke({"content": content})

    def _convert_to_langchain_document(self, record: MemoryRecord) -> Document:
        metadata_dict = record.metadata.model_dump(exclude_none=True)
        return Document(
            page_content=record.content if isinstance(record.content, str) else "",
            metadata={
                "memory_id": record.id,
                "timestamp": record.timestamp,
                **metadata_dict,
            },
        )

    def _reflect_on_memories(self, content: str) -> ReflectionResult:
        """初步提取洞见和原始三元组。"""
        parser = PydanticOutputParser(pydantic_object=ReflectionResult)
        prompt_template = """
        You are a highly intelligent AI assistant performing a self-reflection task.
        Based on the following collection of memories, your goal is to discover underlying patterns, insights, and new structured knowledge.
        1.  **Insights**: Summarize key patterns, recurring themes, or profound conclusions.
        2.  **Raw Triplets**: Extract all possible factual knowledge in the form of [Subject, Predicate, Object] triplets. Be liberal in your extraction.

        {format_instructions}
        Memory Collection:
        ---
        {content}
        ---
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        return chain.invoke({"content": content})

    def _fuse_knowledge(self, raw_triplets: List[List[str]]) -> FusedKnowledge:
        """对原始三元组进行融合与归一。"""
        parser = PydanticOutputParser(pydantic_object=FusedKnowledge)

        raw_triplets_str = "\n".join([f"- {triplet}" for triplet in raw_triplets])

        prompt_template = """
        You are a knowledge fusion expert. Your task is to take a list of raw, potentially messy knowledge triplets and fuse them into a clean, canonical set.
        
        Follow these steps:
        1.  **Entity Normalization**: Identify subjects and objects that refer to the same entity and choose a single, canonical name for them (e.g., "Dr. Li", "Li博士" -> "李博士").
        2.  **Relation Normalization**: Identify predicates that represent the same semantic relationship and choose a single, canonical predicate for them (e.g., "is the leader of", "manages" -> "IS_LEADER_OF"). The canonical predicate should be in uppercase snake_case format.
        3.  **Deduplication**: Remove any duplicate triplets after normalization.

        Return only the final, fused list of triplets.

        {format_instructions}

        Raw Triplets to Fuse:
        ---
        {raw_triplets}
        ---
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["raw_triplets"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        return chain.invoke({"raw_triplets": raw_triplets_str})

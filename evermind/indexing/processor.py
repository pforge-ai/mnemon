"""
EverMind Indexing Processor

多粒度索引处理器，负责从记忆内容中抽取概念、问题、摘要等索引信息。
"""

import asyncio
from typing import List, Optional, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ..models import (
    MemoryRecord,
    ImportanceRating,
    ConceptExtraction,
    QuestionExtraction,
    SummaryGeneration,
)
from ..config import EverMindConfig
from ..storage.protocols import IStorageBackend


class IndexingProcessor:
    """多粒度索引处理器"""

    def __init__(self, llm: BaseLanguageModel, config: EverMindConfig):
        self.llm = llm
        self.config = config
        self._importance_parser = PydanticOutputParser(pydantic_object=ImportanceRating)
        self._concept_parser = PydanticOutputParser(pydantic_object=ConceptExtraction)
        self._question_parser = PydanticOutputParser(pydantic_object=QuestionExtraction)
        self._summary_parser = PydanticOutputParser(pydantic_object=SummaryGeneration)
        self._setup_prompts()

    def _setup_prompts(self):
        """设置LLM提示模板"""
        self._importance_prompt = PromptTemplate(
            template="""
            作为一个智能记忆系统，请评估以下内容的重要性级别。
            评分标准：0分（日常闲聊），1分（一般事实），2分（有用信息），3分（重要洞察），4分（核心规则）。
            同时请抽取内容中的关键实体。
            {format_instructions}
            内容："{content}"
            """,
            input_variables=["content"],
            partial_variables={
                "format_instructions": self._importance_parser.get_format_instructions()
            },
        )
        self._concept_prompt = PromptTemplate(
            template="""
            从以下内容中抽取关键概念、实体和关键词。
            要求：概念是抽象理论，实体是具体人/物，关键词是重要描述词。每类最多{max_concepts}个。
            {format_instructions}
            内容："{content}"
            """,
            input_variables=["content", "max_concepts"],
            partial_variables={
                "format_instructions": self._concept_parser.get_format_instructions()
            },
        )
        self._question_prompt = PromptTemplate(
            template="""
            基于以下内容，生成这段内容能够直接回答的问题。最多{max_questions}个。
            {format_instructions}
            内容："{content}"
            """,
            input_variables=["content", "max_questions"],
            partial_variables={
                "format_instructions": self._question_parser.get_format_instructions()
            },
        )
        self._summary_prompt = PromptTemplate(
            template="""
            为以下内容生成1-2句话的简洁摘要和关键要点。
            {format_instructions}
            内容："{content}"
            """,
            input_variables=["content"],
            partial_variables={
                "format_instructions": self._summary_parser.get_format_instructions()
            },
        )

    async def process_memory(self, memory: MemoryRecord) -> MemoryRecord:
        """处理单条记忆，生成完整的多粒度索引"""
        importance_result = await self._assess_importance(memory.content)
        memory.metadata.importance_score = importance_result.score
        memory.metadata.associated_entities = importance_result.extracted_entities

        cfg = self.config.indexing_config
        tasks = []

        if (
            cfg.enable_concept_extraction
            and importance_result.score >= cfg.concept_extraction_threshold
        ):
            tasks.append(self._extract_concepts(memory.content))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        if (
            cfg.enable_question_extraction
            and importance_result.score >= cfg.question_extraction_threshold
        ):
            tasks.append(self._extract_questions(memory.content))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        if cfg.enable_summary_generation:
            tasks.append(self._generate_summary(memory.content))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        concept_res, question_res, summary_res = await asyncio.gather(*tasks)

        if concept_res:
            memory.indexes.concepts = concept_res.concepts[
                : cfg.max_concepts_per_memory
            ]
            memory.indexes.keywords = concept_res.keywords
            memory.metadata.associated_entities = list(
                set(memory.metadata.associated_entities + concept_res.entities)
            )
        if question_res:
            memory.indexes.questions = question_res.questions[
                : cfg.max_questions_per_memory
            ]
        if summary_res:
            memory.indexes.summary = summary_res.summary

        return memory

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息（简化版）"""
        return {
            "total_processed": 0,
            "avg_processing_time_ms": 0.0,
            "enabled_features": {
                "concept_extraction": self.config.indexing_config.enable_concept_extraction,
                "question_extraction": self.config.indexing_config.enable_question_extraction,
                "summary_generation": self.config.indexing_config.enable_summary_generation,
            },
        }

    async def _assess_importance(self, content: str) -> ImportanceRating:
        try:
            return await (
                self._importance_prompt | self.llm | self._importance_parser
            ).ainvoke({"content": content})
        except Exception as e:
            print(f"Error assessing importance: {e}")
            return ImportanceRating(
                score=1.0, reasoning="评估失败", extracted_entities=[]
            )

    async def _extract_concepts(self, content: str) -> ConceptExtraction:
        try:
            return await (
                self._concept_prompt | self.llm | self._concept_parser
            ).ainvoke(
                {
                    "content": content,
                    "max_concepts": self.config.indexing_config.max_concepts_per_memory,
                }
            )
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            return ConceptExtraction(concepts=[], keywords=[], entities=[])

    async def _extract_questions(self, content: str) -> QuestionExtraction:
        try:
            return await (
                self._question_prompt | self.llm | self._question_parser
            ).ainvoke(
                {
                    "content": content,
                    "max_questions": self.config.indexing_config.max_questions_per_memory,
                }
            )
        except Exception as e:
            print(f"Error extracting questions: {e}")
            return QuestionExtraction(questions=[])

    async def _generate_summary(self, content: str) -> SummaryGeneration:
        try:
            return await (
                self._summary_prompt | self.llm | self._summary_parser
            ).ainvoke({"content": content})
        except Exception as e:
            print(f"Error generating summary: {e}")
            return SummaryGeneration(summary="", key_points=[])


class StreamingIndexProcessor:
    """流式索引处理器，用于实时后台处理"""

    def __init__(
        self,
        base_processor: IndexingProcessor,
        storage_backend: IStorageBackend,
        config: EverMindConfig,
    ):
        self.base_processor = base_processor
        self.storage = storage_backend
        self.config = config
        self._processing_queue = asyncio.Queue()
        self._is_running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def _update_associations(self, memory: MemoryRecord):
        if not self.config.is_feature_enabled("association"):
            return
        namespace = memory.metadata.namespace
        entities = memory.metadata.associated_entities
        for entity in entities:
            await self.storage.association_store.add_association(
                entity, memory.id, namespace
            )
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                await self.storage.association_store.update_association_strength(
                    entity1, entity2, namespace
                )

    async def _process_loop(self):
        """后台处理循环"""
        self._is_running = True
        while self._is_running:
            try:
                memory: MemoryRecord = await asyncio.wait_for(
                    self._processing_queue.get(), timeout=1.0
                )
                processed_memory = await self.base_processor.process_memory(memory)
                await self.storage.vector_store.store_memory(processed_memory)
                await self._update_associations(processed_memory)
                self._processing_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in streaming processor loop: {e}")

    def start_processing(self):
        """启动后台处理任务"""
        if not self._is_running:
            self._worker_task = asyncio.create_task(self._process_loop())

    async def submit_memory(self, memory: MemoryRecord):
        await self._processing_queue.put(memory)

    async def stop_processing(self):
        if not self._is_running:
            return
        await self._processing_queue.join()
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None

    def get_queue_size(self) -> int:
        return self._processing_queue.qsize()

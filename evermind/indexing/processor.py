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
    MultiGranularIndex,
    ImportanceRating,
    ConceptExtraction,
    QuestionExtraction,
    SummaryGeneration,
)
from ..config import EverMindConfig


class IndexingProcessor:
    """多粒度索引处理器"""

    def __init__(self, llm: BaseLanguageModel, config: EverMindConfig):
        self.llm = llm
        self.config = config

        # 初始化解析器
        self._importance_parser = PydanticOutputParser(pydantic_object=ImportanceRating)
        self._concept_parser = PydanticOutputParser(pydantic_object=ConceptExtraction)
        self._question_parser = PydanticOutputParser(pydantic_object=QuestionExtraction)
        self._summary_parser = PydanticOutputParser(pydantic_object=SummaryGeneration)

        # 初始化提示模板
        self._setup_prompts()

    def _setup_prompts(self):
        """设置LLM提示模板"""

        # 重要性评估提示
        self._importance_prompt = PromptTemplate(
            template="""
            作为一个智能记忆系统，请评估以下内容的重要性级别。

            评分标准：
            - 0分：日常闲聊、无意义信息
            - 1分：一般事实、常识性信息  
            - 2分：有用的具体信息、技能知识
            - 3分：重要洞察、关键决策、核心原则
            - 4分：系统性指令、基础设定、不可变规则

            同时请抽取内容中的关键实体（人名、地名、项目名、概念名等）。

            {format_instructions}

            内容："{content}"
            """,
            input_variables=["content"],
            partial_variables={
                "format_instructions": self._importance_parser.get_format_instructions()
            },
        )

        # 概念抽取提示
        self._concept_prompt = PromptTemplate(
            template="""
            从以下内容中抽取关键概念、实体和关键词。

            要求：
            - 概念：抽象的概念、理论、方法论
            - 实体：具体的人、地、物、组织
            - 关键词：重要的描述词、动作词

            保持简洁，每类最多{max_concepts}个。

            {format_instructions}

            内容："{content}"
            """,
            input_variables=["content", "max_concepts"],
            partial_variables={
                "format_instructions": self._concept_parser.get_format_instructions()
            },
        )

        # 问题抽取提示
        self._question_prompt = PromptTemplate(
            template="""
            基于以下内容，生成这段内容能够直接回答的问题。

            要求：
            - 问题应该具体、明确
            - 内容必须能完整回答该问题
            - 优先生成事实性问题
            - 最多生成{max_questions}个问题

            {format_instructions}

            内容："{content}"
            """,
            input_variables=["content", "max_questions"],
            partial_variables={
                "format_instructions": self._question_parser.get_format_instructions()
            },
        )

        # 摘要生成提示
        self._summary_prompt = PromptTemplate(
            template="""
            为以下内容生成简洁的摘要和关键要点。

            要求：
            - 摘要控制在1-2句话内
            - 关键要点用简短的词组表达
            - 保留最重要的信息

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

        # 1. 评估重要性（总是执行）
        importance_result = await self._assess_importance(memory.content)
        memory.metadata.importance_score = importance_result.score
        memory.metadata.associated_entities = importance_result.extracted_entities

        # 2. 根据配置和重要性阈值决定是否进行进一步索引
        indexing_config = self.config.indexing_config

        # 概念抽取
        if (
            indexing_config.enable_concept_extraction
            and memory.metadata.importance_score
            >= indexing_config.concept_extraction_threshold
        ):
            concept_result = await self._extract_concepts(memory.content)
            memory.indexes.concepts = concept_result.concepts[
                : indexing_config.max_concepts_per_memory
            ]
            memory.indexes.keywords = concept_result.keywords
            # 合并实体到关联实体中
            memory.metadata.associated_entities.extend(concept_result.entities)
            memory.metadata.associated_entities = list(
                set(memory.metadata.associated_entities)
            )

        # 问题抽取
        if (
            indexing_config.enable_question_extraction
            and memory.metadata.importance_score
            >= indexing_config.question_extraction_threshold
        ):
            question_result = await self._extract_questions(memory.content)
            memory.indexes.questions = question_result.questions[
                : indexing_config.max_questions_per_memory
            ]

        # 摘要生成
        if indexing_config.enable_summary_generation:
            summary_result = await self._generate_summary(memory.content)
            memory.indexes.summary = summary_result.summary

        return memory

    async def batch_process_memories(
        self, memories: List[MemoryRecord]
    ) -> List[MemoryRecord]:
        """批量处理记忆"""
        # 并发处理，但限制并发数避免过载
        semaphore = asyncio.Semaphore(5)  # 最多5个并发

        async def process_with_semaphore(memory):
            async with semaphore:
                return await self.process_memory(memory)

        tasks = [process_with_semaphore(memory) for memory in memories]
        processed_memories = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤异常结果
        results = []
        for i, result in enumerate(processed_memories):
            if isinstance(result, Exception):
                print(f"Error processing memory {memories[i].id}: {result}")
                results.append(memories[i])  # 返回原始记忆
            else:
                results.append(result)

        return results

    async def _assess_importance(self, content: str) -> ImportanceRating:
        """评估重要性并抽取实体"""
        try:
            chain = self._importance_prompt | self.llm | self._importance_parser
            result = await chain.ainvoke({"content": content})
            return result
        except Exception as e:
            print(f"Error assessing importance: {e}")
            return ImportanceRating(
                score=1.0, reasoning="评估失败，使用默认分数", extracted_entities=[]
            )

    async def _extract_concepts(self, content: str) -> ConceptExtraction:
        """抽取概念和实体"""
        try:
            chain = self._concept_prompt | self.llm | self._concept_parser
            result = await chain.ainvoke(
                {
                    "content": content,
                    "max_concepts": self.config.indexing_config.max_concepts_per_memory,
                }
            )
            return result
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            return ConceptExtraction(concepts=[], keywords=[], entities=[])

    async def _extract_questions(self, content: str) -> QuestionExtraction:
        """抽取问题"""
        try:
            chain = self._question_prompt | self.llm | self._question_parser
            result = await chain.ainvoke(
                {
                    "content": content,
                    "max_questions": self.config.indexing_config.max_questions_per_memory,
                }
            )
            return result
        except Exception as e:
            print(f"Error extracting questions: {e}")
            return QuestionExtraction(questions=[])

    async def _generate_summary(self, content: str) -> SummaryGeneration:
        """生成摘要"""
        try:
            chain = self._summary_prompt | self.llm | self._summary_parser
            result = await chain.ainvoke({"content": content})
            return result
        except Exception as e:
            print(f"Error generating summary: {e}")
            return SummaryGeneration(summary="", key_points=[])

    def should_process_memory(self, memory: MemoryRecord) -> Dict[str, bool]:
        """判断记忆是否需要各种类型的处理"""
        config = self.config.indexing_config
        importance = memory.metadata.importance_score

        return {
            "concepts": (
                config.enable_concept_extraction
                and importance >= config.concept_extraction_threshold
            ),
            "questions": (
                config.enable_question_extraction
                and importance >= config.question_extraction_threshold
            ),
            "summary": config.enable_summary_generation,
        }

    async def reprocess_memory(
        self, memory: MemoryRecord, processing_types: List[str]
    ) -> MemoryRecord:
        """重新处理记忆的特定索引类型"""
        config = self.config.indexing_config

        if "concepts" in processing_types and config.enable_concept_extraction:
            concept_result = await self._extract_concepts(memory.content)
            memory.indexes.concepts = concept_result.concepts[
                : config.max_concepts_per_memory
            ]
            memory.indexes.keywords = concept_result.keywords

        if "questions" in processing_types and config.enable_question_extraction:
            question_result = await self._extract_questions(memory.content)
            memory.indexes.questions = question_result.questions[
                : config.max_questions_per_memory
            ]

        if "summary" in processing_types and config.enable_summary_generation:
            summary_result = await self._generate_summary(memory.content)
            memory.indexes.summary = summary_result.summary

        return memory

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        # 简化实现，实际应该记录处理计数、耗时等
        return {
            "total_processed": 0,
            "avg_processing_time_ms": 0.0,
            "success_rate": 1.0,
            "enabled_features": {
                "concept_extraction": self.config.indexing_config.enable_concept_extraction,
                "question_extraction": self.config.indexing_config.enable_question_extraction,
                "summary_generation": self.config.indexing_config.enable_summary_generation,
            },
        }


class StreamingIndexProcessor:
    """流式索引处理器，用于实时处理"""

    def __init__(self, base_processor: IndexingProcessor):
        self.base_processor = base_processor
        self._processing_queue = asyncio.Queue()
        self._is_running = False

    async def start_processing(self):
        """开始后台处理"""
        self._is_running = True
        while self._is_running:
            try:
                memory = await asyncio.wait_for(
                    self._processing_queue.get(), timeout=1.0
                )
                await self.base_processor.process_memory(memory)
                self._processing_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in streaming processor: {e}")

    async def submit_memory(self, memory: MemoryRecord):
        """提交记忆到处理队列"""
        await self._processing_queue.put(memory)

    async def stop_processing(self):
        """停止处理"""
        self._is_running = False
        await self._processing_queue.join()

    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self._processing_queue.qsize()

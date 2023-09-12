# -*- coding:utf-8 -*-

import json
from textwrap import wrap
from typing import Dict, List, Optional
from queue import Queue
from pydantic import BaseModel

from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel, BaseLanguageModel
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import BaseChatMessageHistory
from langchain.vectorstores import VectorStore
from langchain.vectorstores.base import VectorStoreRetriever
from arxiv_paper_chatbot.paper import Paper, PaperSection

_summarize_with_language_prompt = PromptTemplate(
    template_format="jinja2",
    template="""Write a concise summary of the following:


"{{ text }}"


ALWAYS RESPOND WITH {{ language }}.
CONCISE SUMMARY:""",
    input_variables=["text", "language"])


def _send_progress(progress_queue: Optional[Queue], msg: str):
    if progress_queue:
        progress_queue.put(msg)


def _get_keywords(llm: BaseLanguageModel, language: str, content: str, count: int) -> List[str]:
    prompt = PromptTemplate(
        input_variables=["content", "count", "language"],
        template_format="jinja2",
        template="""
CONTENT:
{{content}}

Give me {{count}} keywords that describes content well.
ALWAYS ANSWER WITH {{language}}.
ANSWER's format are always json type containing keywords, as follows:
{
    "keywords": [word 1, word 2, ..., word n]
}

ANSWER:"""
    )
    retrieve_keyword_chain = LLMChain(llm=llm, prompt=prompt)
    raw_keywords = retrieve_keyword_chain.run(content=content, count=count, language=language)
    return json.loads(raw_keywords)["keywords"]


def _get_significant_keywords(llm: BaseLanguageModel, language: str, keywords: Dict[str, int], count: int) -> List[str]:
    if not keywords:
        return []
    prompt = PromptTemplate(
        input_variables=["keywords", "count", "language"],
        template_format="jinja2",
        template="""
"KEYWORD MAPPINGS" is the list of (keyword: count for each keyword).
Bigger keyword count means that the keyword likely to contain significant keyword.

KEYWORD_MAPPINGS:
{{keywords}}

Based on the KEYWORD_MAPPINGS, give me the most significant {{count}} keywords that describes CONTENT WELL.
ALWAYS ANSWER WITH {{language}}.
ANSWER's format are always json type containing keywords, as follows:
{
    "keywords": [word 1, word 2, ..., word n]
}

ANSWER:""",
    )
    retrieve_keyword_chain = LLMChain(llm=llm, prompt=prompt)
    raw_keywords = retrieve_keyword_chain.run(
        language=language,
        keywords="\n".join([f"({w}:{cnt})" for w, cnt in keywords.items()]),
        count=count,
    )
    return json.loads(raw_keywords)["keywords"]


def _build_section_summaries(llm: BaseChatModel, paper: Paper,
                             knowledge_retriever: VectorStoreRetriever,
                             language: str, progress_queue: Optional[Queue]) -> List[str]:
    # summarize by chunks
    stuff_chain = load_summarize_chain(llm=llm, chain_type="stuff",
                                       prompt=_summarize_with_language_prompt)
    approx_chunk_size = 4096
    if not paper.sections:
        summaries: List[str] = []
        _send_progress(progress_queue, "summarizing paper chunks")
        chunked = wrap(paper.raw, width=approx_chunk_size)
        for chunk in chunked:
            summary = stuff_chain.run(
                input_documents=[Document(page_content=chunk)],
                language=language)
            summaries.append(summary)
            _send_progress(progress_queue, f"chunk summary: {summary}")
            knowledge_retriever.add_documents(documents=[Document(page_content=summary)])
        return summaries

    # summarize sections
    summaries: List[str] = []
    _send_progress(progress_queue, "summarizing paper sections")
    section_mappings = (
            [
                ("OVERVIEW", paper.sections.overview),
                ("ABSTRACT", paper.sections.abstract),
                ("INTRODUCTION", paper.sections.introduction),
            ]
            + [(section.title.content, section) for section in paper.sections.details]
            + [("CONCLUSION", paper.sections.conclusion)]
    )
    for title, section in section_mappings:
        if not section:
            continue
        section_chunk_summaries, section_summary = _summarize_section(
            llm=llm, language=language, section=section, approx_chunk_size=approx_chunk_size)
        if not section_chunk_summaries and not section_summary:
            continue
        _send_progress(progress_queue, f"section summary of {title}:")
        for summ in section_chunk_summaries:
            _send_progress(progress_queue, f"- {summ}")
        _send_progress(progress_queue, f"{title} section summary: {section_summary}")
        knowledge_retriever.add_documents(
            documents=[Document(page_content=section_summary)]
                      + [Document(page_content=s) for s in section_chunk_summaries]
        )
        summaries.append(f"{title}:\n{section_summary}")
    return summaries


def _summarize_section(llm: BaseLanguageModel, language: str,
                       section: PaperSection, approx_chunk_size: int) -> (List[str], str):
    chunk_summ_chain = load_summarize_chain(llm=llm, chain_type="stuff",
                                            prompt=_summarize_with_language_prompt)
    total_summ_chain = load_summarize_chain(llm=llm, chain_type="map_reduce",
                                            map_prompt=_summarize_with_language_prompt,
                                            combine_prompt=_summarize_with_language_prompt)
    total_summary = ""
    section_chunk_summaries = []
    chunked = section.chunked_elems_text(approx_chunk_size)
    for chunk in chunked:
        summary = chunk_summ_chain.run(
            input_documents=[Document(page_content=chunk)],
            language=language)
        section_chunk_summaries.append(summary)

    if section_chunk_summaries:
        summ_docs = [Document(page_content=t) for t in section_chunk_summaries]
        total_summary = total_summ_chain.run(input_documents=summ_docs,
                                             language=language,
                                             return_only_outputs=True)
    return section_chunk_summaries, total_summary


def _build_total_summary(llm: BaseChatModel, language: str,
                         section_summaries: List[str], progress_queue: Optional[Queue]) -> str:
    if not section_summaries:
        return ""
    _send_progress(progress_queue, f"calculating total summary. #{len(section_summaries)} sections.")
    total_summary_chain = load_summarize_chain(llm, chain_type="map_reduce",
                                               map_prompt=_summarize_with_language_prompt,
                                               combine_prompt=_summarize_with_language_prompt, )
    return total_summary_chain.run(
        input_documents=[Document(page_content=t) for t in section_summaries],
        language=language,
        return_only_outputs=True)


def _build_keywords(llm: BaseLanguageModel, language: str, summaries: List[str], progress_queue: Optional[Queue]) -> \
List[str]:
    _send_progress(progress_queue, "retrieving all keywords")
    summary_keywords = [_get_keywords(llm=llm, content=t, count=4, language=language) for t in summaries]
    _send_progress(progress_queue, f"all keywords: {summary_keywords}")
    keywords_with_count: Dict[str, int] = {}
    for ks in summary_keywords:
        for k in ks:
            keywords_with_count[k] = keywords_with_count.get(k, 0) + 1
    _send_progress(progress_queue, "retrieving significant keywords")
    return _get_significant_keywords(llm=llm, language=language, keywords=keywords_with_count, count=4)


class PaperChatbot(BaseModel):
    llm: BaseChatModel
    knowledge_retriever: VectorStoreRetriever
    paper: Paper
    section_summaries: List[str]
    keywords: List[str]
    total_summary: str

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def load(cls, progress_queue: Optional[Queue], llm: BaseChatModel,
             paper: Paper, knowledge_vector_store: VectorStore, language: str,
             with_keywords: bool = False) -> 'PaperChatbot':
        knowledge_reference_size = 5
        knowledge_vector_store = knowledge_vector_store
        knowledge_retriever: VectorStoreRetriever = knowledge_vector_store.as_retriever(
            search_kwargs=dict(k=knowledge_reference_size)
        )
        section_summaries = _build_section_summaries(llm=llm,
                                                     paper=paper,
                                                     knowledge_retriever=knowledge_retriever,
                                                     language=language,
                                                     progress_queue=progress_queue)
        _send_progress(progress_queue, f"section summaries: {section_summaries}")
        total_summary = _build_total_summary(llm=llm,
                                             language=language,
                                             section_summaries=section_summaries,
                                             progress_queue=progress_queue)
        _send_progress(progress_queue, f"total summary: {total_summary}")
        keywords=[]
        if with_keywords:
            keywords = _build_keywords(llm=llm,
                                       language=language,
                                       summaries=section_summaries,
                                       progress_queue=progress_queue)
            _send_progress(progress_queue, f"significant keywords: {keywords}")
        return cls(llm=llm, knowledge_retriever=knowledge_retriever, paper=paper,
                   keywords=keywords, section_summaries=section_summaries, total_summary=total_summary)

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.json(exclude={"knowledge_retriever"}), f, indent=2)

    def overview(self) -> str:
        msgs = [f"Paper Title:\n {self.paper.title}",  f"Summary:\n {self.total_summary}"]
        if self.keywords:
            msgs.append(f"Keywords:\n {self.keywords}")
        return "\n\n".join(msgs)

    def answer(self, progress_queue: Optional[Queue],
               language: str, query: str, chat_history: BaseChatMessageHistory) -> str:
        _send_progress(progress_queue, "retrieving related docs")
        docs = self.knowledge_retriever.get_relevant_documents(query)
        _send_progress(progress_queue, "\n".join(["related docs:"] + [f"\t - {doc.page_content}" for doc in docs]))
        history_limit = 3
        trimmed_history = chat_history.messages[-history_limit:]
        history_summary = "\n".join([f"{item.type}: {item.content}" for item in trimmed_history])
        knowledge_summary = "\n".join([doc.page_content for doc in docs])
        _send_progress(progress_queue, "retrieving answer from llm with knowledge")
        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "knowledge", "user_query", "language"],
                template_format="jinja2",
                template="""
CHAT HISTORY:
{{ chat_history }}

KNOWLEDGES:
{{ knowledge }}


Based on the KNOWLEDGES and CHAT HISTORY, ANSWER the following USER QUERY:
USER QUERY: {{ user_query }}
ALWAYS ANSWER WITH {{ language }}

ANSWER:"""))
        return chain.run(
            language=language, chat_history=history_summary, knowledge=knowledge_summary, user_query=query)

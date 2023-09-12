# -*- coding:utf-8 -*-

import gradio as gr
import langcodes

import arxiv
import ast
import os
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

import faiss
import openai
import threading
from queue import Queue

from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
from langchain.memory.chat_memory import ChatMessageHistory

from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from py_pdf_parser.loaders import load_file

from pydantic import BaseModel

from arxiv_paper_chatbot.chatbot import PaperChatbot
from arxiv_paper_chatbot.parser import PaperParser
from arxiv_paper_chatbot.paper import Paper

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseChatMessageHistory


def select(*queues):
    combined = Queue(maxsize=0)

    def listen_and_forward(queue):
        while True:
            combined.put((queue, queue.get()))

    for queue in queues:
        t = threading.Thread(target=listen_and_forward, args=(queue,))
        t.daemon = True
        t.start()
    while True:
        yield combined.get()


class OpenAIChatHandler(BaseModel, BaseCallbackHandler):
    call_count_limit: int = 100
    call_count: int = 0

    def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        print("on new token")
        self.call_count += 1
        if self.call_count > self.call_count_limit:
            raise Exception("call count limit exceeded")

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                            **kwargs: Any) -> Any:
        pass


def load_paper(arxiv_paper_id: str) -> Paper:
    try:
        search_results = arxiv.Search(id_list=[arxiv_paper_id]).get()
        if not search_results:
            raise RuntimeError("arxiv paper not found")
        paper = next(search_results)
    except Exception as e:
        raise RuntimeError("cannot fetch arxiv paper", e)
    #    load / parse paper pdf
    pdf_file = paper.download_pdf(dirpath='/tmp/')
    try:
        return PaperParser().parse(load_file(pdf_file))
    except Exception as e:
        raise Exception("cannot load / parse paper", e)
    finally:
        if os.path.exists(pdf_file):
            os.remove(pdf_file)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column():
                openai_api_key = gr.Textbox(label="Openai API Key")
                arxiv_paper_id = gr.Textbox(label="Arxiv Paper ID")
            with gr.Column():
                language = gr.Textbox(label="Language", value="english")
                openai_call_limit = gr.Number(label="OpenAI Call limit", value=100)

    def _create_chatbot(openai_api_key: str, arxiv_paper_id: str,
                        language: str = "english", llm_call_limit: int = 100) -> Optional[PaperChatbot]:
        try:
            if not openai_api_key or not arxiv_paper_id:
                gr.Error("fill out the inputs")
                return None
            print('downloading paper')
            paper = load_paper(arxiv_paper_id)
            print('downloading paper done')

            def _load_chatbot(openai_api_key: str, paper: Paper, language: str,
                              chat_handler: OpenAIChatHandler, progress_queue: Queue, done_queue: Queue):
                openai.api_key = openai_api_key
                os.environ["OPENAI_API_KEY"] = openai_api_key
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, callbacks=[chat_handler])

                embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
                index = faiss.IndexFlatL2(embedding_size)
                embedding_fn = OpenAIEmbeddings().embed_query
                vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

                chatbot = PaperChatbot.load(llm=llm, progress_queue=progress_queue,
                                            knowledge_vector_store=vectorstore,
                                            paper=paper, language=language)
                time.sleep(0.5)  # interval to resolve progress queue message
                done_queue.put(chatbot)

            # run background
            chat_handler = OpenAIChatHandler(call_count_limit=llm_call_limit)
            chatbot_progress_queue = Queue(maxsize=0)
            done_queue = Queue(maxsize=0)
            threading.Thread(target=_load_chatbot,
                             kwargs={"openai_api_key": openai_api_key,
                                     "paper": paper,
                                     "language": language,
                                     "chat_handler": chat_handler,
                                     "progress_queue": chatbot_progress_queue,
                                     "done_queue": done_queue}).start()
            for which, data in select(chatbot_progress_queue, done_queue):
                if which is chatbot_progress_queue:
                    print(data)
                elif which is done_queue:
                    return data
            loading_progress(1)
            return None
        except Exception as e:
            raise gr.Error(str(e))

    btn = gr.Button("Create Bot")
    chatbot = gr.State(None)
    loading_progress = gr.Progress()

    btn.click(_create_chatbot,
              inputs=[openai_api_key, arxiv_paper_id, language, openai_call_limit],
              outputs=[chatbot])

    print("HI!!")
    print(chatbot, chatbot.value)
    if chatbot and chatbot.value:
        user_msg = gr.Textbox()
        gr_chatbot = gr.Chatbot()
        clear = gr.ClearButton([user_msg, chatbot])
        chatbot_history_state = gr.State([])  # {type: ..., content: ..}

        def respond(message, chat_history_state: List[BaseMessage], gr_chatbot):
            if not chatbot.value:
                gr.Error("chatbot not created")
                return
            history = ChatMessageHistory(messages=chat_history_state)
            answer = chatbot.value.answer(input=message, chat_history=history)
            history.add_user_message(message)
            history.add_ai_message(answer)

            gr_chatbot = gr_chatbot + [(message, answer)]
            return "", chat_history_state, gr_chatbot

        user_msg.submit(respond,
                        inputs=[user_msg, chatbot_history_state, gr_chatbot],
                        outputs=[user_msg, chatbot_history_state, gr_chatbot])

demo.launch()

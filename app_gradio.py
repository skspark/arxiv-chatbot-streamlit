import gradio as gr

import os
from typing import Any, Dict, List, Optional
from uuid import UUID

import faiss
import openai
import threading
from queue import Queue

from langchain.schema import  BaseMessage
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.vectorstores import FAISS
from py_pdf_parser.loaders import load_file

# TODO: to package
from arxiv_paper_chatbot.chatbot import PaperChatbot
from arxiv_paper_chatbot.parser import PaperParser

from langchain.callbacks.base import BaseCallbackHandler


class OpenAIChatHandler(BaseCallbackHandler):
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


# TODO: cache
def create_chatbot(openai_api_key: str, arxiv_paper_id: str):
    if not openai_api_key or not arxiv_paper_id:
        raise RuntimeError("Please fill in above text fields")
    try:
        pass
        # search_results = arxiv.Search(id_list=[arxiv_paper_id]).get()
        # if not search_results:
        #     raise RuntimeError("arxiv paper not found")
        # paper = next(search_results)
    except Exception as e:
        raise RuntimeError("cannot fetch arxiv paper", e)

    # load / parse paper pdf
    # pdf_file = paper.download_pdf()
    pdf_file = "arxiv_paper_chatbot/2306.15577.pdf"
    try:
        paper = PaperParser().parse(load_file(pdf_file))
    except Exception as e:
        raise Exception("cannot load / parse paper", e)
    finally:
        pass
        # TODO: uncomment it
        # if os.path.exists(pdf_file):
        #     os.remove(pdf_file)

    # create paper chatbot
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, callbacks=[OpenAIChatHandler()])

    # setup vectorstore
    embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
    index = faiss.IndexFlatL2(embedding_size)
    embedding_fn = OpenAIEmbeddings().embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

    chatbot_progress_queue = Queue(maxsize=0)
    chatbot = PaperChatbot(llm=llm, knowledge_vector_store=vectorstore, paper=paper)
    chatbot.load(chatbot_progress_queue)

    # run background
    t = threading.Thread(target=chatbot.load, kwargs={"progress_queue": chatbot_progress_queue})



    # arxiv-expert #jmoonga


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            openai_api_key = gr.Textbox(label="Openai API Key")
            openai_org_id = gr.Textbox(label="Openai Org ID")
            arxiv_paper_id = gr.Textbox(label="Arxiv Paper ID")

    chatbot = gr.State(None)

    def _create_chatbot(openai_api_key, arxiv_paper_id) -> Optional[PaperChatbot]:
        try:
            if not openai_api_key or not arxiv_paper_id:
                gr.Error("fill out the inputs")
                return
            loading_progress = gr.Progress()
            item = create_chatbot(openai_api_key, arxiv_paper_id)
            loading_progress.update(100)
            return item
        except Exception as e:
            raise gr.Error(str(e))


    btn = gr.Button("Create Bot")
    btn.click(_create_chatbot,
              inputs=[openai_api_key, arxiv_paper_id],
              outputs=[chatbot])

    if chatbot and chatbot.value:
        msg = gr.Textbox()
        gr_chatbot = gr.Chatbot()
        clear = gr.ClearButton([msg, chatbot])
        chatbot_history_state = gr.State([])  # {type: ..., content: ..}

        def respond(message, chat_history_state: List[BaseMessage], gr_chatbot):
            history = ChatMessageHistory(messages=chat_history_state)
            answer = chatbot.answer(input=message, chat_history=history)
            history.add_user_message(message)
            history.add_ai_message(answer)

            gr_chatbot = gr_chatbot + [(message, answer)]
            return "", chat_history_state, gr_chatbot

        msg.submit(respond,
                   inputs=[msg, chatbot_history_state, gr_chatbot],
                   outputs=[msg, chatbot_history_state, gr_chatbot])

demo.launch()

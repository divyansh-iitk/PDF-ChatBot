from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from logger import logging
from utils.config import LLM_GroqConfig

import os
from dotenv import load_dotenv
load_dotenv()


class GroqLLM:

    def __init__(self, model_name: str = LLM_GroqConfig.model_name):
        
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            temperature=LLM_GroqConfig.temperature,
            max_tokens=LLM_GroqConfig.max_tokens
        )

        # Move prompt here
        self.prompt_template1 = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
        You are a helpful AI assistant.

        Use ONLY the provided context and chat history to answer the question.

        Context:
        {context}
        
        Chat History:
        {chat_history}

        Question: {question}

        Answer:
        - Be concise
        - If unsure, say "I don't know based on the provided context"
        """
                )
        
        self.prompt_template2 = PromptTemplate(
            input_variables=["question", "chat_history"],
            template="""
        Chat History:
        {chat_history}

        Current Question:
        {question}

        Rewrite the current question into a standalone question using context from the chat history.

        Rules:
        - Do not answer the question.
        - Do not add information not present in the chat history.
        - Return only the rewritten standalone question.
        - If no rewriting is needed, return the original question.
        """
                )

    def _format_context(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    
    def rewrite_query(self, query: str, chat_history):
        """Rewriting query based on chat history.

        Args:
            query (str): _description_
            chat_history (_type_): _description_
            llm (_type_): _description_
        """
        formatted_prompt = self.prompt_template2.format(
            question=query,
            chat_history=chat_history
        )
        
        messages = [
            SystemMessage(content="You return strictly only the rewritten standalone question"),
            HumanMessage(content=formatted_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logging.error(f"LLM error while rewriting query: {e}")
            raise
        

    def generate_response(self, query: str, retrieved_docs, chat_history):

        context = self._format_context(retrieved_docs)

        # Safety trim
        # context = context[:4000]

        formatted_prompt = self.prompt_template1.format(
            context=context,
            question=query,
            chat_history=chat_history
        )

        messages = [
            SystemMessage(content="You answer strictly from context."),
            HumanMessage(content=formatted_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logging.error(f"LLM error: {e}")
            raise
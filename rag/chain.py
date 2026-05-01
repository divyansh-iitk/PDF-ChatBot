from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from logger import logging

load_dotenv()


class GroqLLM:

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=1024
        )

        # Move prompt here
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        You are a helpful AI assistant.

        Use ONLY the provided context to answer the question.

        Context:
        {context}

        Question: {question}

        Answer:
        - Be concise
        - If unsure, say "I don't know based on the provided context"
        """
                )

    def _format_context(self, docs):
        return "\n\n".join([doc["content"] for doc in docs])

    def generate_response(self, query: str, retrieved_docs):

        context = self._format_context(retrieved_docs)

        # Safety trim
        # context = context[:4000]

        formatted_prompt = self.prompt_template.format(
            context=context,
            question=query
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
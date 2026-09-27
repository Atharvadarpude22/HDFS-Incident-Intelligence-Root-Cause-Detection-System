import logging
import os
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HDFS incident investigation assistant. Answer ONLY from the "
     "troubleshooting context. If it is insufficient, say the evidence is "
     "insufficient. Never invent evidence."),
    ("human",
     "User Incident:\n{query}\n\nTroubleshooting Context:\n{context}\n\n"
     "Provide:\n1. Possible root cause\n2. Supporting evidence\n"
     "3. Recommended investigation steps\n4. Recommended action"),
])

NO_CONTEXT_MESSAGE = (
    "No relevant troubleshooting information was found in the knowledge base."
)


class LLMGenerationError(RuntimeError):
    """Raised when the LLM fails after all retries."""


def build_llm(repo_id="mistralai/Mistral-7B-Instruct-v0.3"):
    """Needs env var HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN)."""
    if not (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")):
        raise EnvironmentError(
            "Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) to use the HuggingFace LLM."
        )
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id, task="text-generation",
        max_new_tokens=512, temperature=0.1, timeout=60,
    )
    return ChatHuggingFace(llm=endpoint)


class RAGPipeline:
    def __init__(self, retriever, llm, max_retries=3, backoff=1.0,
                 max_context_chars=6000):
        self.retriever = retriever
        self.chain = PROMPT | llm | StrOutputParser()
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_context_chars = max_context_chars
        self.last_context = []
        self.last_evidence = []

    def retrieve_context(self, query):
        self.last_context = self.retriever.retrieve(query)
        return self.last_context

    def format_context(self, documents):
        parts, total = [], 0
        for i, d in enumerate(documents):
            block = f"[Source {i + 1}: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            if total + len(block) > self.max_context_chars and parts:
                break  # keep the prompt within budget
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

    def _generate(self, inputs):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                answer = self.chain.invoke(inputs)
                if not answer or not answer.strip():
                    raise ValueError("LLM returned an empty answer.")
                return answer
            except Exception as exc:  # network, rate limit, auth, timeout...
                last_error = exc
                logger.warning("LLM attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.backoff * 2 ** (attempt - 1))
        raise LLMGenerationError(
            f"LLM failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def answer(self, query):
        docs = self.retrieve_context(query)
        if not docs:
            return NO_CONTEXT_MESSAGE
        return self._generate({"query": query, "context": self.format_context(docs)})

    def get_evidence(self, query):
        docs = self.retrieve_context(query)
        self.last_evidence = [
            {
                "source": d.metadata.get("source", "unknown"),
                "chunk_id": d.metadata.get("chunk_id"),
                "dense_distance": d.metadata.get("dense_distance"),
                "bm25_score": d.metadata.get("bm25_score"),
                "content": d.page_content,
            }
            for d in docs
        ]
        return self.last_evidence
from typing import List, Optional
from toolpickr.core.tool import ToolDefinition
from toolpickr.core.registry import ToolRegistry
from toolpickr.embeddings.gemini import GeminiEmbeddings
from toolpickr.embeddings.renderer import render_tool_text
from toolpickr.vectorstores.faiss import FaissVectorStore
from toolpickr.retrieval.flat import FlatRetriever
from toolpickr.vectorstores.base import SearchResult

class ToolPickr:
    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initializes the entire ToolPickr pipeline."""
        self.registry = ToolRegistry()
        self.embeddings = GeminiEmbeddings(api_key=gemini_api_key)
        self.vector_store = FaissVectorStore(dimension=768)
        self.retriever = FlatRetriever(self.embeddings, self.vector_store)
        self._is_built = False

    def register_tool(self, tool: ToolDefinition) -> None:
        """Registers a single tool."""
        self.registry.register_tool(tool)

    def build(self) -> None:
        """Embeds all registered tools and builds the search index."""
        tools = self.registry.get_all_tools()
        if not tools:
            raise ValueError("No tools registered! Cannot build index.")

        names = []
        texts = []
        
        # Format each tool into a recognizable string
        for tool in tools:
            names.append(tool.name)
            texts.append(render_tool_text(tool))
            
        # Get vectors from Gemini (batching is faster)
        print(f"Embedding {len(tools)} tools via Gemini...")
        vectors = self.embeddings.embed_batch(texts)
        
        # Save them in FAISS
        self.vector_store.add_vectors(vectors=vectors, tool_names=names)
        self._is_built = True
        print("ToolPickr build complete!")

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Finds the best tools for a given user query."""
        if not self._is_built:
            raise RuntimeError("You must call .build() before retrieving tools!")
            
        return self.retriever.retrieve(query, top_k)

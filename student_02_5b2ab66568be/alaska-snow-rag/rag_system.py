"""
Alaska Emergency Services RAG System
Simplified implementation with better error handling and structure
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from google.cloud import bigquery
import google.generativeai as genai
from config import PROJECT_ID, GEMINI_API_KEY, DATASET_NAME, TABLE_NAME, EMBEDDING_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    project_id: str = PROJECT_ID
    dataset_name: str = DATASET_NAME
    table_name: str = TABLE_NAME
    embedding_model: str = EMBEDDING_MODEL
    gemini_model: str = "gemini-1.5-pro"
    top_k: int = 3
    search_fraction: float = 0.01

class AlaskaRAGSystem:
    """Alaska Emergency Services RAG System"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.bq_client = None
        self.ai_model = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize all services"""
        self._setup_bigquery()
        self._setup_gemini()
    
    def _setup_bigquery(self) -> None:
        """Initialize BigQuery client"""
        try:
            self.bq_client = bigquery.Client(project=self.config.project_id)
            logger.info(f"✅ BigQuery connected to project: {self.config.project_id}")
        except Exception as e:
            logger.error(f"❌ BigQuery connection failed: {e}")
            self.bq_client = None
    
    def _setup_gemini(self) -> None:
        """Initialize Gemini AI model"""
        try:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found")
            
            genai.configure(api_key=GEMINI_API_KEY)
            self.ai_model = genai.GenerativeModel(self.config.gemini_model)
            logger.info("✅ Gemini model initialized")
            
        except Exception as e:
            logger.error(f"❌ Gemini setup failed: {e}")
            self.ai_model = None
    
    def is_ready(self) -> bool:
        """Check if system is ready for queries"""
        return self.bq_client is not None and self.ai_model is not None
    
    def search_context(self, question: str) -> Optional[str]:
        """Search for relevant context using vector similarity"""
        if not self.bq_client:
            logger.warning("BigQuery not available")
            return None
        
        query = self._build_search_query(question)
        
        try:
            job = self.bq_client.query(query)
            results = list(job.result())
            
            if not results:
                logger.info("No matching content found")
                return None
            
            # Combine multiple results if available
            contexts = [row.content for row in results[:self.config.top_k]]
            return "\n\n".join(contexts)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None
    
    def _build_search_query(self, question: str) -> str:
        """Build vector search query for BigQuery"""
        return f"""
        SELECT
            query.query,
            base.content
        FROM
            VECTOR_SEARCH(
                TABLE `{self.config.dataset_name}.{self.config.table_name}`,
                'ml_generate_embedding_result',
                (
                    SELECT
                        ml_generate_embedding_result,
                        content AS query
                    FROM
                        ML.GENERATE_EMBEDDING(
                            MODEL `{self.config.embedding_model}`,
                            (SELECT @question AS content)
                        )
                ),
                top_k => {self.config.top_k},
                options => '{{"fraction_lists_to_search": {self.config.search_fraction}}}'
            )
        """
    
    def generate_answer(self, question: str, context: str = None) -> dict:
        """Generate answer using AI model with optional context"""
        if not self.ai_model:
            return {
                "answer": "AI service is currently unavailable. Please try again later.",
                "context_found": False,
                "validation_status": "error"
            }
        
        # Search for context if not provided
        if context is None:
            context = self.search_context(question)
        
        context_found = context is not None
        prompt = self._build_prompt(question, context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            answer = response.text.strip()
            
            return {
                "answer": answer,
                "context_found": context_found,
                "validation_status": "success"
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "context_found": context_found,
                "validation_status": "error"
            }
    
    def _build_prompt(self, question: str, context: str = None) -> str:
        """Build prompt for AI model"""
        if context:
            return f"""You are an Alaska Emergency Services assistant. Answer the user's question using ONLY the provided information below.

Guidelines:
- Be helpful and accurate
- If the information isn't in the context, say you don't have that specific information
- Keep answers concise but complete
- Focus on emergency services, snow removal, road conditions, and safety procedures

Context Information:
{context}

User Question: {question}

Answer:"""
        else:
            return f"""You are an Alaska Emergency Services assistant. The user asked a question but no specific context was found.

Provide a helpful general response about Alaska emergency services and suggest they:
1. Contact local emergency services for urgent matters
2. Check official Alaska Department websites
3. Call 511 for road conditions

User Question: {question}

Answer:"""

# Convenience functions for backward compatibility
def initialize_services() -> Tuple[Optional[bigquery.Client], Optional[genai.GenerativeModel]]:
    """Legacy function for backward compatibility"""
    rag = AlaskaRAGSystem()
    return rag.bq_client, rag.ai_model

def search_knowledge_base(bq_client: bigquery.Client, user_question: str) -> Optional[str]:
    """Legacy function for backward compatibility"""
    config = RAGConfig()
    rag = AlaskaRAGSystem(config)
    rag.bq_client = bq_client  # Use provided client
    return rag.search_context(user_question)

def generate_response(model: genai.GenerativeModel, user_question: str, context: str) -> str:
    """Legacy function for backward compatibility"""
    config = RAGConfig()
    rag = AlaskaRAGSystem(config)
    rag.ai_model = model  # Use provided model
    result = rag.generate_answer(user_question, context)
    return result["answer"]

# Main RAG class instance for easy import
alaska_rag = AlaskaRAGSystem()

if __name__ == "__main__":
    # Test the system
    rag = AlaskaRAGSystem()
    
    if rag.is_ready():
        print("🚀 Alaska RAG System is ready!")
        
        # Test query
        test_question = "What are snow removal procedures?"
        result = rag.generate_answer(test_question)
        
        print(f"\nTest Question: {test_question}")
        print(f"Answer: {result['answer']}")
        print(f"Context Found: {result['context_found']}")
        print(f"Status: {result['validation_status']}")
    else:
        print("❌ System not ready - check configuration")
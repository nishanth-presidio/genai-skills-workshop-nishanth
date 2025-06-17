"""
Enhanced FastAPI backend for Alaska Emergency Services RAG system
Clean architecture with improved error handling and monitoring
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import os

# Import our enhanced modules
from rag_system import AlaskaRAGSystem, alaska_rag
from prompt_validator import AlaskaPromptValidator, alaska_validator, ValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global services
rag_system: Optional[AlaskaRAGSystem] = None
validator: Optional[AlaskaPromptValidator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    logger.info("🚀 Starting Alaska Emergency Services API...")
    
    global rag_system, validator
    try:
        rag_system = alaska_rag
        validator = alaska_validator
        
        # Check service health
        rag_ready = rag_system.is_ready()
        validator_ready = validator.is_ready()
        
        logger.info(f"RAG System: {'✅ Ready' if rag_ready else '❌ Not Ready'}")
        logger.info(f"Validator: {'✅ Ready' if validator_ready else '❌ Not Ready'}")
        
        if not rag_ready:
            logger.warning("RAG system not fully ready - some features may be limited")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Don't fail startup - allow graceful degradation
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Alaska Emergency Services API...")

# Initialize FastAPI app
app = FastAPI(
    title="Alaska Emergency Services API",
    description="RAG-powered API for Alaska emergency services, snow removal, and safety information",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://alaska-emergency-services.com",  # Production domain
        "*"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response Models
class QuestionRequest(BaseModel):
    """Request model for questions"""
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class ValidationInfo(BaseModel):
    """Validation information"""
    is_valid: bool
    result_type: str
    message: str
    confidence: float = 0.0
    suggestions: List[str] = []

class QuestionResponse(BaseModel):
    """Enhanced response model"""
    question: str
    answer: str
    context_found: bool
    validation: ValidationInfo
    processing_time: float
    timestamp: str
    service_status: Dict[str, bool]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, bool]
    timestamp: str

class SampleQuestionsResponse(BaseModel):
    """Sample questions response"""
    categories: Dict[str, List[str]]
    total_count: int

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled error: {exc} - {request.url}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    services = {
        "rag_system": rag_system.is_ready() if rag_system else False,
        "validator": validator.is_ready() if validator else False,
        "database": rag_system.bq_client is not None if rag_system else False,
        "ai_model": rag_system.ai_model is not None if rag_system else False
    }
    
    overall_status = "healthy" if any(services.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="2.0.0",
        services=services,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Process emergency services questions through the RAG system
    
    Enhanced flow:
    1. Validate question for safety and relevance
    2. Search knowledge base for context
    3. Generate AI response
    4. Return structured response with metadata
    """
    start_time = time.time()
    question = request.question
    
    # Check service availability
    if not rag_system or not validator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services are currently initializing. Please try again in a moment."
        )
    
    try:
        # Step 1: Enhanced validation
        validation_result = validator.validate(question)
        
        if not validation_result.is_valid:
            processing_time = time.time() - start_time
            
            return QuestionResponse(
                question=question,
                answer=validation_result.message,
                context_found=False,
                validation=ValidationInfo(
                    is_valid=False,
                    result_type=validation_result.result_type.value,
                    message=validation_result.message,
                    confidence=validation_result.confidence,
                    suggestions=validation_result.suggestions
                ),
                processing_time=processing_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                service_status={
                    "rag_available": rag_system.is_ready(),
                    "validator_available": True
                }
            )
        
        # Step 2: Generate answer using RAG system
        rag_result = rag_system.generate_answer(question)
        processing_time = time.time() - start_time
        
        return QuestionResponse(
            question=question,
            answer=rag_result["answer"],
            context_found=rag_result["context_found"],
            validation=ValidationInfo(
                is_valid=True,
                result_type=ValidationResult.VALID.value,
                message="Question validated and processed successfully",
                confidence=validation_result.confidence
            ),
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            service_status={
                "rag_available": rag_system.is_ready(),
                "validator_available": validator.is_ready()
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing question '{question}': {e}")
        processing_time = time.time() - start_time
        
        return QuestionResponse(
            question=question,
            answer="I'm experiencing technical difficulties. Please try again later or contact support for urgent matters.",
            context_found=False,
            validation=ValidationInfo(
                is_valid=True,
                result_type="error",
                message=f"Processing error: {str(e)}"
            ),
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            service_status={
                "rag_available": False,
                "validator_available": validator.is_ready() if validator else False
            }
        )

@app.get("/sample-questions", response_model=SampleQuestionsResponse)
async def get_sample_questions():
    """Get categorized sample questions for Alaska emergency services"""
    categories = {
        "snow_removal": [
            "What are the snow removal procedures?",
            "How quickly are main roads cleared after snowfall?",
            "Which roads are prioritized for snow plowing?"
        ],
        "road_conditions": [
            "How do I report hazardous road conditions?",
            "What are the current road closures?",
            "How do I check road conditions before traveling?"
        ],
        "emergency_procedures": [
            "What are the winter emergency protocols?",
            "When do emergency shelters open?",
            "How do I prepare for winter storms?"
        ],
        "safety_information": [
            "What are winter driving safety tips?",
            "How do I stay safe during extreme cold?",
            "What should I include in an emergency kit?"
        ]
    }
    
    total_count = sum(len(questions) for questions in categories.values())
    
    return SampleQuestionsResponse(
        categories=categories,
        total_count=total_count
    )

@app.get("/status")
async def detailed_status():
    """Detailed system status for monitoring"""
    return {
        "api_version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "services": {
            "rag_system": {
                "available": rag_system.is_ready() if rag_system else False,
                "bigquery_connected": rag_system.bq_client is not None if rag_system else False,
                "ai_model_ready": rag_system.ai_model is not None if rag_system else False
            },
            "validator": {
                "available": validator.is_ready() if validator else False,
                "model_loaded": validator.validator_model is not None if validator else False
            }
        },
        "environment": {
            "project_id": os.getenv("PROJECT_ID", "not_set"),
            "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
            "dataset_configured": bool(os.getenv("DATASET_NAME"))
        }
    }

@app.get("/validate")
async def validate_question_endpoint(question: str):
    """Standalone validation endpoint for testing"""
    if not validator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Validator service not available"
        )
    
    result = validator.validate(question)
    
    return {
        "question": question,
        "is_valid": result.is_valid,
        "result_type": result.result_type.value,
        "message": result.message,
        "confidence": result.confidence,
        "suggestions": result.suggestions
    }

# Development server
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting development server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
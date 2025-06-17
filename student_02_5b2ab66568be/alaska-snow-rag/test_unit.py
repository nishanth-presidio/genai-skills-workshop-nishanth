"""
Comprehensive Unit Tests for Alaska Emergency Services RAG System
Tests all components with proper mocking and fixtures
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from dataclasses import dataclass
from typing import Dict, List

# Import our enhanced modules
from rag_system import AlaskaRAGSystem, RAGConfig
from prompt_validator import AlaskaPromptValidator, ValidationResult, ValidationResponse
from evaluation_service import AlaskaRAGEvaluator, EvaluationResult

# Test data
MOCK_FAQ_CONTENT = {
    "snow_removal": "Snow removal procedures follow priority routes: 1) Emergency access roads, 2) Main arterials, 3) Secondary roads. Equipment includes plows, sanders, and salt trucks operating 24/7 during storm events.",
    "road_hazards": "Report road hazards by calling 511 or using the Alaska DOT online reporting system. Include location details, hazard type, and severity level.",
    "emergency_shelters": "Emergency shelters open when temperatures drop below -20°F or during severe weather warnings. Locations include community centers, schools, and designated warming centers.",
    "winter_safety": "Winter driving safety: Use studded tires or chains, carry emergency kit, check 511 for conditions, reduce speed on ice, and maintain safe following distance."
}

SAMPLE_QUESTIONS = [
    {
        "question": "What are snow removal procedures?",
        "expected_context": "snow_removal",
        "should_be_valid": True
    },
    {
        "question": "How do I report road hazards?", 
        "expected_context": "road_hazards",
        "should_be_valid": True
    },
    {
        "question": "Buy cheap Bitcoin now!",
        "expected_context": None,
        "should_be_valid": False
    },
    {
        "question": "",
        "expected_context": None,
        "should_be_valid": False
    }
]

class TestRAGConfig:
    """Test RAG configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RAGConfig()
        
        assert config.gemini_model == "gemini-1.5-pro"
        assert config.top_k == 3
        assert config.search_fraction == 0.01
        assert hasattr(config, 'project_id')
        assert hasattr(config, 'dataset_name')
        assert hasattr(config, 'table_name')

    def test_custom_config(self):
        """Test custom configuration"""
        config = RAGConfig(
            project_id="test-project",
            top_k=5,
            search_fraction=0.02
        )
        
        assert config.project_id == "test-project"
        assert config.top_k == 5
        assert config.search_fraction == 0.02

class TestAlaskaRAGSystem:
    """Test the enhanced RAG system"""
    
    @pytest.fixture
    def mock_bq_client(self):
        """Mock BigQuery client"""
        mock_client = Mock()
        mock_row = Mock()
        mock_row.content = MOCK_FAQ_CONTENT["snow_removal"]
        
        mock_job = Mock()
        mock_job.result.return_value = [mock_row]
        mock_client.query.return_value = mock_job
        
        return mock_client
    
    @pytest.fixture
    def mock_ai_model(self):
        """Mock AI model"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Based on Alaska emergency procedures, snow removal follows these steps..."
        mock_model.generate_content.return_value = mock_response
        
        return mock_model
    
    @pytest.fixture
    def rag_system(self, mock_bq_client, mock_ai_model):
        """Create RAG system with mocked dependencies"""
        with patch('rag_system.bigquery.Client') as mock_bq_class, \
             patch('rag_system.genai.configure') as mock_configure, \
             patch('rag_system.genai.GenerativeModel') as mock_genai_class:
            
            mock_bq_class.return_value = mock_bq_client
            mock_genai_class.return_value = mock_ai_model
            
            rag = AlaskaRAGSystem()
            return rag
    
    def test_initialization_success(self, rag_system):
        """Test successful RAG system initialization"""
        assert rag_system.bq_client is not None
        assert rag_system.ai_model is not None
        assert rag_system.is_ready() is True
    
    @patch('rag_system.bigquery.Client')
    @patch('rag_system.genai.configure')
    def test_initialization_bigquery_failure(self, mock_configure, mock_bq_client):
        """Test RAG system with BigQuery failure"""
        mock_bq_client.side_effect = Exception("BigQuery connection failed")
        
        rag = AlaskaRAGSystem()
        
        assert rag.bq_client is None
        assert rag.is_ready() is False
    
    @patch('rag_system.genai.configure')
    @patch('rag_system.genai.GenerativeModel')
    def test_initialization_gemini_failure(self, mock_genai_model, mock_configure):
        """Test RAG system with Gemini failure"""
        mock_configure.side_effect = Exception("Gemini API failed")
        
        rag = AlaskaRAGSystem()
        
        assert rag.ai_model is None
        assert rag.is_ready() is False
    
    def test_search_context_success(self, rag_system):
        """Test successful context search"""
        result = rag_system.search_context("What are snow removal procedures?")
        
        assert result is not None
        assert "snow removal" in result.lower()
        rag_system.bq_client.query.assert_called_once()
    
    def test_search_context_no_results(self, rag_system):
        """Test context search with no results"""
        # Mock empty results
        mock_job = Mock()
        mock_job.result.return_value = []
        rag_system.bq_client.query.return_value = mock_job
        
        result = rag_system.search_context("Unknown question")
        
        assert result is None
    
    def test_search_context_error(self, rag_system):
        """Test context search with error"""
        rag_system.bq_client.query.side_effect = Exception("Query failed")
        
        result = rag_system.search_context("Test question")
        
        assert result is None
    
    def test_generate_answer_with_context(self, rag_system):
        """Test answer generation with context"""
        result = rag_system.generate_answer(
            "What are snow removal procedures?",
            MOCK_FAQ_CONTENT["snow_removal"]
        )
        
        assert result["answer"] is not None
        assert result["context_found"] is True
        assert result["validation_status"] == "success"
        assert len(result["answer"]) > 0
    
    def test_generate_answer_without_context(self, rag_system):
        """Test answer generation without context"""
        # Mock search to return None
        with patch.object(rag_system, 'search_context', return_value=None):
            result = rag_system.generate_answer("Unknown question")
        
        assert result["answer"] is not None
        assert result["context_found"] is False
        assert result["validation_status"] == "success"
    
    def test_generate_answer_ai_error(self, rag_system):
        """Test answer generation with AI error"""
        rag_system.ai_model.generate_content.side_effect = Exception("AI API failed")
        
        result = rag_system.generate_answer("Test question", "Test context")
        
        assert "error" in result["answer"].lower()
        assert result["validation_status"] == "error"
    
    def test_generate_answer_no_ai_model(self, rag_system):
        """Test answer generation without AI model"""
        rag_system.ai_model = None
        
        result = rag_system.generate_answer("Test question")
        
        assert "unavailable" in result["answer"].lower()
        assert result["validation_status"] == "error"
    
    def test_build_search_query(self, rag_system):
        """Test search query building"""
        query = rag_system._build_search_query("test question")
        
        assert "VECTOR_SEARCH" in query
        assert "ML.GENERATE_EMBEDDING" in query
        assert f"top_k => {rag_system.config.top_k}" in query

class TestAlaskaPromptValidator:
    """Test the enhanced prompt validator"""
    
    @pytest.fixture
    def mock_validator_model(self):
        """Mock validator model"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "This question is appropriate for emergency services."
        
        mock_candidate = Mock()
        mock_candidate.finish_reason.name = 'STOP'
        mock_response.candidates = [mock_candidate]
        
        mock_model.generate_content.return_value = mock_response
        return mock_model
    
    @pytest.fixture
    def validator(self, mock_validator_model):
        """Create validator with mocked dependencies"""
        with patch('prompt_validator.genai.configure') as mock_configure, \
             patch('prompt_validator.genai.GenerativeModel') as mock_genai_class:
            
            mock_genai_class.return_value = mock_validator_model
            
            validator = AlaskaPromptValidator()
            return validator
    
    def test_initialization_success(self, validator):
        """Test successful validator initialization"""
        assert validator.validator_model is not None
        assert validator.is_ready() is True
    
    @patch('prompt_validator.genai.configure')
    def test_initialization_failure(self, mock_configure):
        """Test validator initialization failure"""
        mock_configure.side_effect = Exception("API key not found")
        
        validator = AlaskaPromptValidator()
        
        assert validator.validator_model is None
        assert validator.is_ready() is False
    
    def test_validate_empty_prompt(self, validator):
        """Test validation of empty prompt"""
        result = validator.validate("")
        
        assert result.is_valid is False
        assert result.result_type == ValidationResult.ERROR
        assert "empty" in result.message.lower()
    
    def test_validate_spam_patterns(self, validator):
        """Test spam pattern detection"""
        spam_prompts = [
            "Buy cheap Bitcoin now!",
            "Click here for amazing deals",
            "Visit http://spam-site.com",
            "Adult content available"
        ]
        
        for prompt in spam_prompts:
            result = validator.validate(prompt)
            
            assert result.is_valid is False
            assert result.result_type == ValidationResult.BLOCKED_SPAM
            assert len(result.suggestions) > 0
    
    def test_validate_irrelevant_content(self, validator):
        """Test irrelevant content detection"""
        irrelevant_prompts = [
            "What's the best pizza in New York?",
            "How do I fix my computer?",
            "Tell me about football scores"
        ]
        
        for prompt in irrelevant_prompts:
            result = validator.validate(prompt)
            
            # Should be blocked as irrelevant
            assert result.is_valid is False
            assert result.result_type == ValidationResult.BLOCKED_IRRELEVANT
            assert len(result.suggestions) > 0
    
    def test_validate_relevant_content(self, validator):
        """Test validation of relevant Alaska emergency content"""
        relevant_prompts = [
            "What are snow removal procedures?",
            "How do I report road hazards?",
            "Emergency shelter information",
            "Winter safety tips for Alaska"
        ]
        
        for prompt in relevant_prompts:
            result = validator.validate(prompt)
            
            assert result.is_valid is True
            assert result.result_type == ValidationResult.VALID
    
    def test_validate_safety_blocked(self, validator):
        """Test safety blocking by AI model"""
        # Mock safety-blocked response
        mock_candidate = Mock()
        mock_candidate.finish_reason.name = 'SAFETY'
        
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        
        validator.validator_model.generate_content.return_value = mock_response
        
        result = validator.validate("Potentially harmful content")
        
        assert result.is_valid is False
        assert result.result_type == ValidationResult.BLOCKED_SAFETY
    
    def test_validate_ai_error_handling(self, validator):
        """Test AI validation error handling"""
        validator.validator_model.generate_content.side_effect = Exception("Safety filter triggered")
        
        result = validator.validate("Test question about Alaska emergency")
        
        # Should fail open for availability
        assert result.is_valid is True
        assert result.confidence < 1.0
    
    def test_check_relevance_scoring(self, validator):
        """Test relevance scoring logic"""
        # High relevance
        high_relevance = validator._check_relevance("Alaska snow removal emergency procedures")
        assert high_relevance.is_valid is True
        assert high_relevance.confidence > 0.5
        
        # Low relevance
        low_relevance = validator._check_relevance("Pizza delivery service")
        assert low_relevance.is_valid is False

class TestAlaskaRAGEvaluator:
    """Test the evaluation service"""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG system for evaluation"""
        mock_rag = Mock()
        mock_rag.is_ready.return_value = True
        mock_rag.generate_answer.return_value = {
            "answer": "Snow removal follows priority procedures...",
            "context_found": True,
            "validation_status": "success"
        }
        return mock_rag
    
    @pytest.fixture
    def mock_validator(self):
        """Mock validator for evaluation"""
        mock_validator = Mock()
        mock_validation = ValidationResponse(
            is_valid=True,
            result_type=ValidationResult.VALID,
            message="Valid question"
        )
        mock_validator.validate.return_value = mock_validation
        return mock_validator
    
    @pytest.fixture
    def evaluator(self, mock_rag_system, mock_validator):
        """Create evaluator with mocked dependencies"""
        evaluator = AlaskaRAGEvaluator()
        evaluator.rag_system = mock_rag_system
        evaluator.validator = mock_validator
        return evaluator
    
    def test_evaluate_single_question_success(self, evaluator):
        """Test successful single question evaluation"""
        result = evaluator.evaluate_single_question(
            "What are snow removal procedures?",
            "Reference answer about snow removal"
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.question == "What are snow removal procedures?"
        assert result.context_found is True
        assert result.validation_status == "passed"
        assert result.processing_time > 0
    
    def test_evaluate_single_question_validation_blocked(self, evaluator):
        """Test evaluation with blocked validation"""
        # Mock blocked validation
        blocked_validation = ValidationResponse(
            is_valid=False,
            result_type=ValidationResult.BLOCKED_SPAM,
            message="Spam detected"
        )
        evaluator.validator.validate.return_value = blocked_validation
        
        result = evaluator.evaluate_single_question("Buy Bitcoin now!")
        
        assert result.validation_status == "blocked"
        assert result.context_found is False
    
    def test_evaluate_single_question_error(self, evaluator):
        """Test evaluation with system error"""
        evaluator.rag_system.generate_answer.side_effect = Exception("System error")
        
        result = evaluator.evaluate_single_question("Test question")
        
        assert result.validation_status == "error"
        assert "error" in result.rag_response.lower()
    
    def test_calculate_local_scores(self, evaluator):
        """Test local scoring calculation"""
        test_result = EvaluationResult(
            question="Test question",
            rag_response="This is a comprehensive answer about Alaska snow removal procedures and emergency protocols.",
            reference_answer="Reference answer",
            context_found=True,
            validation_status="passed",
            processing_time=1.5
        )
        
        scores = evaluator.calculate_local_scores(test_result)
        
        assert "response_length_score" in scores
        assert "context_found_score" in scores
        assert "validation_score" in scores
        assert "efficiency_score" in scores
        assert "error_free_score" in scores
        assert "alaska_relevance_score" in scores
        
        # Check score ranges
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_generate_summary(self, evaluator):
        """Test evaluation summary generation"""
        test_results = [
            EvaluationResult("Q1", "A1", "R1", True, "passed", 1.0),
            EvaluationResult("Q2", "A2", "R2", False, "blocked", 0.5),
            EvaluationResult("Q3", "A3", "R3", True, "passed", 2.0)
        ]
        
        summary = evaluator.generate_summary(test_results, "test_dataset")
        
        assert summary.total_questions == 3
        assert summary.successful_responses == 2
        assert summary.context_found_rate == 2/3
        assert summary.validation_pass_rate == 2/3
        assert summary.dataset_name == "test_dataset"
        assert len(summary.overall_scores) > 0

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def full_system(self):
        """Create a complete mocked system"""
        with patch('rag_system.bigquery.Client') as mock_bq, \
             patch('rag_system.genai.configure'), \
             patch('rag_system.genai.GenerativeModel') as mock_genai, \
             patch('prompt_validator.genai.configure'), \
             patch('prompt_validator.genai.GenerativeModel') as mock_validator:
            
            # Setup mocks
            mock_bq_instance = Mock()
            mock_row = Mock()
            mock_row.content = MOCK_FAQ_CONTENT["snow_removal"]
            mock_job = Mock()
            mock_job.result.return_value = [mock_row]
            mock_bq_instance.query.return_value = mock_job
            mock_bq.return_value = mock_bq_instance
            
            mock_ai_model = Mock()
            mock_response = Mock()
            mock_response.text = "Alaska snow removal procedures include..."
            mock_ai_model.generate_content.return_value = mock_response
            mock_genai.return_value = mock_ai_model
            
            mock_validator_model = Mock()
            mock_val_response = Mock()
            mock_val_response.text = "Appropriate question"
            mock_candidate = Mock()
            mock_candidate.finish_reason.name = 'STOP'
            mock_val_response.candidates = [mock_candidate]
            mock_validator_model.generate_content.return_value = mock_val_response
            mock_validator.return_value = mock_validator_model
            
            return {
                'rag': AlaskaRAGSystem(),
                'validator': AlaskaPromptValidator(),
                'evaluator': AlaskaRAGEvaluator()
            }
    
    def test_end_to_end_flow(self, full_system):
        """Test complete end-to-end question processing"""
        rag = full_system['rag']
        validator = full_system['validator']
        
        question = "What are snow removal procedures?"
        
        # Step 1: Validate
        validation_result = validator.validate(question)
        assert validation_result.is_valid is True
        
        # Step 2: Generate answer
        rag_result = rag.generate_answer(question)
        assert rag_result["answer"] is not None
        assert rag_result["context_found"] is True
        assert rag_result["validation_status"] == "success"
    
    def test_evaluation_workflow(self, full_system):
        """Test evaluation workflow"""
        evaluator = full_system['evaluator']
        evaluator.rag_system = full_system['rag']
        evaluator.validator = full_system['validator']
        
        test_questions = [
            {"question": "What are snow removal procedures?", "reference_answer": "Snow removal reference"}
        ]
        
        results = evaluator.evaluate_dataset(test_questions, "integration_test")
        
        assert len(results) == 1
        assert results[0].validation_status == "passed"
        assert results[0].context_found is True

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

# Test runner
if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--cov=rag_system",
        "--cov=prompt_validator", 
        "--cov=evaluation_service",
        "--cov-report=html"
    ])
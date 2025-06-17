"""
Enhanced Prompt Validation System for Alaska Emergency Services
Comprehensive safety and relevance validation with detailed feedback
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import google.generativeai as genai
from config import GEMINI_API_KEY, SAFETY_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Validation result types"""
    VALID = "valid"
    BLOCKED_SAFETY = "blocked_safety"
    BLOCKED_IRRELEVANT = "blocked_irrelevant"
    BLOCKED_SPAM = "blocked_spam"
    ERROR = "error"

@dataclass
class ValidationResponse:
    """Structured validation response"""
    is_valid: bool
    result_type: ValidationResult
    message: str
    confidence: float = 0.0
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

class AlaskaPromptValidator:
    """Enhanced prompt validator for Alaska Emergency Services"""
    
    # Keywords related to Alaska emergency services
    RELEVANT_KEYWORDS = {
        'emergency', 'snow', 'winter', 'road', 'weather', 'ice', 'storm',
        'shelter', 'evacuation', 'rescue', 'safety', 'hazard', 'alaska',
        'cold', 'frozen', 'blizzard', 'avalanche', 'earthquake', 'fire',
        'medical', 'police', 'ambulance', 'hospital', 'closure', 'alert',
        'warning', 'advisory', 'condition', 'temperature', 'visibility'
    }
    
    # Spam/irrelevant patterns
    SPAM_PATTERNS = [
        r'\b(?:buy|sell|purchase|price|money|cost|cheap|free|discount)\b',
        r'\b(?:bitcoin|crypto|investment|loan|mortgage)\b',
        r'\b(?:sex|adult|porn|dating|casino|gambling)\b',
        r'http[s]?://\S+',  # URLs
        r'\b(?:click here|subscribe|follow me)\b'
    ]
    
    def __init__(self):
        self.validator_model = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the Gemini validator model"""
        try:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found")
            
            genai.configure(api_key=GEMINI_API_KEY)
            
            self.validator_model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                safety_settings=SAFETY_SETTINGS
            )
            
            logger.info("✅ Prompt validator initialized")
            
        except Exception as e:
            logger.error(f"❌ Validator initialization failed: {e}")
            self.validator_model = None
    
    def is_ready(self) -> bool:
        """Check if validator is ready"""
        return self.validator_model is not None
    
    def validate(self, prompt: str) -> ValidationResponse:
        """
        Comprehensive prompt validation
        
        Args:
            prompt: User input to validate
            
        Returns:
            ValidationResponse with validation results
        """
        # Basic input validation
        if not prompt or not prompt.strip():
            return ValidationResponse(
                is_valid=False,
                result_type=ValidationResult.ERROR,
                message="Empty prompt provided"
            )
        
        prompt = prompt.strip()
        
        # Check for spam/irrelevant content first
        spam_check = self._check_spam_patterns(prompt)
        if not spam_check.is_valid:
            return spam_check
        
        # Check relevance to Alaska emergency services
        relevance_check = self._check_relevance(prompt)
        if not relevance_check.is_valid:
            return relevance_check
        
        # AI safety validation
        if self.validator_model:
            safety_check = self._ai_safety_validation(prompt)
            if not safety_check.is_valid:
                return safety_check
        
        # If all checks pass
        return ValidationResponse(
            is_valid=True,
            result_type=ValidationResult.VALID,
            message="Prompt is valid and safe",
            confidence=0.95
        )
    
    def _check_spam_patterns(self, prompt: str) -> ValidationResponse:
        """Check for spam/inappropriate patterns"""
        prompt_lower = prompt.lower()
        
        for pattern in self.SPAM_PATTERNS:
            if re.search(pattern, prompt_lower):
                return ValidationResponse(
                    is_valid=False,
                    result_type=ValidationResult.BLOCKED_SPAM,
                    message="Your message appears to contain inappropriate content. Please ask about Alaska emergency services.",
                    suggestions=[
                        "Ask about snow removal procedures",
                        "Inquire about road conditions",
                        "Request emergency shelter information"
                    ]
                )
        
        return ValidationResponse(
            is_valid=True,
            result_type=ValidationResult.VALID,
            message="No spam patterns detected"
        )
    
    def _check_relevance(self, prompt: str) -> ValidationResponse:
        """Check if prompt is relevant to Alaska emergency services"""
        prompt_lower = prompt.lower()
        
        # Count relevant keywords
        relevant_words = sum(1 for keyword in self.RELEVANT_KEYWORDS 
                           if keyword in prompt_lower)
        
        # Basic relevance threshold
        if relevant_words == 0 and len(prompt.split()) > 3:
            # Use simple heuristics for common emergency-related words
            emergency_indicators = ['help', 'emergency', 'urgent', 'problem', 'issue', 'report']
            if not any(word in prompt_lower for word in emergency_indicators):
                return ValidationResponse(
                    is_valid=False,
                    result_type=ValidationResult.BLOCKED_IRRELEVANT,
                    message="I can only help with Alaska emergency services. Please ask about snow removal, road conditions, or emergency procedures.",
                    suggestions=[
                        "What are the current road conditions?",
                        "How do I report a road hazard?",
                        "When do emergency shelters open?",
                        "What are winter driving safety tips?"
                    ]
                )
        
        confidence = min(relevant_words / 3.0, 1.0)  # Normalize to 0-1
        
        return ValidationResponse(
            is_valid=True,
            result_type=ValidationResult.VALID,
            message="Content appears relevant",
            confidence=confidence
        )
    
    def _ai_safety_validation(self, prompt: str) -> ValidationResponse:
        """Use AI model for safety validation"""
        try:
            # Create a safe test prompt
            test_prompt = f"""Analyze this question about Alaska emergency services: "{prompt}"

Is this appropriate for an emergency services assistant?"""
            
            response = self.validator_model.generate_content(test_prompt)
            
            # Check if response was blocked for safety
            if (response.candidates and 
                hasattr(response.candidates[0], 'finish_reason') and
                response.candidates[0].finish_reason.name == 'SAFETY'):
                
                return ValidationResponse(
                    is_valid=False,
                    result_type=ValidationResult.BLOCKED_SAFETY,
                    message="Your question contains content that cannot be processed. Please rephrase focusing on emergency services.",
                    suggestions=[
                        "Ask about emergency procedures",
                        "Inquire about safety protocols",
                        "Request information about services"
                    ]
                )
            
            # Additional safety checks on the response content
            if response.text and any(flag in response.text.lower() 
                                   for flag in ['inappropriate', 'unsafe', 'harmful']):
                return ValidationResponse(
                    is_valid=False,
                    result_type=ValidationResult.BLOCKED_SAFETY,
                    message="Please rephrase your question to focus on Alaska emergency services."
                )
            
            return ValidationResponse(
                is_valid=True,
                result_type=ValidationResult.VALID,
                message="AI safety check passed",
                confidence=0.9
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if any(keyword in error_msg for keyword in ['safety', 'block', 'harmful']):
                return ValidationResponse(
                    is_valid=False,
                    result_type=ValidationResult.BLOCKED_SAFETY,
                    message="Your question was flagged for safety reasons. Please ask about emergency services in a different way."
                )
            
            logger.warning(f"AI validation error: {e}")
            # Don't block on technical errors - fail open for availability
            return ValidationResponse(
                is_valid=True,
                result_type=ValidationResult.VALID,
                message="Safety validation unavailable, proceeding with caution",
                confidence=0.5
            )
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics (placeholder for future implementation)"""
        return {
            "total_validations": 0,
            "blocked_safety": 0,
            "blocked_irrelevant": 0,
            "blocked_spam": 0,
            "passed": 0
        }

# Global validator instance
alaska_validator = AlaskaPromptValidator()

# Legacy functions for backward compatibility
def initialize_validator():
    """Legacy function for backward compatibility"""
    validator = AlaskaPromptValidator()
    return validator.validator_model

def validate_prompt(validator_model, prompt: str) -> Tuple[bool, str]:
    """Legacy function for backward compatibility"""
    # Use global validator regardless of passed model
    result = alaska_validator.validate(prompt)
    return result.is_valid, result.message

# Enhanced validation function
def validate_user_input(prompt: str) -> ValidationResponse:
    """
    Enhanced validation function that returns detailed results
    
    Args:
        prompt: User input to validate
        
    Returns:
        ValidationResponse with detailed validation results
    """
    return alaska_validator.validate(prompt)

if __name__ == "__main__":
    # Test the validation system
    validator = AlaskaPromptValidator()
    
    test_prompts = [
        "What are snow removal procedures?",
        "How do I report a road hazard?",
        "Buy cheap bitcoin now!",
        "What's the weather like?",
        "Emergency shelter locations?",
        ""
    ]
    
    print("🧪 Testing Alaska Prompt Validator\n")
    
    for prompt in test_prompts:
        result = validator.validate(prompt)
        print(f"Input: '{prompt}'")
        print(f"Valid: {result.is_valid}")
        print(f"Type: {result.result_type.value}")
        print(f"Message: {result.message}")
        if result.suggestions:
            print(f"Suggestions: {', '.join(result.suggestions)}")
        print("-" * 50)
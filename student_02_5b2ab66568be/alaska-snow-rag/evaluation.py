"""
Enhanced Evaluation Service for Alaska Emergency Services RAG System
Comprehensive evaluation with multiple metrics and detailed reporting
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np

# Google Cloud imports
try:
    import vertexai
    from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples
    from google.cloud import aiplatform
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logging.warning("Vertex AI not available - using local evaluation only")

# Import our enhanced RAG system
from rag_system import AlaskaRAGSystem, alaska_rag
from prompt_validator import AlaskaPromptValidator, alaska_validator
from config import PROJECT_ID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question: str
    rag_response: str
    reference_answer: str
    context_found: bool
    validation_status: str
    processing_time: float
    scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.scores is None:
            self.scores = {}

@dataclass
class EvaluationSummary:
    """Summary of evaluation results"""
    total_questions: int
    successful_responses: int
    average_processing_time: float
    context_found_rate: float
    validation_pass_rate: float
    overall_scores: Dict[str, float]
    timestamp: str
    dataset_name: str

class AlaskaRAGEvaluator:
    """Comprehensive evaluation system for Alaska RAG"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.rag_system = alaska_rag
        self.validator = alaska_validator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Vertex AI if available
        if VERTEX_AI_AVAILABLE:
            try:
                vertexai.init(project=PROJECT_ID, location="us-central1")
                self.vertex_ai_ready = True
            except Exception as e:
                logger.warning(f"Vertex AI initialization failed: {e}")
                self.vertex_ai_ready = False
        else:
            self.vertex_ai_ready = False
    
    def evaluate_single_question(self, question: str, reference_answer: str = None) -> EvaluationResult:
        """Evaluate a single question through the RAG system"""
        start_time = time.time()
        
        try:
            # Validate question
            validation_result = self.validator.validate(question)
            
            if not validation_result.is_valid:
                return EvaluationResult(
                    question=question,
                    rag_response=validation_result.message,
                    reference_answer=reference_answer or "N/A",
                    context_found=False,
                    validation_status="blocked",
                    processing_time=time.time() - start_time
                )
            
            # Generate RAG response
            rag_result = self.rag_system.generate_answer(question)
            processing_time = time.time() - start_time
            
            return EvaluationResult(
                question=question,
                rag_response=rag_result["answer"],
                reference_answer=reference_answer or "N/A",
                context_found=rag_result["context_found"],
                validation_status="passed",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error evaluating question '{question}': {e}")
            return EvaluationResult(
                question=question,
                rag_response=f"Error: {str(e)}",
                reference_answer=reference_answer or "N/A",
                context_found=False,
                validation_status="error",
                processing_time=time.time() - start_time
            )
    
    def evaluate_dataset(self, questions: List[Dict[str, str]], dataset_name: str = "alaska_eval") -> List[EvaluationResult]:
        """Evaluate a complete dataset"""
        logger.info(f"🚀 Starting evaluation of {len(questions)} questions...")
        
        results = []
        for i, item in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {item['question'][:50]}...")
            
            result = self.evaluate_single_question(
                question=item["question"],
                reference_answer=item.get("reference_answer")
            )
            results.append(result)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        logger.info("✅ Dataset evaluation complete!")
        return results
    
    def calculate_local_scores(self, result: EvaluationResult) -> Dict[str, float]:
        """Calculate local evaluation scores"""
        scores = {}
        
        # Response length score (reasonable length indicator)
        response_length = len(result.rag_response.split())
        scores["response_length_score"] = min(response_length / 50.0, 1.0)  # Normalize to 0-1
        
        # Context utilization score
        scores["context_found_score"] = 1.0 if result.context_found else 0.0
        
        # Validation success score
        scores["validation_score"] = 1.0 if result.validation_status == "passed" else 0.0
        
        # Processing efficiency score (inverse of processing time, capped)
        scores["efficiency_score"] = max(0.0, 1.0 - (result.processing_time / 10.0))
        
        # Error detection score
        error_indicators = ["error", "sorry", "couldn't", "unavailable", "try again"]
        has_error = any(indicator in result.rag_response.lower() for indicator in error_indicators)
        scores["error_free_score"] = 0.0 if has_error else 1.0
        
        # Alaska relevance score (basic keyword matching)
        alaska_keywords = ["alaska", "snow", "winter", "road", "emergency", "safety"]
        relevance_count = sum(1 for keyword in alaska_keywords if keyword in result.rag_response.lower())
        scores["alaska_relevance_score"] = min(relevance_count / 3.0, 1.0)
        
        return scores
    
    def generate_summary(self, results: List[EvaluationResult], dataset_name: str) -> EvaluationSummary:
        """Generate evaluation summary"""
        successful_responses = sum(1 for r in results if r.validation_status == "passed" and r.context_found)
        context_found_count = sum(1 for r in results if r.context_found)
        validation_passed_count = sum(1 for r in results if r.validation_status == "passed")
        
        # Calculate average scores
        all_scores = {}
        for result in results:
            local_scores = self.calculate_local_scores(result)
            for metric, score in local_scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        average_scores = {metric: np.mean(scores) for metric, scores in all_scores.items()}
        
        return EvaluationSummary(
            total_questions=len(results),
            successful_responses=successful_responses,
            average_processing_time=np.mean([r.processing_time for r in results]),
            context_found_rate=context_found_count / len(results) if results else 0,
            validation_pass_rate=validation_passed_count / len(results) if results else 0,
            overall_scores=average_scores,
            timestamp=datetime.now().isoformat(),
            dataset_name=dataset_name
        )
    
    def save_results(self, results: List[EvaluationResult], summary: EvaluationSummary):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_df = pd.DataFrame([asdict(result) for result in results])
        results_file = self.output_dir / f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save summary
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"📄 Summary saved to: {summary_file}")
        
        return results_file, summary_file
    
    def print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print(f"🏔️  ALASKA RAG EVALUATION SUMMARY")
        print("="*60)
        print(f"📊 Dataset: {summary.dataset_name}")
        print(f"⏰ Timestamp: {summary.timestamp}")
        print(f"📝 Total Questions: {summary.total_questions}")
        print(f"✅ Successful Responses: {summary.successful_responses} ({summary.successful_responses/summary.total_questions*100:.1f}%)")
        print(f"🔍 Context Found Rate: {summary.context_found_rate:.2%}")
        print(f"✔️  Validation Pass Rate: {summary.validation_pass_rate:.2%}")
        print(f"⚡ Avg Processing Time: {summary.average_processing_time:.2f}s")
        
        print(f"\n📈 Local Evaluation Scores:")
        for metric, score in summary.overall_scores.items():
            print(f"   {metric}: {score:.3f}")
        print("="*60)
    
    def run_vertex_ai_evaluation(self, results: List[EvaluationResult], experiment_name: str) -> Optional[Dict]:
        """Run Vertex AI evaluation if available"""
        if not self.vertex_ai_ready:
            logger.warning("Vertex AI not available - skipping AI evaluation")
            return None
        
        try:
            # Prepare dataset for Vertex AI
            eval_data = []
            for result in results:
                if result.validation_status == "passed" and result.context_found:
                    eval_data.append({
                        "instruction": "Answer questions about Alaska emergency services based on the provided context.",
                        "context": f"Question: {result.question}",
                        "response": result.rag_response,
                        "reference": result.reference_answer
                    })
            
            if not eval_data:
                logger.warning("No valid data for Vertex AI evaluation")
                return None
            
            eval_df = pd.DataFrame(eval_data)
            
            # Create evaluation task
            eval_task = EvalTask(
                dataset=eval_df,
                metrics=[
                    MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
                    MetricPromptTemplateExamples.Pointwise.INSTRUCTION_FOLLOWING,
                    MetricPromptTemplateExamples.Pointwise.SAFETY,
                    MetricPromptTemplateExamples.Pointwise.SUMMARIZATION_QUALITY
                ],
                experiment=experiment_name
            )
            
            # Run evaluation
            logger.info("🔄 Running Vertex AI evaluation...")
            result = eval_task.evaluate(
                prompt_template="Instruction: {instruction}\nContext: {context}\nResponse: {response}",
                experiment_run_name=f"{experiment_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            
            logger.info("✅ Vertex AI evaluation complete!")
            return result
            
        except Exception as e:
            logger.error(f"Vertex AI evaluation failed: {e}")
            return None
    
    def run_comprehensive_evaluation(self, questions: List[Dict[str, str]], dataset_name: str = "alaska_eval"):
        """Run comprehensive evaluation with all available methods"""
        logger.info(f"🚀 Starting comprehensive evaluation: {dataset_name}")
        
        # Check system readiness
        if not self.rag_system.is_ready():
            logger.error("❌ RAG system not ready!")
            return
        
        # Run local evaluation
        results = self.evaluate_dataset(questions, dataset_name)
        summary = self.generate_summary(results, dataset_name)
        
        # Save results
        results_file, summary_file = self.save_results(results, summary)
        
        # Print summary
        self.print_summary(summary)
        
        # Run Vertex AI evaluation if available
        if self.vertex_ai_ready:
            vertex_result = self.run_vertex_ai_evaluation(results, f"alaska-rag-{dataset_name}")
            if vertex_result:
                print(f"\n🤖 Vertex AI evaluation results available in Google Cloud Console")
        
        return results, summary

# Sample evaluation questions
SAMPLE_EVALUATION_QUESTIONS = [
    {
        "question": "What are snow removal procedures?",
        "reference_answer": "Snow removal procedures include prioritizing main roads, using salt and sand for ice control, and following specific plowing schedules based on snowfall severity."
    },
    {
        "question": "How do I report road hazards?",
        "reference_answer": "Road hazards can be reported by calling 511 or using the official Alaska DOT website reporting system."
    },
    {
        "question": "When do emergency shelters open?",
        "reference_answer": "Emergency shelters typically open when temperatures drop below -20°F or during severe weather warnings as declared by local emergency management."
    },
    {
        "question": "What are winter driving safety tips?",
        "reference_answer": "Winter driving safety includes using winter tires, carrying emergency supplies, checking road conditions, and driving slowly in icy conditions."
    },
    {
        "question": "How quickly are main roads cleared after snowfall?",
        "reference_answer": "Main roads are typically cleared within 4-6 hours after snowfall ends, depending on severity and equipment availability."
    }
]

def run_quick_evaluation():
    """Run a quick evaluation with sample questions"""
    evaluator = AlaskaRAGEvaluator()
    return evaluator.run_comprehensive_evaluation(
        SAMPLE_EVALUATION_QUESTIONS[:3], 
        "quick_test"
    )

def run_full_evaluation():
    """Run full evaluation with all sample questions"""
    evaluator = AlaskaRAGEvaluator()
    return evaluator.run_comprehensive_evaluation(
        SAMPLE_EVALUATION_QUESTIONS,
        "full_evaluation"
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("🏃‍♂️ Running quick evaluation...")
        run_quick_evaluation()
    elif len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("🏔️ Running full evaluation...")
        run_full_evaluation()
    else:
        print("Usage:")
        print("  python evaluation_service.py --quick   # Quick test with 3 questions")
        print("  python evaluation_service.py --full    # Full evaluation with all questions")
        print("  python evaluation_service.py           # Show this help")
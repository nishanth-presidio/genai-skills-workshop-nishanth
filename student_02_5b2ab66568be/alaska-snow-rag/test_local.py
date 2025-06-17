#!/usr/bin/env python3
"""
Test script for Alaska FAQ RAG system with hardcoded questions
"""

from rag_system import initialize_services, search_knowledge_base, generate_response
from prompt_validator import initialize_validator, validate_prompt

def test_rag_system():
    """Test the complete RAG system with hardcoded questions"""
    
    # Hardcoded test questions
    TEST_QUESTIONS = [
        "What are the snow removal procedures?",
        "What equipment is used for snow removal?",
        "What are the emergency protocols during blizzards?",
        "How do I report hazardous road conditions?",
        "What is the cricket score?"
    ]
    
    print("🚀 Starting Alaska FAQ RAG System Test\n")
    
    # Initialize all services
    print("📋 Initializing services...")
    bq_client, genai_model = initialize_services()
    validator_model = initialize_validator()
    
    if not all([bq_client, genai_model, validator_model]):
        print("❌ Failed to initialize services. Please check your configuration.")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Test each question
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"📌 Test {i}/{len(TEST_QUESTIONS)}")
        print(f"❓ Question: {question}")
        
        # Step 1: Validate prompt
        print("   🔍 Validating prompt...")
        is_valid, validation_msg = validate_prompt(validator_model, question)
        
        if not is_valid:
            print(f"   ❌ Prompt validation failed: {validation_msg}")
            print("\n" + "-"*60 + "\n")
            continue
        
        print("   ✅ Prompt is safe")
        
        # Step 2: Search knowledge base
        print("   📚 Searching knowledge base...")
        context = search_knowledge_base(bq_client, question)
        
        if not context:
            print("   ⚠️ No relevant information found in knowledge base")
            print("\n" + "-"*60 + "\n")
            continue
        
        print(f"   ✅ Found relevant context ({len(context)} characters)")
        
        # Step 3: Generate response
        print("   🤖 Generating response...")
        answer = generate_response(genai_model, question, context)
        
        print(f"\n💬 Answer: {answer}\n")
        print("-"*60 + "\n")
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_rag_system()
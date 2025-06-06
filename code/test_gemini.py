#!/usr/bin/env python3
"""
Quick test script to verify Gemini API functionality
Tests different scenarios to isolate issues
"""

import os
import asyncio
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test 1: Basic Vertex AI Test
def test_basic_vertex_ai():
    """Test basic Vertex AI connectivity"""
    print("=" * 50)
    print("TEST 1: Basic Vertex AI Connectivity")
    print("=" * 50)

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project_id = os.getenv('GCP_PROJECT', 'gen-lang-client-0750686103')
        location = 'us-central1'

        print(f"Project ID: {project_id}")
        print(f"Location: {location}")

        vertexai.init(project=project_id, location=location)
        print("✅ Vertex AI initialized successfully")

        return True
    except Exception as e:
        print(f"❌ Vertex AI initialization failed: {e}")
        return False

# Test 2: Simple Text Generation
def test_simple_generation():
    """Test simple text generation"""
    print("\n" + "=" * 50)
    print("TEST 2: Simple Text Generation")
    print("=" * 50)

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project_id = os.getenv('GCP_PROJECT', 'gen-lang-client-0750686103')
        vertexai.init(project=project_id, location='us-central1')

        # Test different models
        models_to_test = [
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash-exp'
        ]

        for model_name in models_to_test:
            try:
                print(f"\nTesting model: {model_name}")
                model = GenerativeModel(model_name)

                start_time = time.time()
                response = model.generate_content(
                    "Hello! Please respond with 'API working correctly'",
                    generation_config={
                        "max_output_tokens": 50,
                        "temperature": 0.1
                    }
                )
                end_time = time.time()

                print(f"✅ {model_name}: {response.text}")
                print(f"   Response time: {end_time - start_time:.2f}s")
                return model_name  # Return the first working model

            except Exception as e:
                print(f"❌ {model_name}: {e}")

        return None

    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return None

# Test 3: Complex Prompt (like what NLWeb sends)
def test_complex_prompt(working_model):
    """Test with a complex prompt similar to NLWeb"""
    print("\n" + "=" * 50)
    print("TEST 3: Complex Prompt Test")
    print("=" * 50)

    if not working_model:
        print("❌ No working model found, skipping complex test")
        return False

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project_id = os.getenv('GCP_PROJECT', 'gen-lang-client-0750686103')
        vertexai.init(project=project_id, location='us-central1')

        model = GenerativeModel(working_model)

        # Complex prompt similar to what NLWeb might send
        complex_prompt = """
        You are an AI assistant helping to rank podcast episodes.
        Please analyze this episode title and provide a relevance score from 1-10
        for the query "summarize latest episode":

        Episode: "Sam Altman: Entrepreneurial prodigy, Y Combinator President and OpenAI CEO"

        Consider:
        1. How recent this episode might be
        2. How interesting the content would be
        3. The relevance to the user's query

        Respond with just a number from 1-10 and a brief explanation.
        """

        print(f"Testing complex prompt with {working_model}...")
        print(f"Prompt length: {len(complex_prompt)} characters")

        start_time = time.time()
        response = model.generate_content(
            complex_prompt,
            generation_config={
                "max_output_tokens": 200,
                "temperature": 0.3
            }
        )
        end_time = time.time()

        print(f"✅ Complex prompt successful!")
        print(f"   Response time: {end_time - start_time:.2f}s")
        print(f"   Response: {response.text[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Complex prompt failed: {e}")
        return False

# Test 4: Test with different token limits
def test_token_limits(working_model):
    """Test different max_output_tokens values"""
    print("\n" + "=" * 50)
    print("TEST 4: Token Limits Test")
    print("=" * 50)

    if not working_model:
        print("❌ No working model found, skipping token test")
        return

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project_id = os.getenv('GCP_PROJECT', 'gen-lang-client-0750686103')
        vertexai.init(project=project_id, location='us-central1')

        model = GenerativeModel(working_model)

        token_limits = [100, 1000, 4096, 8192]

        for max_tokens in token_limits:
            try:
                print(f"\nTesting max_output_tokens: {max_tokens}")

                start_time = time.time()
                response = model.generate_content(
                    "Write a short explanation of artificial intelligence.",
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": 0.5
                    }
                )
                end_time = time.time()

                print(f"✅ {max_tokens} tokens: Success ({end_time - start_time:.2f}s)")
                print(f"   Response length: {len(response.text)} chars")

            except Exception as e:
                print(f"❌ {max_tokens} tokens: {e}")

    except Exception as e:
        print(f"❌ Token limits test failed: {e}")

# Test 5: Test timeout scenarios
def test_timeout_scenarios(working_model):
    """Test how the API handles timeouts"""
    print("\n" + "=" * 50)
    print("TEST 5: Timeout Scenarios")
    print("=" * 50)

    if not working_model:
        print("❌ No working model found, skipping timeout test")
        return

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project_id = os.getenv('GCP_PROJECT', 'gen-lang-client-0750686103')
        vertexai.init(project=project_id, location='us-central1')

        model = GenerativeModel(working_model)

        # Test a potentially slow request
        slow_prompt = """
        Please write a very detailed analysis of the impact of artificial intelligence
        on society, covering at least 10 different aspects in great detail.
        Make sure to provide specific examples and cite various perspectives.
        """

        print("Testing potentially slow request...")
        print("(This might take a while or timeout)")

        start_time = time.time()
        try:
            response = model.generate_content(
                slow_prompt,
                generation_config={
                    "max_output_tokens": 4096,
                    "temperature": 0.7
                }
            )
            end_time = time.time()

            print(f"✅ Slow request completed in {end_time - start_time:.2f}s")
            print(f"   Response length: {len(response.text)} chars")

        except Exception as e:
            end_time = time.time()
            print(f"❌ Slow request failed after {end_time - start_time:.2f}s: {e}")

    except Exception as e:
        print(f"❌ Timeout test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Gemini API Tests")
    print(f"GCP_PROJECT: {os.getenv('GCP_PROJECT', 'Not set')}")
    print(f"GOOGLE_API_KEY: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not set'}")

    # Run tests in sequence
    if not test_basic_vertex_ai():
        print("\n❌ Basic connectivity failed. Check your credentials.")
        return

    working_model = test_simple_generation()
    if not working_model:
        print("\n❌ No models are working. Check your project permissions.")
        return

    print(f"\n✅ Found working model: {working_model}")

    # test_complex_prompt(working_model)
    # test_token_limits(working_model)
    # test_timeout_scenarios(working_model)

    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("=" * 50)

    print("\nRecommendations:")
    print("- If all tests pass, the issue is likely in NLWeb's prompt complexity")
    print("- If complex prompts fail, try reducing max_output_tokens in NLWeb")
    print("- If timeouts occur, increase timeout values in NLWeb configuration")

if __name__ == "__main__":
    main()
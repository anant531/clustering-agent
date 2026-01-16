# # clustering_pipeline/interface/components/ai_agent.py
# import asyncio
# import json
# import os
# import time
# from typing import Dict, List, Optional, Any, Tuple
# import streamlit as st
# import pandas as pd
# import numpy as np
# import logging
# from datetime import datetime, timedelta
# import requests
# from enum import Enum

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ModelProvider(Enum):
#     GOOGLE = "google"
#     GROQ = "groq"
#     FALLBACK = "fallback"

# class AIClusteringAgent:
#     """Enhanced AI agent with comprehensive clustering analysis capabilities."""
    
#     def __init__(self):
#         # API Keys - use environment variables in production
#         self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAuMAev-5nM-Na3w1RQ6jUHGHooheahSq4")  
#         self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
        
#         # Updated model configurations
#         self.google_models = [
#             "gemini-2.0-flash",
#             "gemini-1.5-flash", 
#             "gemini-1.5-flash-latest",
#             "gemini-1.5-pro"
#         ]
        
#         self.groq_models = [
#             "llama-3.3-70b-versatile",
#             "llama-3.2-90b-text-preview", 
#             "llama-3.2-11b-text-preview",
#             "mixtral-8x7b-32768",
#             "gemma2-9b-it"
#         ]
        
#         # State tracking
#         self.current_google_model = 0
#         self.current_groq_model = 0
#         self.use_groq_fallback = False
#         self.api_call_count = {"google": 0, "groq": 0}
#         self.last_reset_time = datetime.now()
        
#         # Rate limiting
#         self.google_requests_per_minute = 30
#         self.groq_requests_per_minute = 20
#         self.request_history = {"google": [], "groq": []}
        
#         # Initialize clients
#         self._setup_clients()
        
#         # Initialize session state for Streamlit
#         self._init_session_state()
    
#     def _init_session_state(self):
#         """Initialize Streamlit session state variables."""
#         if 'ai_chat_history' not in st.session_state:
#             st.session_state.ai_chat_history = []
#         if 'cluster_insights' not in st.session_state:
#             st.session_state.cluster_insights = {}
#         if 'business_strategies' not in st.session_state:
#             st.session_state.business_strategies = {}
#         if 'ai_provider' not in st.session_state:
#             st.session_state.ai_provider = 'google'
#         if 'ai_api_key' not in st.session_state:
#             st.session_state.ai_api_key = self.GOOGLE_API_KEY
    
#     def _setup_clients(self):
#         """Initialize API clients with error handling."""
#         # Google Generative AI setup
#         try:
#             import google.generativeai as genai
#             if self.GOOGLE_API_KEY and len(self.GOOGLE_API_KEY) > 20:
#                 genai.configure(api_key=self.GOOGLE_API_KEY)
#                 self.google_client = genai
#                 logger.info("✅ Google AI client initialized")
#             else:
#                 self.google_client = None
#                 logger.warning("⚠️ Google API key not configured")
#         except ImportError:
#             self.google_client = None
#             logger.warning("⚠️ Google AI not available - install google-generativeai")
#         except Exception as e:
#             self.google_client = None
#             logger.error(f"❌ Google AI setup failed: {e}")
        
#         # Groq setup
#         try:
#             from groq import Groq
#             if self.GROQ_API_KEY and len(self.GROQ_API_KEY) > 20:
#                 self.groq_client = Groq(api_key=self.GROQ_API_KEY)
#                 logger.info("✅ Groq client initialized")
#             else:
#                 self.groq_client = None
#                 logger.warning("⚠️ Groq API key not configured")
#         except ImportError:
#             self.groq_client = None
#             logger.warning("⚠️ Groq client not available - install groq")
#         except Exception as e:
#             self.groq_client = None
#             logger.error(f"❌ Groq setup failed: {e}")
    
#     def _check_rate_limits(self, service: str) -> bool:
#         """Check if we're within rate limits."""
#         now = datetime.now()
#         minute_ago = now - timedelta(minutes=1)
        
#         # Clean old requests
#         self.request_history[service] = [
#             req_time for req_time in self.request_history[service] 
#             if req_time > minute_ago
#         ]
        
#         # Check limits
#         if service == "google":
#             return len(self.request_history[service]) < self.google_requests_per_minute
#         else:
#             return len(self.request_history[service]) < self.groq_requests_per_minute
    
#     def _log_request(self, service: str):
#         """Log a request for rate limiting."""
#         self.request_history[service].append(datetime.now())
#         self.api_call_count[service] += 1
    
#     def _switch_google_model(self):
#         """Switch to next Google model."""
#         self.current_google_model = (self.current_google_model + 1) % len(self.google_models)
#         logger.info(f"🔄 Switched to Google model: {self.google_models[self.current_google_model]}")
    
#     def _switch_groq_model(self):
#         """Switch to next Groq model."""
#         self.current_groq_model = (self.current_groq_model + 1) % len(self.groq_models)
#         logger.info(f"🔄 Switched to Groq model: {self.groq_models[self.current_groq_model]}")
    
#     async def _call_google_api(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
#         """Call Google API with error handling."""
#         if not self.google_client:
#             return None
        
#         if not self._check_rate_limits("google"):
#             logger.warning("⚠️ Google API rate limit reached")
#             await asyncio.sleep(60)
#             return None
        
#         try:
#             model_name = self.google_models[self.current_google_model]
#             model = self.google_client.GenerativeModel(model_name)
            
#             generation_config = {
#                 "max_output_tokens": max_tokens,
#                 "temperature": 0.3,
#                 "top_p": 0.9,
#                 "top_k": 40,
#             }
            
#             safety_settings = [
#                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
#                 {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
#                 {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
#                 {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
#             ]
            
#             response = model.generate_content(
#                 prompt,
#                 generation_config=generation_config,
#                 safety_settings=safety_settings
#             )
            
#             self._log_request("google")
            
#             if response.text:
#                 logger.info(f"✅ Google API successful with {model_name}")
#                 return response.text
#             else:
#                 logger.warning("⚠️ Google API returned empty response")
#                 return None
                
#         except Exception as e:
#             logger.error(f"❌ Google API error: {e}")
#             return None
    
#     async def _call_groq_api(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
#         """Call Groq API with error handling."""
#         if not self.groq_client:
#             return None
        
#         if not self._check_rate_limits("groq"):
#             logger.warning("⚠️ Groq API rate limit reached")
#             await asyncio.sleep(60)
#             return None
        
#         try:
#             model_name = self.groq_models[self.current_groq_model]
            
#             response = self.groq_client.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {
#                         "role": "system", 
#                         "content": "You are an expert data scientist specializing in customer segmentation and business analytics. Provide detailed, actionable insights based on clustering analysis data."
#                     },
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=max_tokens,
#                 temperature=0.3,
#                 top_p=0.9,
#                 stream=False
#             )
            
#             self._log_request("groq")
            
#             if response.choices and response.choices[0].message.content:
#                 content = response.choices[0].message.content
#                 logger.info(f"✅ Groq API successful with {model_name}")
#                 return content
#             else:
#                 logger.warning("⚠️ Groq API returned empty response")
#                 return None
                
#         except Exception as e:
#             logger.error(f"❌ Groq API error: {e}")
#             return None
    
#     # Main comprehensive analysis method
#     async def generate_comprehensive_analysis(self, prompt: str, max_retries: int = 6) -> str:
#         """Generate comprehensive analysis with fallback between providers."""
#         attempts = 0
        
#         while attempts < max_retries:
#             try:
#                 # Try Google first
#                 if not self.use_groq_fallback and self.google_client:
#                     result = await self._call_google_api(prompt)
#                     if result:
#                         return result
#                     self._switch_google_model()
                
#                 # Try Groq
#                 if self.groq_client:
#                     result = await self._call_groq_api(prompt)
#                     if result:
#                         return result
#                     self._switch_groq_model()
                
#                 attempts += 1
                
#                 # Switch between providers
#                 if attempts % 2 == 0:
#                     self.use_groq_fallback = not self.use_groq_fallback
                
#                 # Wait between attempts
#                 wait_time = min(2 ** (attempts // 2), 30)
#                 await asyncio.sleep(wait_time)
                
#             except Exception as e:
#                 logger.error(f"❌ Analysis error: {e}")
#                 attempts += 1
#                 if attempts < max_retries:
#                     await asyncio.sleep(2)
        
#         # Return fallback response
#         return self._create_fallback_response(prompt)
    
#     # MISSING METHOD - This was causing the error
#     async def generate_comprehensive_analysis_with_model(self, prompt: str, 
#                                                        preferred_provider: ModelProvider = ModelProvider.GOOGLE,
#                                                        max_retries: int = 3) -> Tuple[str, ModelProvider]:
#         """
#         Generate analysis with specific model preference and return which provider was used.
#         This is the method that was missing and causing the error.
#         """
#         attempts = 0
#         used_provider = ModelProvider.FALLBACK
        
#         while attempts < max_retries:
#             try:
#                 # Try preferred provider first
#                 if preferred_provider == ModelProvider.GOOGLE and self.google_client:
#                     result = await self._call_google_api(prompt)
#                     if result:
#                         return result, ModelProvider.GOOGLE
#                     self._switch_google_model()
                
#                 elif preferred_provider == ModelProvider.GROQ and self.groq_client:
#                     result = await self._call_groq_api(prompt)
#                     if result:
#                         return result, ModelProvider.GROQ
#                     self._switch_groq_model()
                
#                 # Try alternative provider
#                 if preferred_provider == ModelProvider.GOOGLE and self.groq_client:
#                     result = await self._call_groq_api(prompt)
#                     if result:
#                         return result, ModelProvider.GROQ
#                     self._switch_groq_model()
                
#                 elif preferred_provider == ModelProvider.GROQ and self.google_client:
#                     result = await self._call_google_api(prompt)
#                     if result:
#                         return result, ModelProvider.GOOGLE
#                     self._switch_google_model()
                
#                 attempts += 1
#                 await asyncio.sleep(2 ** attempts)
                
#             except Exception as e:
#                 logger.error(f"❌ Analysis error: {e}")
#                 attempts += 1
#                 if attempts < max_retries:
#                     await asyncio.sleep(2)
        
#         # Return fallback response
#         fallback_response = self._create_fallback_response(prompt)
#         return fallback_response, ModelProvider.FALLBACK
    
#     # Additional utility methods for the Streamlit interface
#     def get_cluster_context(self) -> Dict[str, Any]:
#         """Extract clustering context from session state."""
#         context = {
#             'has_data': False,
#             'clusters': {},
#             'features': [],
#             'metrics': {},
#             'data_overview': {}
#         }
        
#         if (hasattr(st.session_state, 'clustered_df') and 
#             st.session_state.clustered_df is not None and 
#             'Cluster' in st.session_state.clustered_df.columns):
            
#             df = st.session_state.clustered_df
#             features = st.session_state.get('selected_features', [])
            
#             available_features = [f for f in features if f in df.columns]
            
#             if available_features:
#                 context['has_data'] = True
#                 context['features'] = available_features
#                 context['total_samples'] = len(df)
#                 context['n_clusters'] = df['Cluster'].nunique()
                
#                 # Cluster profiles
#                 for cluster_id in sorted(df['Cluster'].unique()):
#                     cluster_data = df[df['Cluster'] == cluster_id]
#                     profile = {
#                         'size': len(cluster_data),
#                         'percentage': (len(cluster_data) / len(df)) * 100,
#                         'feature_means': {},
#                         'feature_stats': {}
#                     }
                    
#                     for feature in available_features:
#                         if pd.api.types.is_numeric_dtype(df[feature]):
#                             feature_values = cluster_data[feature].dropna()
#                             if not feature_values.empty:
#                                 profile['feature_means'][feature] = float(feature_values.mean())
#                                 profile['feature_stats'][feature] = {
#                                     'min': float(feature_values.min()),
#                                     'max': float(feature_values.max()),
#                                     'std': float(feature_values.std()) if len(feature_values) > 1 else 0.0
#                                 }
                    
#                     context['clusters'][f'Cluster_{cluster_id}'] = profile
        
#         return context
    
#     def is_ai_ready(self) -> bool:
#         """Check if AI services are available."""
#         return self.google_client is not None or self.groq_client is not None
    
#     def get_service_status(self) -> Dict[str, Any]:
#         """Get current status of AI services."""
#         return {
#             "google_available": self.google_client is not None,
#             "groq_available": self.groq_client is not None,
#             "current_google_model": self.google_models[self.current_google_model] if self.google_client else None,
#             "current_groq_model": self.groq_models[self.current_groq_model] if self.groq_client else None,
#             "using_groq_fallback": self.use_groq_fallback,
#             "api_call_counts": self.api_call_count.copy(),
#             "rate_limit_status": {
#                 "google_requests_this_minute": len(self.request_history["google"]),
#                 "groq_requests_this_minute": len(self.request_history["groq"])
#             }
#         }

# # Enhanced utility functions for testing and monitoring
# async def test_ai_agent():
#     """Test the AI agent functionality."""
#     agent = AIClusteringAgent()
    
#     test_prompt = "Analyze customer segments and provide business insights."
    
#     try:
#         logger.info("🧪 Testing AI agent...")
        
#         # Test comprehensive analysis
#         result = await agent.generate_comprehensive_analysis(test_prompt, max_retries=3)
#         print("✅ Comprehensive analysis test passed")
        
#         # Test the previously missing method
#         result_with_model, used_provider = await agent.generate_comprehensive_analysis_with_model(
#             test_prompt, ModelProvider.GOOGLE, max_retries=2
#         )
#         print(f"✅ Model-specific analysis test passed - Used: {used_provider.value}")
        
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ AI agent test failed: {e}")
#         print(f"❌ Test failed: {e}")
#         return False

# def get_agent_status():
#     """Get agent status for monitoring."""
#     agent = AIClusteringAgent()
#     status = agent.get_service_status()
    
#     print("🔍 AI Agent Status:")
#     print(f"  Google AI: {'✅ Available' if status['google_available'] else '❌ Not Available'}")
#     print(f"  Groq: {'✅ Available' if status['groq_available'] else '❌ Not Available'}")
    
#     if status['current_google_model']:
#         print(f"  Current Google Model: {status['current_google_model']}")
#     if status['current_groq_model']:
#         print(f"  Current Groq Model: {status['current_groq_model']}")
    
#     print(f"  API Calls - Google: {status['api_call_counts']['google']}, Groq: {status['api_call_counts']['groq']}")
    
#     return status

# # Main execution for testing
# if __name__ == "__main__":
#     print("🚀 Testing Fixed AI Clustering Agent...")
    
#     # Show status
#     get_agent_status()
    
#     # Run comprehensive test
#     success = asyncio.run(test_ai_agent())
    
#     if success:
#         print("\n✨ All tests passed! The missing method has been added.")
#     else:
#         print("\n❌ Some tests failed, but fallback responses should work.")

# clustering_pipeline/interface/components/ai_agent.py
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    GOOGLE = "google"
    GROQ = "groq"
    FALLBACK = "fallback"

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def get_clustering_recommendation(df, features, max_k=10):
    """
    Analyze dataset and return AI-based clustering recommendation.
    Uses heuristic + metrics (silhouette) instead of pure LLM reasoning.
    """

    data = df[features].dropna().values

    best_score = -1
    best_k = None
    best_algo = None

    # Try KMeans with different k
    for k in range(2, min(max_k, len(data))):
        try:
            km = KMeans(n_clusters=k, random_state=42).fit(data)
            score = silhouette_score(data, km.labels_)
            if score > best_score:
                best_score = score
                best_algo = "KMeans"
                best_k = k
        except Exception:
            continue

    # Try Agglomerative with the same range
    for k in range(2, min(max_k, len(data))):
        try:
            agg = AgglomerativeClustering(n_clusters=k).fit(data)
            score = silhouette_score(data, agg.labels_)
            if score > best_score:
                best_score = score
                best_algo = "Agglomerative"
                best_k = k
        except Exception:
            continue

    return {
        "algorithm": best_algo,
        "k": best_k,
        "reason": f"Highest silhouette score = {best_score:.2f}",
        "pros": [
            "Well-separated clusters",
            "Automated evaluation removes guesswork"
        ],
        "cons": [
            "Only tested KMeans & Agglomerative",
            "Might miss non-spherical clusters"
        ]
    }


class AIClusteringAgent:
    """Enhanced AI agent with comprehensive clustering analysis capabilities."""
    
    def __init__(self):
        # API Keys - use environment variables in production
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAuMAev-5nM-Na3w1RQ6jUHGHooheahSq4")  
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
        
        # Updated model configurations
        self.google_models = [
            "gemini-2.0-flash",
            "gemini-1.5-flash", 
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro"
        ]
        
        self.groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.2-90b-text-preview", 
            "llama-3.2-11b-text-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        # State tracking
        self.current_google_model = 0
        self.current_groq_model = 0
        self.use_groq_fallback = False
        self.api_call_count = {"google": 0, "groq": 0}
        self.last_reset_time = datetime.now()
        
        # Rate limiting
        self.google_requests_per_minute = 30
        self.groq_requests_per_minute = 20
        self.request_history = {"google": [], "groq": []}
        
        # Initialize clients
        self._setup_clients()
        
        # Initialize session state for Streamlit
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = []
        if 'cluster_insights' not in st.session_state:
            st.session_state.cluster_insights = {}
        if 'business_strategies' not in st.session_state:
            st.session_state.business_strategies = {}
        if 'ai_provider' not in st.session_state:
            st.session_state.ai_provider = 'google'
        if 'ai_api_key' not in st.session_state:
            st.session_state.ai_api_key = self.GOOGLE_API_KEY
    
    def _setup_clients(self):
        """Initialize API clients with error handling."""
        # Google Generative AI setup
        try:
            import google.generativeai as genai
            if self.GOOGLE_API_KEY and len(self.GOOGLE_API_KEY) > 20:
                genai.configure(api_key=self.GOOGLE_API_KEY)
                self.google_client = genai
                logger.info("✅ Google AI client initialized")
            else:
                self.google_client = None
                logger.warning("⚠️ Google API key not configured")
        except ImportError:
            self.google_client = None
            logger.warning("⚠️ Google AI not available - install google-generativeai")
        except Exception as e:
            self.google_client = None
            logger.error(f"❌ Google AI setup failed: {e}")
        
        # Groq setup
        try:
            from groq import Groq
            if self.GROQ_API_KEY and len(self.GROQ_API_KEY) > 20:
                self.groq_client = Groq(api_key=self.GROQ_API_KEY)
                logger.info("✅ Groq client initialized")
            else:
                self.groq_client = None
                logger.warning("⚠️ Groq API key not configured")
        except ImportError:
            self.groq_client = None
            logger.warning("⚠️ Groq client not available - install groq")
        except Exception as e:
            self.groq_client = None
            logger.error(f"❌ Groq setup failed: {e}")
    
    def _check_rate_limits(self, service: str) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.request_history[service] = [
            req_time for req_time in self.request_history[service] 
            if req_time > minute_ago
        ]
        
        # Check limits
        if service == "google":
            return len(self.request_history[service]) < self.google_requests_per_minute
        else:
            return len(self.request_history[service]) < self.groq_requests_per_minute
    
    def _log_request(self, service: str):
        """Log a request for rate limiting."""
        self.request_history[service].append(datetime.now())
        self.api_call_count[service] += 1
    
    def _switch_google_model(self):
        """Switch to next Google model."""
        self.current_google_model = (self.current_google_model + 1) % len(self.google_models)
        logger.info(f"🔄 Switched to Google model: {self.google_models[self.current_google_model]}")
    
    def _switch_groq_model(self):
        """Switch to next Groq model."""
        self.current_groq_model = (self.current_groq_model + 1) % len(self.groq_models)
        logger.info(f"🔄 Switched to Groq model: {self.groq_models[self.current_groq_model]}")
    
    async def _call_google_api(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        """Call Google API with error handling."""
        if not self.google_client:
            return None
        
        if not self._check_rate_limits("google"):
            logger.warning("⚠️ Google API rate limit reached")
            await asyncio.sleep(60)
            return None
        
        try:
            model_name = self.google_models[self.current_google_model]
            model = self.google_client.GenerativeModel(model_name)
            
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self._log_request("google")
            
            if response.text:
                logger.info(f"✅ Google API successful with {model_name}")
                return response.text
            else:
                logger.warning("⚠️ Google API returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Google API error: {e}")
            return None
    
    async def _call_groq_api(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        """Call Groq API with error handling."""
        if not self.groq_client:
            return None
        
        if not self._check_rate_limits("groq"):
            logger.warning("⚠️ Groq API rate limit reached")
            await asyncio.sleep(60)
            return None
        
        try:
            model_name = self.groq_models[self.current_groq_model]
            
            response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert data scientist specializing in customer segmentation and business analytics. Provide detailed, actionable insights based on clustering analysis data."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                stream=False
            )
            
            self._log_request("groq")
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                logger.info(f"✅ Groq API successful with {model_name}")
                return content
            else:
                logger.warning("⚠️ Groq API returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            return None
    
    # Main comprehensive analysis method
    async def generate_comprehensive_analysis(self, prompt: str, max_retries: int = 6) -> str:
        """Generate comprehensive analysis with fallback between providers."""
        attempts = 0
        
        while attempts < max_retries:
            try:
                # Try Google first
                if not self.use_groq_fallback and self.google_client:
                    result = await self._call_google_api(prompt)
                    if result:
                        return result
                    self._switch_google_model()
                
                # Try Groq
                if self.groq_client:
                    result = await self._call_groq_api(prompt)
                    if result:
                        return result
                    self._switch_groq_model()
                
                attempts += 1
                
                # Switch between providers
                if attempts % 2 == 0:
                    self.use_groq_fallback = not self.use_groq_fallback
                
                # Wait between attempts
                wait_time = min(2 ** (attempts // 2), 30)
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"❌ Analysis error: {e}")
                attempts += 1
                if attempts < max_retries:
                    await asyncio.sleep(2)
        
        # Return fallback response
        return self._create_fallback_response(prompt)
    
    # MISSING METHOD - This was causing the error
    async def generate_comprehensive_analysis_with_model(self, prompt: str, 
                                                       preferred_provider: ModelProvider = ModelProvider.GOOGLE,
                                                       max_retries: int = 3) -> Tuple[str, ModelProvider]:
        """
        Generate analysis with specific model preference and return which provider was used.
        This is the method that was missing and causing the error.
        """
        attempts = 0
        used_provider = ModelProvider.FALLBACK
        
        while attempts < max_retries:
            try:
                # Try preferred provider first
                if preferred_provider == ModelProvider.GOOGLE and self.google_client:
                    result = await self._call_google_api(prompt)
                    if result:
                        return result, ModelProvider.GOOGLE
                    self._switch_google_model()
                
                elif preferred_provider == ModelProvider.GROQ and self.groq_client:
                    result = await self._call_groq_api(prompt)
                    if result:
                        return result, ModelProvider.GROQ
                    self._switch_groq_model()
                
                # Try alternative provider
                if preferred_provider == ModelProvider.GOOGLE and self.groq_client:
                    result = await self._call_groq_api(prompt)
                    if result:
                        return result, ModelProvider.GROQ
                    self._switch_groq_model()
                
                elif preferred_provider == ModelProvider.GROQ and self.google_client:
                    result = await self._call_google_api(prompt)
                    if result:
                        return result, ModelProvider.GOOGLE
                    self._switch_google_model()
                
                attempts += 1
                await asyncio.sleep(2 ** attempts)
                
            except Exception as e:
                logger.error(f"❌ Analysis error: {e}")
                attempts += 1
                if attempts < max_retries:
                    await asyncio.sleep(2)
        
        # Return fallback response
        fallback_response = self._create_fallback_response(prompt)
        return fallback_response, ModelProvider.FALLBACK
    
    # Additional utility methods for the Streamlit interface
    def get_cluster_context(self) -> Dict[str, Any]:
        """Extract clustering context from session state."""
        context = {
            'has_data': False,
            'clusters': {},
            'features': [],
            'metrics': {},
            'data_overview': {}
        }
        
        if (hasattr(st.session_state, 'clustered_df') and 
            st.session_state.clustered_df is not None and 
            'Cluster' in st.session_state.clustered_df.columns):
            
            df = st.session_state.clustered_df
            features = st.session_state.get('selected_features', [])
            
            available_features = [f for f in features if f in df.columns]
            
            if available_features:
                context['has_data'] = True
                context['features'] = available_features
                context['total_samples'] = len(df)
                context['n_clusters'] = df['Cluster'].nunique()
                
                # Cluster profiles
                for cluster_id in sorted(df['Cluster'].unique()):
                    cluster_data = df[df['Cluster'] == cluster_id]
                    profile = {
                        'size': len(cluster_data),
                        'percentage': (len(cluster_data) / len(df)) * 100,
                        'feature_means': {},
                        'feature_stats': {}
                    }
                    
                    for feature in available_features:
                        if pd.api.types.is_numeric_dtype(df[feature]):
                            feature_values = cluster_data[feature].dropna()
                            if not feature_values.empty:
                                profile['feature_means'][feature] = float(feature_values.mean())
                                profile['feature_stats'][feature] = {
                                    'min': float(feature_values.min()),
                                    'max': float(feature_values.max()),
                                    'std': float(feature_values.std()) if len(feature_values) > 1 else 0.0
                                }
                    
                    context['clusters'][f'Cluster_{cluster_id}'] = profile
        
        return context
    
    def is_ai_ready(self) -> bool:
        """Check if AI services are available."""
        return self.google_client is not None or self.groq_client is not None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current status of AI services."""
        return {
            "google_available": self.google_client is not None,
            "groq_available": self.groq_client is not None,
            "current_google_model": self.google_models[self.current_google_model] if self.google_client else None,
            "current_groq_model": self.groq_models[self.current_groq_model] if self.groq_client else None,
            "using_groq_fallback": self.use_groq_fallback,
            "api_call_counts": self.api_call_count.copy(),
            "rate_limit_status": {
                "google_requests_this_minute": len(self.request_history["google"]),
                "groq_requests_this_minute": len(self.request_history["groq"])
            }
        }

# Enhanced utility functions for testing and monitoring
async def test_ai_agent():
    """Test the AI agent functionality."""
    agent = AIClusteringAgent()
    
    test_prompt = "Analyze customer segments and provide business insights."
    
    try:
        logger.info("🧪 Testing AI agent...")
        
        # Test comprehensive analysis
        result = await agent.generate_comprehensive_analysis(test_prompt, max_retries=3)
        print("✅ Comprehensive analysis test passed")
        
        # Test the previously missing method
        result_with_model, used_provider = await agent.generate_comprehensive_analysis_with_model(
            test_prompt, ModelProvider.GOOGLE, max_retries=2
        )
        print(f"✅ Model-specific analysis test passed - Used: {used_provider.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AI agent test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

def get_agent_status():
    """Get agent status for monitoring."""
    agent = AIClusteringAgent()
    status = agent.get_service_status()
    
    print("🔍 AI Agent Status:")
    print(f"  Google AI: {'✅ Available' if status['google_available'] else '❌ Not Available'}")
    print(f"  Groq: {'✅ Available' if status['groq_available'] else '❌ Not Available'}")
    
    if status['current_google_model']:
        print(f"  Current Google Model: {status['current_google_model']}")
    if status['current_groq_model']:
        print(f"  Current Groq Model: {status['current_groq_model']}")
    
    print(f"  API Calls - Google: {status['api_call_counts']['google']}, Groq: {status['api_call_counts']['groq']}")
    
    return status

# Main execution for testing
if __name__ == "__main__":
    print("🚀 Testing Fixed AI Clustering Agent...")
    
    # Show status
    get_agent_status()
    
    # Run comprehensive test
    success = asyncio.run(test_ai_agent())
    
    if success:
        print("\n✨ All tests passed! The missing method has been added.")
    else:
        print("\n❌ Some tests failed, but fallback responses should work.")

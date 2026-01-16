# clustering_pipeline/test_integration.py

import sys
import os

def test_imports():
    """Test if all imports work correctly"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        # Test if the components can be imported
        from interface.components.ai_agent import AIClusteringAgent
        print("✅ AI Agent import successful")
    except ImportError as e:
        print(f"❌ AI Agent import failed: {e}")
        return False
    
    try:
        from interface.components.results_dashboard import render_results_dashboard
        print("✅ Results Dashboard import successful")
    except ImportError as e:
        print(f"❌ Results Dashboard import failed: {e}")
        return False
    
    return True

def test_ai_agent_initialization():
    """Test AI agent initialization"""
    print("\nTesting AI Agent initialization...")
    
    try:
        from interface.components.ai_agent import AIClusteringAgent
        
        # Create agent instance
        agent = AIClusteringAgent()
        print("✅ AI Agent created successfully")
        
        # Test methods
        if hasattr(agent, 'is_ai_ready'):
            ready_status = agent.is_ai_ready()
            print(f"✅ AI readiness check: {ready_status}")
        
        if hasattr(agent, 'get_cluster_context'):
            context = agent.get_cluster_context()
            print(f"✅ Context extraction works: {context.get('has_data', False)}")
        
        print("✅ AI Agent initialization test passed")
        return True
        
    except Exception as e:
        print(f"❌ AI Agent initialization failed: {e}")
        return False

def check_file_structure():
    """Check if file structure is correct"""
    print("\nChecking file structure...")
    
    required_files = [
        "clustering_pipeline/interface/components/__init__.py",
        "clustering_pipeline/interface/components/ai_agent.py", 
        "clustering_pipeline/interface/components/results_dashboard.py",
        "clustering_pipeline/interface/components/clustering.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("Make sure you have the correct file structure and all files are saved.")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 Testing Clustering Pipeline Integration\n")
    print("=" * 50)
    
    # Test file structure
    if not check_file_structure():
        print("\n❌ File structure test failed")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed")
        return
    
    # Test AI agent
    if not test_ai_agent_initialization():
        print("\n❌ AI Agent test failed")
        return
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
if __name__ == "__main__":
    main()
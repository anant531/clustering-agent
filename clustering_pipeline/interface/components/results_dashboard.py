# clustering_pipeline/interface/components/results_dashboard.py
"""
Simplified Customer Clustering Results Dashboard

A clean, maintainable dashboard for customer segmentation analysis with:
- Simple visualization components
- Consistent error handling
- Clear separation of concerns
- Reduced complexity while maintaining functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from typing import Dict, List, Tuple, Optional, Any

# Constants
SIGNIFICANCE_THRESHOLD = 0.15  # 15% threshold for significant features
LARGE_SEGMENT_THRESHOLD = 0.25  # 25% threshold for large segments

# Try to import AI agent
try:
    from components.ai_agent import AIClusteringAgent
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.sidebar.warning("⚠️ AI analysis not available")


class SimpleDashboard:
    """Main dashboard class with simplified structure"""
    
    def __init__(self):
        self.profiles = {}
        self.features = []
        self.df = None
        self.ai_agent = AIClusteringAgent() if AI_AVAILABLE else None
    
    def render(self):
        """Main render method"""
        if not self._validate_data():
            return
        
        self._load_data()
        self._display_header()
        self._display_metrics()
        self._display_visualizations()
        self._display_profiles()
        self._display_ai_insights()
    
    def _validate_data(self) -> bool:
        """Simple data validation"""
        if not getattr(st.session_state, 'clustering_complete', False):
            st.warning("🤖 Please run clustering analysis first")
            if st.button("➡️ Go to Clustering", type="primary"):
                st.session_state['target_tab'] = 'Clustering Configuration'
                st.rerun()
            return False
        
        if not hasattr(st.session_state, 'clustered_df') or st.session_state.clustered_df is None:
            st.error("❌ No clustering results found")
            return False
        
        return True
    
    def _load_data(self):
        """Load and prepare data"""
        self.df = st.session_state.clustered_df
        self.features = st.session_state.clustering_params.get('features', [])
        self.profiles = self._create_profiles()
    
    def _create_profiles(self) -> Dict:
        """Create cluster profiles"""
        profiles = {}
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.df)) * 100,
                'features': {}
            }
            
            # Calculate feature stats
            for feature in self.features:
                if feature in self.df.columns and pd.api.types.is_numeric_dtype(self.df[feature]):
                    cluster_mean = cluster_data[feature].mean()
                    overall_mean = self.df[feature].mean()
                    
                    if not pd.isna(cluster_mean) and overall_mean != 0:
                        profile['features'][feature] = {
                            'mean': cluster_mean,
                            'overall_mean': overall_mean,
                            'difference': (cluster_mean / overall_mean) - 1
                        }
            
            profiles[cluster_id] = profile
        
        return profiles
    
    def _display_header(self):
        """Display dashboard header"""
        st.markdown("# 📊 Customer Segmentation Results")
        st.markdown("*AI-powered analysis of your customer segments*")
        st.divider()
    
    def _display_metrics(self):
        """Display key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Customers", f"{len(self.df):,}")
        
        with col2:
            st.metric("🎯 Segments", len(self.profiles))
        
        with col3:
            metrics = getattr(st.session_state, 'clustering_metrics', {})
            silhouette = metrics.get('silhouette_score', 0)
            quality = self._get_quality_label(silhouette)
            st.metric("🏆 Quality", quality, f"{silhouette:.3f}" if silhouette else "N/A")
        
        with col4:
            exec_time = metrics.get('execution_time', 0)
            st.metric("⚡ Time", f"{exec_time:.2f}s")
    
    def _get_quality_label(self, score: float) -> str:
        """Get quality label from silhouette score"""
        if score > 0.7:
            return "Excellent"
        elif score > 0.5:
            return "Good"
        elif score > 0.3:
            return "Fair"
        else:
            return "Poor"
    
    def _display_visualizations(self):
        """Display visualizations"""
        st.markdown("## 🎨 Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster size distribution
            fig_pie = self._create_size_chart()
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Feature importance
            fig_bar = self._create_importance_chart()
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Heatmap
        fig_heatmap = self._create_heatmap()
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def _create_size_chart(self) -> go.Figure:
        """Create cluster size pie chart"""
        sizes = [p['size'] for p in self.profiles.values()]
        labels = [f"Cluster {cid}" for cid in self.profiles.keys()]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=sizes,
            hole=0.4,
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title="🎯 Segment Distribution",
            height=400
        )
        
        return fig
    
    def _create_importance_chart(self) -> go.Figure:
        """Create feature importance chart"""
        feature_scores = {}
        
        for feature in self.features:
            scores = []
            for profile in self.profiles.values():
                if feature in profile['features']:
                    scores.append(abs(profile['features'][feature]['difference']))
            
            if scores:
                feature_scores[feature] = np.mean(scores)
        
        # Sort by importance
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        scores = [f[1] for f in sorted_features]
        
        fig = go.Figure([go.Bar(
            x=scores,
            y=features,
            orientation='h',
            marker_color='lightblue'
        )])
        
        fig.update_layout(
            title="📊 Feature Importance",
            xaxis_title="Discrimination Power",
            height=max(300, len(features) * 40)
        )
        
        return fig
    
    def _create_heatmap(self) -> go.Figure:
        """Create feature heatmap"""
        heatmap_data = []
        cluster_names = []
        
        for cluster_id, profile in self.profiles.items():
            cluster_names.append(f"Cluster {cluster_id}")
            row_data = []
            
            for feature in self.features:
                if feature in profile['features']:
                    diff = profile['features'][feature]['difference']
                    row_data.append(diff)
                else:
                    row_data.append(0)
            
            heatmap_data.append(row_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=self.features,
            y=cluster_names,
            colorscale='RdBu_r',
            zmid=0,
            text=[[f"{val:.1%}" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Difference: %{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title="🔥 Feature Comparison Matrix",
            height=max(300, len(cluster_names) * 60)
        )
        
        return fig
    
    def _display_profiles(self):
        """Display cluster profiles"""
        st.markdown("## 📋 Segment Profiles")
        
        # Overview table
        overview_data = []
        for cluster_id, profile in self.profiles.items():
            key_features = self._get_key_features(profile)
            overview_data.append({
                'Segment': f"Cluster {cluster_id}",
                'Size': f"{profile['size']:,}",
                'Share': f"{profile['percentage']:.1f}%",
                'Key Characteristics': ', '.join(key_features[:3]) if key_features else 'Balanced'
            })
        
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True)
        
        # Detailed profiles
        for cluster_id, profile in self.profiles.items():
            with st.expander(f"🔍 Cluster {cluster_id} Details ({profile['size']:,} customers)"):
                self._display_single_profile(cluster_id, profile)
    
    def _get_key_features(self, profile: Dict) -> List[str]:
        """Get key distinguishing features"""
        key_features = []
        
        for feature, stats in profile['features'].items():
            if abs(stats['difference']) > SIGNIFICANCE_THRESHOLD:
                direction = "↑" if stats['difference'] > 0 else "↓"
                key_features.append(f"{feature}{direction}")
        
        return sorted(key_features, key=lambda x: abs(profile['features'][x.replace('↑', '').replace('↓', '')]['difference']), reverse=True)
    
    def _display_single_profile(self, cluster_id: int, profile: Dict):
        """Display single cluster profile"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Segment Overview**")
            st.write(f"• **Size**: {profile['size']:,} customers")
            st.write(f"• **Market Share**: {profile['percentage']:.1f}%")
            
            # Size category
            if profile['percentage'] > LARGE_SEGMENT_THRESHOLD * 100:
                st.write("• **Category**: Large Segment 🔵")
            elif profile['percentage'] > 10:
                st.write("• **Category**: Medium Segment 🟡")
            else:
                st.write("• **Category**: Niche Segment 🟠")
        
        with col2:
            st.markdown("**🎯 Key Differences**")
            
            significant_features = []
            for feature, stats in profile['features'].items():
                if abs(stats['difference']) > SIGNIFICANCE_THRESHOLD:
                    direction = "Higher" if stats['difference'] > 0 else "Lower"
                    significant_features.append((feature, stats['difference'], direction))
            
            if significant_features:
                # Sort by absolute difference
                significant_features.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for feature, diff, direction in significant_features[:5]:
                    st.write(f"• **{feature}**: {direction} ({diff:+.1%})")
            else:
                st.write("• No strongly distinguishing features")
    
    def _display_ai_insights(self):
        """Display AI insights section"""
        st.markdown("## 🧠 AI Business Intelligence")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("*Get AI-powered business insights and recommendations*")
        
        with col2:
            generate_insights = st.button("🚀 Generate AI Insights", type="primary")
        
        if generate_insights or 'ai_insights' in st.session_state:
            self._handle_ai_insights(generate_insights)
    
    def _handle_ai_insights(self, regenerate: bool):
        """Handle AI insights generation - AI only"""
        if regenerate or 'ai_insights' not in st.session_state:
            if not self.ai_agent:
                st.error("❌ AI agent not available. Please check API configuration.")
                return
            
            try:
                with st.spinner("🤖 AI is analyzing your segments..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔍 Analyzing cluster patterns...")
                    progress_bar.progress(25)
                    time.sleep(0.3)
                    
                    status_text.text("🧮 Processing business metrics...")
                    progress_bar.progress(50)
                    time.sleep(0.3)
                    
                    status_text.text("🤖 Generating AI insights...")
                    progress_bar.progress(75)
                    
                    insights = self._generate_ai_insights()
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.session_state['ai_insights'] = insights
                    st.success("🎉 AI analysis completed successfully!")
                    
            except Exception as e:
                st.error(f"❌ AI analysis failed: {str(e)}")
                st.info("💡 Try checking your API keys or network connection")
                return
        
        # Display insights
        self._display_insights(st.session_state['ai_insights'])
    
    def _generate_ai_insights(self) -> Dict[str, str]:
        """Generate AI insights using the correct method"""
        # Create analysis prompt
        prompt = self._create_analysis_prompt()
        
        # Get AI response using the correct method from ai_agent.py
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Use the correct method that exists in AIClusteringAgent
            response = loop.run_until_complete(
                self.ai_agent.generate_comprehensive_analysis(prompt, max_retries=3)
            )
            
            # Parse response
            insights = self._parse_ai_response(response)
            return insights
            
        finally:
            loop.close()
    
    def _create_analysis_prompt(self) -> str:
        """Create comprehensive analysis prompt for AI"""
        # Summarize cluster data
        cluster_summary = {}
        for cluster_id, profile in self.profiles.items():
            # Get top distinguishing features
            key_features = {}
            for feature, stats in profile['features'].items():
                if abs(stats['difference']) > 0.1:  # 10% threshold
                    key_features[feature] = {
                        'mean': round(stats['mean'], 2),
                        'difference': f"{stats['difference']:+.1%}",
                        'vs_overall': round(stats['overall_mean'], 2)
                    }
            
            cluster_summary[f"Cluster_{cluster_id}"] = {
                "size": profile['size'],
                "percentage": f"{profile['percentage']:.1f}%",
                "distinguishing_features": key_features
            }
        
        prompt = f"""
You are an expert business analyst specializing in customer segmentation. Analyze these customer segments and provide actionable business insights.

CUSTOMER SEGMENTS DATA:
Total Customers: {len(self.df):,}
Number of Segments: {len(self.profiles)}
Features Analyzed: {', '.join(self.features)}

SEGMENT DETAILS:
{json.dumps(cluster_summary, indent=2)}

Please provide specific, data-driven insights for these 3 key business areas:

1. HIGH-VALUE SEGMENTS: Which segments represent the highest business value and why? Consider size, characteristics, and revenue potential.

2. MARKETING STRATEGIES: What specific marketing approaches, channels, and messaging would work best for each segment? Be tactical and actionable.

3. BUSINESS PRIORITIZATION: How should we rank and prioritize these segments for resource allocation? Provide clear reasoning and investment recommendations.

Requirements:
- Be specific and actionable, not generic
- Use the actual data provided
- Format as JSON with keys: high_value_segments, marketing_strategies, business_prioritization
- Each section should be a detailed markdown string
- Include concrete examples and recommendations

Response Format:
{{
    "high_value_segments": "## High-Value Segments Analysis\n[detailed analysis]",
    "marketing_strategies": "## Marketing Strategy Framework\n[detailed strategies]", 
    "business_prioritization": "## Business Priority Matrix\n[detailed prioritization]"
}}
"""
        return prompt
    
    def _parse_ai_response(self, response: str) -> Dict[str, str]:
        """Parse AI response with robust JSON extraction"""
        if not response or len(response.strip()) < 50:
            raise ValueError("Empty or too short AI response")
        
        try:
            # Clean the response
            cleaned = response.strip()
            
            # Extract JSON from markdown code blocks
            if '```json' in cleaned:
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                if json_match:
                    cleaned = json_match.group(1)
            
            # Extract JSON from the response
            if '{' in cleaned and '}' in cleaned:
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                json_str = cleaned[start:end]
                
                parsed = json.loads(json_str)
                
                if isinstance(parsed, dict) and len(parsed) >= 2:
                    # Validate that we have meaningful content
                    valid_insights = {}
                    for key, value in parsed.items():
                        if value and isinstance(value, str) and len(value.strip()) > 50:
                            valid_insights[key] = value
                    
                    if len(valid_insights) >= 2:
                        return valid_insights
            
            # If JSON parsing fails, try structured text extraction
            return self._extract_structured_content(response)
            
        except Exception as e:
            st.warning(f"⚠️ Response parsing issue: {str(e)}")
            return self._extract_structured_content(response)
    
    def _extract_structured_content(self, response: str) -> Dict[str, str]:
        """Extract structured content from free-form response"""
        sections = {}
        
        # Split by common section patterns
        parts = response.split('\n\n')
        content_parts = [part.strip() for part in parts if len(part.strip()) > 30]
        
        if len(content_parts) >= 3:
            sections = {
                'high_value_segments': content_parts[0],
                'marketing_strategies': content_parts[1], 
                'business_prioritization': content_parts[2]
            }
        elif len(content_parts) >= 2:
            sections = {
                'high_value_segments': content_parts[0],
                'marketing_strategies': content_parts[1],
                'business_prioritization': "Business prioritization analysis included in the comprehensive insights above."
            }
        else:
            # Single comprehensive response
            sections = {
                'high_value_segments': response[:len(response)//3] if len(response) > 300 else response,
                'marketing_strategies': response[len(response)//3:2*len(response)//3] if len(response) > 300 else "Marketing strategies integrated in analysis above.",
                'business_prioritization': response[2*len(response)//3:] if len(response) > 300 else "Prioritization recommendations included above."
            }
        
        return sections
    
    def _generate_statistical_insights(self) -> Dict[str, str]:
        """Generate statistical insights as fallback"""
        # Find highest value segment
        largest_segment = max(self.profiles.items(), key=lambda x: x[1]['size'])
        
        # Find most distinctive segment
        most_distinctive = None
        max_distinctiveness = 0
        
        for cluster_id, profile in self.profiles.items():
            distinctiveness = sum(abs(stats['difference']) for stats in profile['features'].values())
            if distinctiveness > max_distinctiveness:
                max_distinctiveness = distinctiveness
                most_distinctive = cluster_id
        
        insights = {
            'high_value': f"""**Highest Value Segments:**
• **Cluster {largest_segment[0]}**: Largest segment ({largest_segment[1]['size']:,} customers, {largest_segment[1]['percentage']:.1f}%)
• **Cluster {most_distinctive}**: Most distinctive characteristics
• Focus on these segments for immediate business impact""",

            'marketing_strategies': f"""**Marketing Recommendations:**
• **Cluster {largest_segment[0]}**: Volume-based campaigns, broad reach
• **Cluster {most_distinctive}**: Highly targeted, personalized approach
• Use segment-specific messaging and channels
• A/B test different approaches by segment""",

            'prioritization': f"""**Business Prioritization:**
1. **Tier 1**: Clusters {largest_segment[0]} (size) and {most_distinctive} (distinctiveness)
2. **Tier 2**: Medium-sized segments with growth potential
3. **Tier 3**: Niche segments for specialized strategies

**Resource Allocation**: 60% to Tier 1, 30% to Tier 2, 10% to Tier 3"""
        }
        
        return insights
    
    def _display_insights(self, insights: Dict[str, str]):
        """Display AI insights with proper formatting"""
        if not insights:
            st.error("❌ No insights available. Please regenerate AI analysis.")
            return
        
        # Map old keys to new keys for backward compatibility
        key_mapping = {
            'high_value': 'high_value_segments',
            'prioritization': 'business_prioritization'
        }
        
        # Normalize keys
        normalized_insights = {}
        for key, value in insights.items():
            mapped_key = key_mapping.get(key, key)
            normalized_insights[mapped_key] = value
        
        # Define sections to display
        insight_sections = [
            ("⭐ High-Value Customer Segments", "high_value_segments", True),
            ("📈 Targeted Marketing Strategies", "marketing_strategies", True),
            ("💰 Business Prioritization Matrix", "business_prioritization", False)
        ]
        
        # Display insights
        for title, key, is_expanded in insight_sections:
            if key in normalized_insights:
                with st.expander(title, expanded=is_expanded):
                    content = normalized_insights[key]
                    if content and len(content.strip()) > 20:
                        # Clean and format content
                        formatted_content = self._format_insight_content(content)
                        st.markdown(formatted_content)
                    else:
                        st.info("🔄 Analysis content not available for this section.")
            else:
                with st.expander(title, expanded=False):
                    st.warning(f"🔄 {title.split(' ', 1)[1]} analysis not available. Please regenerate insights.")
        
        # Add action items section
        self._display_action_recommendations()
    
    def _format_insight_content(self, content: str) -> str:
        """Format insight content for better display"""
        # Clean the content
        content = content.strip()
        
        # Remove any JSON artifacts
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        
        # Clean up escaped characters
        content = content.replace('\\n', '\n').replace('\\"', '"')
        
        # Ensure proper markdown formatting
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Ensure headers are properly formatted
                if line.startswith('##') and not line.startswith('## '):
                    line = '## ' + line[2:].strip()
                elif line.startswith('#') and not line.startswith('# '):
                    line = '# ' + line[1:].strip()
                
                formatted_lines.append(line)
        
        return '\n\n'.join(formatted_lines) if formatted_lines else content
    
    def _display_action_recommendations(self):
        """Display actionable recommendations"""
        st.markdown("---")
        st.markdown("### 🎯 Next Steps & Action Items")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚀 Immediate Actions (1-2 weeks)**")
            actions = [
                "✅ Validate segment characteristics with business teams",
                "📊 Set up segment performance tracking",
                "🎯 Identify quick-win opportunities in top segments",
                "📋 Create segment-specific customer personas"
            ]
            for action in actions:
                st.markdown(f"- {action}")
        
        with col2:
            st.markdown("**📈 Strategic Initiatives (1-3 months)**")
            initiatives = [
                "🛍️ Develop segment-specific product offerings",
                "📱 Design targeted marketing campaigns",
                "🔄 Implement segment-based pricing strategies", 
                "📈 Create segment migration strategies"
            ]
            for initiative in initiatives:
                st.markdown(f"- {initiative}")
        
        # Success metrics
        st.markdown("**📊 Key Success Metrics to Track**")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.markdown("- Segment growth rates")
            st.markdown("- Customer lifetime value by segment")
        
        with metrics_col2:
            st.markdown("- Conversion rates by segment")
            st.markdown("- Campaign ROI by segment")
        
        with metrics_col3:
            st.markdown("- Segment migration patterns")
            st.markdown("- Customer satisfaction scores")


# Main render function
def render_results_dashboard():
    """Render the simplified dashboard"""
    dashboard = SimpleDashboard()
    dashboard.render()


# Backward compatibility
render_enhanced_results_dashboard = render_results_dashboard
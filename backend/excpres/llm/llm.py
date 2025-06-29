import openai
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
from abc import ABC, abstractmethod
import pandas as pd


@dataclass
class LLMInsight:
    """Structure for LLM-generated insights"""
    category: str  # "relationship", "visualization", "business_insight", "data_quality"
    description: str
    confidence: float  # 0-1
    suggested_action: str
    technical_details: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedVisualizationRec:
    """Enhanced visualization recommendation with LLM insights"""
    chart_type: str
    columns: List[str]
    title: str
    description: str
    priority: int
    llm_reasoning: str
    business_context: str
    interactive_features: List[str]


class LLMProvider(ABC):
    """Abstract base class for different LLM providers"""

    @abstractmethod
    def generate_insights(self, context: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def suggest_visualizations(self, context: str) -> List[Dict[str, Any]]:
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT integration"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate_insights(self, context: str) -> Dict[str, Any]:
        """Generate comprehensive insights using GPT"""

        system_prompt = """
        You are an expert data analyst. Analyze the provided dataset context and generate insights in the following JSON format:
        {
            "relationships": [
                {
                    "columns": ["col1", "col2"],
                    "type": "correlation|causation|grouping|temporal",
                    "description": "detailed description",
                    "confidence": 0.8,
                    "business_impact": "high|medium|low"
                }
            ],
            "business_insights": [
                {
                    "insight": "description of insight",
                    "confidence": 0.9,
                    "action_required": "suggested action",
                    "impact": "high|medium|low"
                }
            ],
            "data_quality": [
                {
                    "issue": "description of issue",
                    "severity": "high|medium|low",
                    "recommendation": "how to fix"
                }
            ],
            "domain_context": {
                "likely_domain": "sales|finance|marketing|healthcare|etc",
                "key_metrics": ["metric1", "metric2"],
                "business_questions": ["question1", "question2"]
            }
        }
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # Parse JSON response
            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {"error": str(e)}

    def suggest_visualizations(self, context: str) -> List[Dict[str, Any]]:
        """Generate visualization suggestions with detailed reasoning"""

        system_prompt = """
        You are a data visualization expert. Based on the dataset context, suggest the most effective visualizations.
        Return a JSON array of visualization recommendations:
        [
            {
                "chart_type": "bar_chart|line_chart|scatter_plot|pie_chart|heatmap|box_plot|histogram",
                "columns": ["col1", "col2"],
                "title": "Chart Title",
                "description": "What this chart shows",
                "priority": 9,
                "reasoning": "Why this visualization is effective",
                "business_value": "What business questions it answers",
                "interactive_features": ["zoom", "filter", "drill_down"],
                "styling_suggestions": {
                    "color_scheme": "suggested colors",
                    "annotations": ["key points to highlight"]
                }
            }
        ]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.6,
                max_tokens=1500
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return []


class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate_insights(self, context: str) -> Dict[str, Any]:
        """Generate insights using Claude"""

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        prompt = f"""
        Analyze this dataset and provide insights in JSON format:
        
        {context}
        
        Focus on:
        1. Hidden relationships between variables
        2. Business implications and opportunities
        3. Data quality assessment
        4. Anomalies or interesting patterns
        
        Format as structured JSON with confidence scores.
        """

        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(
                self.base_url, headers=headers, json=payload)
            result = response.json()

            if "content" in result and len(result["content"]) > 0:
                content = result["content"][0]["text"]
                # Extract JSON from response
                start = content.find('{')
                end = content.rfind('}') + 1
                json_content = content[start:end]
                return json.loads(json_content)
            else:
                return {"error": "No content in response"}

        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return {"error": str(e)}

    def suggest_visualizations(self, context: str) -> List[Dict[str, Any]]:
        """Generate visualization suggestions using Claude"""
        # Similar implementation to OpenAI but with Anthropic API
        return []


class LLMEnhancedAnalyzer:
    """Enhanced analyzer that combines statistical analysis with LLM insights"""

    def __init__(self, base_analyzer, llm_provider: LLMProvider):
        self.base_analyzer = base_analyzer
        self.llm_provider = llm_provider
        self.llm_insights = []
        self.enhanced_recommendations = []

    def analyze_with_llm(self) -> Dict[str, Any]:
        """Perform comprehensive analysis combining statistical and LLM approaches"""

        # First, run base analysis
        if not self.base_analyzer.relationships:
            self.base_analyzer.detect_relationships()

        # Prepare context for LLM
        context = self._prepare_llm_context()

        # Get LLM insights
        llm_results = self.llm_provider.generate_insights(context)

        # Get LLM visualization suggestions
        viz_suggestions = self.llm_provider.suggest_visualizations(context)

        # Combine and enhance recommendations
        enhanced_analysis = self._combine_insights(
            llm_results, viz_suggestions)

        return enhanced_analysis

    def _prepare_llm_context(self) -> str:
        """Prepare detailed context for LLM analysis"""

        # Get basic context from base analyzer
        from .excel_analyzer import prepare_llm_context, format_for_llm_prompt

        basic_context = prepare_llm_context(self.base_analyzer)
        formatted_context = format_for_llm_prompt(basic_context)

        # Add domain-specific information
        enhanced_context = f"""
        {formatted_context}
        
        Additional Context:
        - Dataset size: {self.base_analyzer.data.shape[0]} records
        - Time span: {self._get_time_span()}
        - Key numeric ranges: {self._get_numeric_ranges()}
        - Categorical distributions: {self._get_categorical_distributions()}
        
        Please provide:
        1. Semantic understanding of the business domain
        2. Hidden relationships not captured by statistical methods
        3. Actionable business insights
        4. Advanced visualization strategies
        5. Data storytelling suggestions
        """

        return enhanced_context

    def _get_time_span(self) -> str:
        """Extract time span information if available"""
        datetime_cols = [col for col, profile in self.base_analyzer.column_profiles.items()
                         if profile.data_type.value == "datetime"]

        if datetime_cols:
            col = datetime_cols[0]
            try:
                dates = pd.to_datetime(self.base_analyzer.data[col])
                span = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                return span
            except:
                pass

        return "No temporal data detected"

    def _get_numeric_ranges(self) -> Dict[str, str]:
        """Get ranges for numeric columns"""
        ranges = {}
        for col, profile in self.base_analyzer.column_profiles.items():
            if profile.data_type.value == "numeric":
                stats = profile.statistics
                min_val = stats.get('min', 0)
                max_val = stats.get('max', 0)
                ranges[col] = f"{min_val:.2f} - {max_val:.2f}"
        return ranges

    def _get_categorical_distributions(self) -> Dict[str, Dict]:
        """Get distributions for categorical columns"""
        distributions = {}
        for col, profile in self.base_analyzer.column_profiles.items():
            if profile.data_type.value == "categorical":
                distributions[col] = profile.statistics.get(
                    'category_distribution', {})
        return distributions

    def _combine_insights(self, llm_results: Dict[str, Any], viz_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine statistical analysis with LLM insights"""

        # Process LLM insights
        processed_insights = []

        if "relationships" in llm_results:
            for rel in llm_results["relationships"]:
                insight = LLMInsight(
                    category="relationship",
                    description=rel.get("description", ""),
                    confidence=rel.get("confidence", 0.5),
                    suggested_action=f"Explore {rel.get('type', '')} relationship between {rel.get('columns', [])}",
                    technical_details=rel
                )
                processed_insights.append(insight)

        if "business_insights" in llm_results:
            for insight_data in llm_results["business_insights"]:
                insight = LLMInsight(
                    category="business_insight",
                    description=insight_data.get("insight", ""),
                    confidence=insight_data.get("confidence", 0.5),
                    suggested_action=insight_data.get("action_required", ""),
                    technical_details=insight_data
                )
                processed_insights.append(insight)

        # Process enhanced visualizations
        enhanced_viz = []
        for viz in viz_suggestions:
            enhanced_rec = EnhancedVisualizationRec(
                chart_type=viz.get("chart_type", ""),
                columns=viz.get("columns", []),
                title=viz.get("title", ""),
                description=viz.get("description", ""),
                priority=viz.get("priority", 5),
                llm_reasoning=viz.get("reasoning", ""),
                business_context=viz.get("business_value", ""),
                interactive_features=viz.get("interactive_features", [])
            )
            enhanced_viz.append(enhanced_rec)

        # Combine everything
        combined_analysis = {
            "statistical_relationships": [
                {
                    "type": rel.type.value,
                    "columns": rel.columns,
                    "strength": rel.strength,
                    "description": rel.description,
                    "p_value": rel.p_value
                }
                for rel in self.base_analyzer.relationships
            ],
            "llm_insights": [
                {
                    "category": insight.category,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "suggested_action": insight.suggested_action
                }
                for insight in processed_insights
            ],
            "enhanced_visualizations": [
                {
                    "chart_type": viz.chart_type,
                    "columns": viz.columns,
                    "title": viz.title,
                    "description": viz.description,
                    "priority": viz.priority,
                    "reasoning": viz.llm_reasoning,
                    "business_value": viz.business_context,
                    "interactive_features": viz.interactive_features
                }
                for viz in enhanced_viz
            ],
            "domain_context": llm_results.get("domain_context", {}),
            "data_quality_assessment": llm_results.get("data_quality", []),
            "recommended_next_steps": self._generate_next_steps(llm_results)
        }

        return combined_analysis

    def _generate_next_steps(self, llm_results: Dict[str, Any]) -> List[str]:
        """Generate actionable next steps based on analysis"""
        steps = [
            "Create interactive dashboard with top-priority visualizations",
            "Set up data monitoring for quality issues",
            "Implement automated reporting for key metrics"
        ]

        # Add domain-specific steps
        domain = llm_results.get("domain_context", {}).get("likely_domain", "")
        if domain == "sales":
            steps.extend([
                "Track sales funnel conversion rates",
                "Monitor seasonal trends and patterns",
                "Analyze customer segmentation opportunities"
            ])
        elif domain == "finance":
            steps.extend([
                "Implement financial KPI monitoring",
                "Set up budget variance analysis",
                "Create cash flow forecasting models"
            ])

        return steps

# Usage example with error handling and configuration


class AnalysisOrchestrator:
    """Main orchestrator for the entire analysis pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}

    def run_complete_analysis(self, excel_file_path: str) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""

        try:
            # Step 1: Load and analyze data
            from .excel_analyzer import ExcelDataAnalyzer
            analyzer = ExcelDataAnalyzer()
            analyzer.load_excel_file(excel_file_path)

            # Step 2: Set up LLM provider
            llm_provider = self._setup_llm_provider()

            # Step 3: Run enhanced analysis
            if llm_provider:
                enhanced_analyzer = LLMEnhancedAnalyzer(analyzer, llm_provider)
                results = enhanced_analyzer.analyze_with_llm()
            else:
                # Fallback to statistical analysis only
                results = self._fallback_analysis(analyzer)

            # Step 4: Prepare results for frontend
            formatted_results = self._format_for_frontend(results)

            self.results = formatted_results
            return formatted_results

        except Exception as e:
            print(f"Analysis failed: {e}")
            return {"error": str(e), "status": "failed"}

    def _setup_llm_provider(self) -> Optional[LLMProvider]:
        """Set up LLM provider based on configuration"""

        if "openai_api_key" in self.config:
            return OpenAIProvider(
                api_key=self.config["openai_api_key"],
                model=self.config.get("openai_model", "gpt-4")
            )
        elif "anthropic_api_key" in self.config:
            return AnthropicProvider(self.config["anthropic_api_key"])
        else:
            print("No LLM API key provided, using statistical analysis only")
            return None

    def _fallback_analysis(self, analyzer) -> Dict[str, Any]:
        """Fallback analysis without LLM"""
        from .excel_analyzer import VisualizationRecommender

        recommender = VisualizationRecommender(analyzer)
        recommendations = recommender.generate_recommendations()

        return {
            "statistical_relationships": [
                {
                    "type": rel.type.value,
                    "columns": rel.columns,
                    "strength": rel.strength,
                    "description": rel.description
                }
                for rel in analyzer.relationships
            ],
            "basic_visualizations": [
                {
                    "chart_type": rec.chart_type.value,
                    "columns": rec.columns,
                    "title": rec.title,
                    "description": rec.description,
                    "priority": rec.priority
                }
                for rec in recommendations
            ]
        }

    def _format_for_frontend(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results for frontend consumption"""

        return {
            "status": "success",
            "summary": {
                "total_relationships": len(results.get("statistical_relationships", [])),
                "total_insights": len(results.get("llm_insights", [])),
                "total_visualizations": len(results.get("enhanced_visualizations", [])),
                "domain": results.get("domain_context", {}).get("likely_domain", "unknown")
            },
            "relationships": results.get("statistical_relationships", []),
            "insights": results.get("llm_insights", []),
            "visualizations": results.get("enhanced_visualizations", []),
            "next_steps": results.get("recommended_next_steps", []),
            "data_quality": results.get("data_quality_assessment", [])
        }


# Example configuration and usage
if __name__ == "__main__":
    # Configuration
    config = {
        # Replace with actual key
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_model": "gpt-4o-mini",
    }

    # Run analysis
    orchestrator = AnalysisOrchestrator(config)
    results = orchestrator.run_complete_analysis("data/data.xlsx")

    # Print results
    print(json.dumps(results, indent=2))

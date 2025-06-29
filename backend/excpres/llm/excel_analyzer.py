
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"


class RelationshipType(Enum):
    CORRELATION = "correlation"
    GROUPING = "grouping"
    TIME_SERIES = "time_series"
    DISTRIBUTION = "distribution"
    HIERARCHY = "hierarchy"


class VisualizationType(Enum):
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    STACKED_BAR = "stacked_bar"


@dataclass
class ColumnProfile:
    name: str
    data_type: DataType
    unique_count: int
    null_count: int
    sample_values: List[Any]
    statistics: Dict[str, Any]


@dataclass
class Relationship:
    type: RelationshipType
    columns: List[str]
    strength: float  # 0-1 scale
    description: str
    statistical_test: Optional[str] = None
    p_value: Optional[float] = None


@dataclass
class VisualizationRecommendation:
    chart_type: VisualizationType
    columns: List[str]
    title: str
    description: str
    priority: int  # 1-10, higher is more important
    config: Optional[Dict[str, Any]] = None


class ExcelDataAnalyzer:
    """
    Core analyzer that processes Excel files and detects data relationships
    """

    def __init__(self):
        self.data = None
        self.column_profiles = {}
        self.relationships = []

    def load_excel_file(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load Excel file and perform initial data cleaning

        Args:
            file_path: Path to Excel file
            sheet_name: Specific sheet to load (None for first sheet)

        Returns:
            Cleaned DataFrame
        """
        try:
            # Read Excel file
            if sheet_name:
                self.data = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                self.data = pd.read_excel(file_path)

            # Basic cleaning
            self.data = self._clean_data(self.data)

            print(f"Loaded data with shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")

            return self.data

        except Exception as e:
            raise Exception(f"Error loading Excel file: {str(e)}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)

        return df

    def profile_columns(self) -> Dict[str, ColumnProfile]:
        """
        Analyze each column to understand its characteristics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_excel_file first.")

        profiles = {}

        for column in self.data.columns:
            col_data = self.data[column].dropna()

            # Determine data type
            data_type = self._infer_data_type(col_data)

            # Basic statistics
            stats = self._calculate_column_statistics(col_data, data_type)

            # Sample values (up to 5)
            sample_values = col_data.head(5).tolist()

            profile = ColumnProfile(
                name=column,
                data_type=data_type,
                unique_count=col_data.nunique(),
                null_count=self.data[column].isnull().sum(),
                sample_values=sample_values,
                statistics=stats
            )

            profiles[column] = profile

        self.column_profiles = profiles
        return profiles

    def _infer_data_type(self, series: pd.Series) -> DataType:
        """Infer the semantic data type of a column"""

        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERIC

        # Check if datetime
        try:
            pd.to_datetime(series.head(10))
            return DataType.DATETIME
        except:
            pass

        # Check if boolean
        unique_vals = set(str(v).lower() for v in series.unique()[:10])
        boolean_indicators = {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n'}
        if unique_vals.issubset(boolean_indicators):
            return DataType.BOOLEAN

        # Check if categorical (low cardinality)
        if series.nunique() / len(series) < 0.1 and series.nunique() < 20:
            return DataType.CATEGORICAL

        return DataType.TEXT

    def _calculate_column_statistics(self, series: pd.Series, data_type: DataType) -> Dict[str, Any]:
        """Calculate relevant statistics based on data type"""
        stats = {}

        if data_type == DataType.NUMERIC:
            stats.update({
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis())
            })
        elif data_type == DataType.CATEGORICAL:
            value_counts = series.value_counts()
            stats.update({
                'mode': value_counts.index[0],
                'mode_frequency': int(value_counts.iloc[0]),
                'category_distribution': value_counts.head(10).to_dict()
            })
        elif data_type == DataType.TEXT:
            stats.update({
                'avg_length': float(series.astype(str).str.len().mean()),
                'max_length': int(series.astype(str).str.len().max())
            })

        return stats

    def detect_relationships(self) -> List[Relationship]:
        """
        Detect various types of relationships in the data
        """
        if not self.column_profiles:
            self.profile_columns()

        relationships = []

        # 1. Numeric correlations
        relationships.extend(self._detect_correlations())

        # 2. Categorical groupings
        relationships.extend(self._detect_groupings())

        # 3. Time series patterns
        relationships.extend(self._detect_time_patterns())

        # 4. Distribution relationships
        relationships.extend(self._detect_distributions())

        self.relationships = relationships
        return relationships

    def _detect_correlations(self) -> List[Relationship]:
        """Detect correlations between numeric columns"""
        relationships = []
        numeric_cols = [col for col, profile in self.column_profiles.items()
                        if profile.data_type == DataType.NUMERIC]

        if len(numeric_cols) < 2:
            return relationships

        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()

        # Find significant correlations
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                correlation = corr_matrix.loc[col1, col2]

                if abs(correlation) > 0.3:  # Threshold for meaningful correlation
                    # Perform statistical test
                    _, p_value = stats.pearsonr(
                        self.data[col1].dropna(),
                        self.data[col2].dropna()
                    )

                    strength = abs(correlation)
                    description = f"{col1} and {col2} show {'positive' if correlation > 0 else 'negative'} correlation"

                    relationships.append(Relationship(
                        type=RelationshipType.CORRELATION,
                        columns=[col1, col2],
                        strength=strength,
                        description=description,
                        statistical_test="pearson_correlation",
                        p_value=p_value
                    ))

        return relationships

    def _detect_groupings(self) -> List[Relationship]:
        """Detect relationships between categorical and numeric columns"""
        relationships = []

        categorical_cols = [col for col, profile in self.column_profiles.items()
                            if profile.data_type == DataType.CATEGORICAL]
        numeric_cols = [col for col, profile in self.column_profiles.items()
                        if profile.data_type == DataType.NUMERIC]

        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                # Perform ANOVA test
                groups = [group[num_col].dropna()
                          for name, group in self.data.groupby(cat_col)]
                # Remove groups with single values
                groups = [g for g in groups if len(g) > 1]

                if len(groups) >= 2:
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)

                        if p_value < 0.05:  # Significant difference between groups
                            # Normalize F-statistic
                            strength = min(1.0, f_stat / 10)
                            description = f"{num_col} varies significantly across {cat_col} categories"

                            relationships.append(Relationship(
                                type=RelationshipType.GROUPING,
                                columns=[cat_col, num_col],
                                strength=strength,
                                description=description,
                                statistical_test="anova",
                                p_value=p_value
                            ))
                    except:
                        continue

        return relationships

    def _detect_time_patterns(self) -> List[Relationship]:
        """Detect time-based patterns"""
        relationships = []

        datetime_cols = [col for col, profile in self.column_profiles.items()
                         if profile.data_type == DataType.DATETIME]
        numeric_cols = [col for col, profile in self.column_profiles.items()
                        if profile.data_type == DataType.NUMERIC]

        for date_col in datetime_cols:
            for num_col in numeric_cols:
                # Convert to datetime if not already
                try:
                    date_series = pd.to_datetime(self.data[date_col])

                    # Sort by date and calculate trend
                    sorted_data = self.data.copy()
                    sorted_data[date_col] = date_series
                    sorted_data = sorted_data.sort_values(date_col)

                    # Calculate correlation with time index
                    time_index = range(len(sorted_data))
                    correlation, p_value = stats.pearsonr(
                        time_index, sorted_data[num_col].fillna(0))

                    if abs(correlation) > 0.3 and p_value < 0.05:
                        strength = abs(correlation)
                        trend = "increasing" if correlation > 0 else "decreasing"
                        description = f"{num_col} shows {trend} trend over time ({date_col})"

                        relationships.append(Relationship(
                            type=RelationshipType.TIME_SERIES,
                            columns=[date_col, num_col],
                            strength=strength,
                            description=description,
                            statistical_test="time_correlation",
                            p_value=p_value
                        ))
                except:
                    continue

        return relationships

    def _detect_distributions(self) -> List[Relationship]:
        """Detect interesting distribution patterns"""
        relationships = []

        numeric_cols = [col for col, profile in self.column_profiles.items()
                        if profile.data_type == DataType.NUMERIC]

        for col in numeric_cols:
            data = self.data[col].dropna()
            if len(data) < 10:
                continue

            # Test for normality
            _, p_value = stats.normaltest(data)

            profile = self.column_profiles[col]
            skewness = abs(profile.statistics.get('skewness', 0))

            if p_value < 0.05 or skewness > 1:  # Non-normal or highly skewed
                if skewness > 1:
                    description = f"{col} has a highly skewed distribution"
                else:
                    description = f"{col} has a non-normal distribution"

                strength = min(1.0, skewness / 2)

                relationships.append(Relationship(
                    type=RelationshipType.DISTRIBUTION,
                    columns=[col],
                    strength=strength,
                    description=description,
                    statistical_test="normality_test",
                    p_value=p_value
                ))

        return relationships


class VisualizationRecommender:
    """
    Recommends appropriate visualizations based on detected relationships
    """

    def __init__(self, analyzer: ExcelDataAnalyzer):
        self.analyzer = analyzer

    def generate_recommendations(self) -> List[VisualizationRecommendation]:
        """Generate visualization recommendations based on data analysis"""
        recommendations = []

        # Ensure analysis is complete
        if not self.analyzer.relationships:
            self.analyzer.detect_relationships()

        # Generate recommendations for each relationship
        for relationship in self.analyzer.relationships:
            recs = self._recommend_for_relationship(relationship)
            recommendations.extend(recs)

        # Add general exploratory visualizations
        recommendations.extend(self._generate_exploratory_visualizations())

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)

        return recommendations

    def _recommend_for_relationship(self, relationship: Relationship) -> List[VisualizationRecommendation]:
        """Recommend visualizations for a specific relationship"""
        recommendations = []

        if relationship.type == RelationshipType.CORRELATION:
            # Scatter plot for correlations
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.SCATTER_PLOT,
                columns=relationship.columns,
                title=f"Correlation: {' vs '.join(relationship.columns)}",
                description=f"Scatter plot showing {relationship.description}",
                priority=8,
                config={"show_trendline": True}
            ))

        elif relationship.type == RelationshipType.GROUPING:
            cat_col, num_col = relationship.columns

            # Bar chart for groupings
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.BAR_CHART,
                columns=relationship.columns,
                title=f"{num_col} by {cat_col}",
                description=f"Bar chart showing {relationship.description}",
                priority=7
            ))

            # Box plot for distribution comparison
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.BOX_PLOT,
                columns=relationship.columns,
                title=f"{num_col} Distribution by {cat_col}",
                description=f"Box plot comparing {num_col} distributions across {cat_col}",
                priority=6
            ))

        elif relationship.type == RelationshipType.TIME_SERIES:
            # Line chart for time series
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.LINE_CHART,
                columns=relationship.columns,
                title=f"{relationship.columns[1]} Over Time",
                description=f"Time series showing {relationship.description}",
                priority=9
            ))

        elif relationship.type == RelationshipType.DISTRIBUTION:
            col = relationship.columns[0]

            # Histogram for distributions
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.HISTOGRAM,
                columns=[col],
                title=f"Distribution of {col}",
                description=f"Histogram showing {relationship.description}",
                priority=5
            ))

        return recommendations

    def _generate_exploratory_visualizations(self) -> List[VisualizationRecommendation]:
        """Generate general exploratory visualizations"""
        recommendations = []

        # Pie charts for categorical columns with reasonable cardinality
        categorical_cols = [col for col, profile in self.analyzer.column_profiles.items()
                            if profile.data_type == DataType.CATEGORICAL and
                            3 <= profile.unique_count <= 8]

        for col in categorical_cols:
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.PIE_CHART,
                columns=[col],
                title=f"Distribution of {col}",
                description=f"Pie chart showing the distribution of {col} categories",
                priority=4
            ))

        # Correlation heatmap if multiple numeric columns
        numeric_cols = [col for col, profile in self.analyzer.column_profiles.items()
                        if profile.data_type == DataType.NUMERIC]

        if len(numeric_cols) >= 3:
            recommendations.append(VisualizationRecommendation(
                chart_type=VisualizationType.HEATMAP,
                columns=numeric_cols,
                title="Correlation Matrix",
                description="Heatmap showing correlations between all numeric variables",
                priority=6
            ))

        return recommendations


if __name__ == "__main__":
    # Example of how to use the system
    analyzer = ExcelDataAnalyzer()

    # This would be your actual Excel file
    # analyzer.load_excel_file("your_data.xlsx")

    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Sales': np.random.normal(1000, 200, 100) + np.arange(100) * 5,
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'Product_Category': np.random.choice(['A', 'B', 'C'], 100),
        'Marketing_Spend': np.random.normal(500, 100, 100),
        'Customer_Satisfaction': np.random.normal(4.2, 0.8, 100)
    })

    # Simulate loading data
    analyzer.data = sample_data

    # Analyze the data
    print("=== Column Profiles ===")
    profiles = analyzer.profile_columns()
    for name, profile in profiles.items():
        print(f"\n{name}:")
        print(f"  Type: {profile.data_type.value}")
        print(f"  Unique values: {profile.unique_count}")
        print(f"  Missing values: {profile.null_count}")
        if profile.data_type == DataType.NUMERIC:
            print(f"  Mean: {profile.statistics.get('mean', 'N/A'):.2f}")
            print(f"  Std: {profile.statistics.get('std', 'N/A'):.2f}")

    print("\n=== Detected Relationships ===")
    relationships = analyzer.detect_relationships()
    for rel in relationships:
        print(f"\n{rel.type.value}: {' + '.join(rel.columns)}")
        print(f"  Strength: {rel.strength:.3f}")
        print(f"  Description: {rel.description}")
        if rel.p_value:
            print(f"  P-value: {rel.p_value:.3f}")

    print("\n=== Visualization Recommendations ===")
    recommender = VisualizationRecommender(analyzer)
    recommendations = recommender.generate_recommendations()

    for i, rec in enumerate(recommendations[:10], 1):  # Top 10 recommendations
        print(f"\n{i}. {rec.chart_type.value.replace('_', ' ').title()}")
        print(f"   Columns: {', '.join(rec.columns)}")
        print(f"   Title: {rec.title}")
        print(f"   Priority: {rec.priority}/10")
        print(f"   Description: {rec.description}")

# Additional utility functions for LLM integration


def prepare_llm_context(analyzer: ExcelDataAnalyzer) -> Dict[str, Any]:
    """
    Prepare structured context for LLM analysis
    """

    if analyzer.data is None:
        raise ValueError("No data loaded. Call load_excel_file first.")

    data = analyzer.data
    context = {
        "data_overview": {
            "shape": data.shape,
            "columns": list(data.columns),
            "missing_data": data.isnull().sum().to_dict()
        },
        "column_profiles": {
            name: asdict(profile) for name, profile in analyzer.column_profiles.items()
        },
        "relationships": [
            asdict(rel) for rel in analyzer.relationships
        ],
        "sample_data": analyzer.data.head(3).to_dict('records')
    }

    return context


def format_for_llm_prompt(context: Dict[str, Any]) -> str:
    """
    Format analysis context into a structured prompt for LLM
    """
    prompt = f"""
    Data Analysis Context:
    
    Dataset Overview:
    - Shape: {context['data_overview']['shape'][0]} rows, {context['data_overview']['shape'][1]} columns
    - Columns: {', '.join(context['data_overview']['columns'])}
    
    Column Details:
    """

    for col_name, profile in context['column_profiles'].items():
        prompt += f"\n- {col_name}: {profile['data_type']} ({profile['unique_count']} unique values)"
        if profile['data_type'] == 'numeric' and 'mean' in profile['statistics']:
            prompt += f" [Mean: {profile['statistics']['mean']:.2f}]"

    prompt += f"\n\nDetected Relationships ({len(context['relationships'])}):"
    for rel in context['relationships']:
        prompt += f"\n- {rel['type']}: {' + '.join(rel['columns'])} (strength: {rel['strength']:.2f})"

    prompt += "\n\nPlease analyze this data and suggest:"
    prompt += "\n1. Additional meaningful relationships to explore"
    prompt += "\n2. Business insights that could be derived"
    prompt += "\n3. Specific visualization recommendations with reasoning"
    prompt += "\n4. Data quality concerns or preprocessing suggestions"

    return prompt

# Test Runner and Validation Script
# This script tests the entire analysis pipeline without needing the API

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from excpres.llm.llm import AnalysisOrchestrator
from excpres.llm.excel_analyzer import ExcelDataAnalyzer, VisualizationRecommender


load_dotenv()

# Add src to path if running standalone
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    sys.path.append(str(script_dir))


def create_sample_data():
    """Create realistic sample data for testing"""

    # Set random seed for reproducible results
    np.random.seed(42)

    # Generate 200 days of data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    n_records = len(dates)

    # Create realistic business data
    data = {
        'Date': dates,

        # Sales data with trend and seasonality
        'Daily_Sales': (
            np.random.normal(1000, 150, n_records) +  # Base sales
            # Weekly seasonality
            np.sin(np.arange(n_records) * 2 * np.pi / 7) * 100 +
            np.arange(n_records) * 2  # Growth trend
        ),

        # Marketing spend with correlation to sales
        'Marketing_Spend': lambda x: x['Daily_Sales'] * 0.3 + np.random.normal(0, 50, n_records),

        # Regional data
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_records,
                                   p=[0.3, 0.25, 0.25, 0.2]),

        # Product categories
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home'], n_records,
                                             p=[0.4, 0.35, 0.25]),

        # Customer satisfaction (correlated with sales)
        'Customer_Satisfaction': lambda x: (
            3.5 + (x['Daily_Sales'] - 1000) / 1000 +
            np.random.normal(0, 0.3, n_records)
        ).clip(1, 5),

        # Number of customers
        'Customer_Count': lambda x: (x['Daily_Sales'] / 50 + np.random.normal(0, 5, n_records)).astype(int),

        # Weather impact (affects sales)
        'Temperature': np.random.normal(20, 10, n_records),

        # Promotional campaigns (binary)
        'Has_Promotion': np.random.choice([0, 1], n_records, p=[0.8, 0.2]),
    }

    # Create DataFrame
    df = pd.DataFrame({k: v for k, v in data.items() if not callable(v)})

    # Apply lambda functions
    for key, value in data.items():
        if callable(value):
            df[key] = value(df)

    # Add some realistic correlations
    # Higher marketing spend during promotions
    promotion_mask = df['Has_Promotion'] == 1
    df.loc[promotion_mask, 'Marketing_Spend'] *= 1.5

    # Weather affects sales (people buy more when it's moderate temperature)
    temp_factor = 1 + 0.1 * np.exp(-((df['Temperature'] - 22) ** 2) / 50)
    df['Daily_Sales'] *= temp_factor

    # Regional differences
    region_multipliers = {'North': 1.2, 'South': 0.9, 'East': 1.1, 'West': 1.0}
    for region, multiplier in region_multipliers.items():
        mask = df['Region'] == region
        df.loc[mask, 'Daily_Sales'] *= multiplier

    return df


def save_sample_excel(df, filename='sample_data.xlsx'):
    """Save sample data to Excel file"""

    # Create multiple sheets to test sheet handling
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='Daily_Sales_Data', index=False)

        # Summary sheet
        summary = df.groupby('Region').agg({
            'Daily_Sales': ['mean', 'sum', 'count'],
            'Customer_Satisfaction': 'mean',
            'Marketing_Spend': 'sum'
        }).round(2)
        summary.to_excel(writer, sheet_name='Regional_Summary')

        # Monthly aggregation
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly['Date'].dt.to_period('M')
        monthly_agg = df_monthly.groupby('Month').agg({
            'Daily_Sales': 'sum',
            'Marketing_Spend': 'sum',
            'Customer_Count': 'sum',
            'Customer_Satisfaction': 'mean'
        }).round(2)
        monthly_agg.to_excel(writer, sheet_name='Monthly_Trends')

    print(f"Sample Excel file saved as: {filename}")
    return filename


def test_basic_analysis():
    """Test the basic statistical analysis without LLM"""

    try:

        print("=== Testing Basic Analysis ===")

        # Create sample data
        df = create_sample_data()
        excel_file = save_sample_excel(df)

        # Initialize analyzer
        analyzer = ExcelDataAnalyzer()

        # Load data
        print(f"Loading Excel file: {excel_file}")
        data = analyzer.load_excel_file(excel_file)
        print(f"Loaded data shape: {data.shape}")

        # Profile columns
        print("\n--- Column Profiling ---")
        profiles = analyzer.profile_columns()
        for name, profile in profiles.items():
            print(
                f"{name}: {profile.data_type.value} ({profile.unique_count} unique, {profile.null_count} null)")

        # Detect relationships
        print("\n--- Relationship Detection ---")
        relationships = analyzer.detect_relationships()
        print(f"Found {len(relationships)} relationships:")

        for i, rel in enumerate(relationships, 1):
            print(f"{i}. {rel.type.value}: {' + '.join(rel.columns)}")
            print(
                f"   Strength: {rel.strength:.3f}, P-value: {rel.p_value:.4f if rel.p_value else 'N/A'}")
            print(f"   Description: {rel.description}")

        # Generate visualizations
        print("\n--- Visualization Recommendations ---")
        recommender = VisualizationRecommender(analyzer)
        recommendations = recommender.generate_recommendations()

        print(
            f"Generated {len(recommendations)} visualization recommendations:")
        for i, rec in enumerate(recommendations[:10], 1):
            print(f"{i}. {rec.chart_type.value.replace('_', ' ').title()}")
            print(f"   Columns: {', '.join(rec.columns)}")
            print(f"   Priority: {rec.priority}/10")
            print(f"   Title: {rec.title}")

        # Clean up
        os.remove(excel_file)

        return True

    except Exception as e:
        print(f"Basic analysis test failed: {e}")
        return False


def test_llm_integration():
    """Test LLM integration (requires API keys)"""

    try:

        print("\n=== Testing LLM Integration ===")

        # Check for API keys
        config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        }

        if not any(config.values()):
            print("No LLM API keys found in environment variables.")
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test LLM integration.")
            return False

        # Create sample data
        df = create_sample_data()
        excel_file = save_sample_excel(df)

        # Run enhanced analysis
        orchestrator = AnalysisOrchestrator(config)
        results = orchestrator.run_complete_analysis(excel_file)

        if results.get("status") == "success":
            print("LLM integration successful!")
            print(f"Summary: {results['summary']}")

            if "insights" in results:
                print(f"\nGenerated {len(results['insights'])} LLM insights:")
                for insight in results['insights'][:3]:
                    print(
                        f"- {insight['description']} (confidence: {insight['confidence']})")
        else:
            print(
                f"LLM integration failed: {results.get('error', 'Unknown error')}")
            return False

        # Clean up
        os.remove(excel_file)

        return True

    except Exception as e:
        print(f"LLM integration test failed: {e}")
        return False


def test_api_simulation():
    """Simulate the API workflow"""

    print("\n=== Testing API Simulation ===")

    try:
        # Create sample data
        df = create_sample_data()
        excel_file = save_sample_excel(df, 'api_test_sample.xlsx')

        config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "max_file_size_mb": 50,
            "correlation_threshold": 0.3,
        }

        orchestrator = AnalysisOrchestrator(config)

        # Time the analysis
        start_time = datetime.now()
        results = orchestrator.run_complete_analysis(excel_file)
        end_time = datetime.now()

        analysis_time = (end_time - start_time).total_seconds()

        print(f"Analysis completed in {analysis_time:.2f} seconds")
        print(f"Status: {results.get('status', 'unknown')}")

        if results.get("status") == "success":
            summary = results.get("summary", {})
            print(f"Results summary:")
            print(
                f"  - Relationships found: {summary.get('total_relationships', 0)}")
            print(
                f"  - Insights generated: {summary.get('total_insights', 0)}")
            print(
                f"  - Visualizations recommended: {summary.get('total_visualizations', 0)}")
            print(f"  - Detected domain: {summary.get('domain', 'unknown')}")

        # Save results for inspection
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("Detailed results saved to: test_results.json")

        # Clean up
        os.remove(excel_file)

        return True

    except Exception as e:
        print(f"API simulation test failed: {e}")
        return False


def run_performance_test():
    """Test performance with different data sizes"""

    print("\n=== Performance Testing ===")

    sizes = [100, 500, 1000, 2000]
    results = {}

    for size in sizes:
        print(f"\nTesting with {size} records...")

        try:
            # Create data of specific size
            dates = pd.date_range('2023-01-01', periods=size, freq='D')
            df = pd.DataFrame({
                'Date': dates,
                'Value1': np.random.normal(100, 20, size),
                'Value2': np.random.normal(50, 10, size),
                'Category': np.random.choice(['A', 'B', 'C'], size),
                'Region': np.random.choice(['North', 'South'], size)
            })

            excel_file = f'perf_test_{size}.xlsx'
            df.to_excel(excel_file, index=False)

            analyzer = ExcelDataAnalyzer()
            start_time = datetime.now()

            analyzer.load_excel_file(excel_file)
            analyzer.profile_columns()
            analyzer.detect_relationships()

            end_time = datetime.now()
            analysis_time = (end_time - start_time).total_seconds()

            results[size] = analysis_time
            print(f"  Analysis time: {analysis_time:.2f} seconds")

            # Clean up
            os.remove(excel_file)

        except Exception as e:
            print(f"  Failed: {e}")
            results[size] = "failed"

    print("\n--- Performance Summary ---")
    for size, time in results.items():
        if time != "failed":
            print(f"{size:,} records: {time:.2f}s")
        else:
            print(f"{size:,} records: FAILED")

    return results


def validate_installation():
    """Validate that all required packages are installed"""

    print("=== Validating Installation ===")

    required_packages = [
        'pandas', 'numpy', 'scipy', 'openpyxl', 'xlrd'
    ]

    optional_packages = [
        'openai', 'anthropic', 'requests', 'fastapi'
    ]

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (REQUIRED)")
            missing_required.append(package)

    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"? {package} (optional)")
            missing_optional.append(package)

    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print(f"\nMissing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))

    print("\n✓ All required packages are installed!")
    return True


def main():
    """Run all tests"""

    print("Excel Data Relationship Analyzer - Test Suite")
    print("=" * 50)

    # Validate installation
    if not validate_installation():
        print("Installation validation failed. Please install required packages.")
        return

    # Run tests
    test_results = {}

    test_results['basic_analysis'] = test_basic_analysis()
    test_results['llm_integration'] = test_llm_integration()
    test_results['api_simulation'] = test_api_simulation()

    # Performance test
    print("\nRunning performance tests...")
    perf_results = run_performance_test()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nPerformance test completed for {len(perf_results)} data sizes")

    if all(test_results.values()):
        print("\n🎉 All tests passed! Your system is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

    print("\nNext steps:")
    print("1. Set up environment variables (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    print("2. Run the FastAPI server: python main.py")
    print("3. Test with your own Excel files")


if __name__ == "__main__":
    main()

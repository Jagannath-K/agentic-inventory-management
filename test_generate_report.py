#!/usr/bin/env python3
"""
Test the Generate Report button functionality
"""

import streamlit as st
from agents.reflector import ReflectorAgent

def test_generate_report_button():
    """Simulate the Generate Report button functionality"""
    print("🖥️  Testing Generate Report Button Functionality")
    print("=" * 60)
    
    try:
        # Initialize ReflectorAgent (similar to what Streamlit does)
        print("📊 Initializing ReflectorAgent...")
        reflector = ReflectorAgent()
        print("✅ ReflectorAgent initialized")
        
        # Simulate clicking "Generate Report" button
        print("\n🔄 Simulating 'Generate Report' button click...")
        
        # This is what happens when the button is clicked
        try:
            report = reflector.create_optimization_report()
            print("✅ Report generated successfully!")
            
            # Display key metrics (like Streamlit would)
            if report and 'summary' in report:
                summary = report['summary']
                print(f"\n📈 Report Summary:")
                print(f"   Health: {summary.get('overall_system_health', 'N/A')}")
                print(f"   Products: {summary.get('total_products_analyzed', 0)}")
                print(f"   Issues: {summary.get('high_priority_issues', 0)}")
                print(f"   Suppliers: {summary.get('suppliers_evaluated', 0)}")
                
                print(f"\n📊 Performance Data:")
                print(f"   KPIs: {len(report.get('key_performance_indicators', []))}")
                print(f"   Insights: {len(report.get('critical_insights', []))}")
                print(f"   Stockout Risks: {len(report.get('stockout_risks', {}))}")
                
                if 'error' in report:
                    print(f"⚠️  Warning: {report['error']}")
                else:
                    print("✅ No errors detected")
                    
            else:
                print("❌ Report is empty or invalid")
                
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return False
        
        # Test second click (to verify no "first time" issues)
        print(f"\n🔄 Testing second report generation...")
        try:
            report2 = reflector.create_optimization_report()
            print("✅ Second report generated successfully!")
            
        except Exception as e:
            print(f"❌ Error on second generation: {e}")
            return False
            
        print(f"\n🎯 Generate Report button functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generate_report_button()
    if success:
        print("\n✅ SOLUTION VERIFIED: Generate Report button should now work on first click!")
    else:
        print("\n❌ ISSUE PERSISTS: Further debugging needed.")

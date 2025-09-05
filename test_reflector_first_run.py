#!/usr/bin/env python3
"""
Test ReflectorAgent error handling for first-time report generation
"""

from agents.reflector import ReflectorAgent
import json

def test_reflector_first_run():
    print("🔍 Testing ReflectorAgent First-Time Report Generation")
    print("=" * 60)
    
    try:
        # Initialize ReflectorAgent
        reflector = ReflectorAgent()
        
        print("✅ ReflectorAgent initialized successfully")
        
        # Test report generation
        print("\n📊 Generating optimization report...")
        report = reflector.create_optimization_report()
        
        if report:
            print("✅ Report generated successfully!")
            print(f"📋 Report Summary:")
            print(f"   - Report Date: {report.get('report_date', 'N/A')}")
            print(f"   - Products Analyzed: {report.get('summary', {}).get('total_products_analyzed', 0)}")
            print(f"   - High Priority Issues: {report.get('summary', {}).get('high_priority_issues', 0)}")
            print(f"   - System Health: {report.get('summary', {}).get('overall_system_health', 'UNKNOWN')}")
            print(f"   - KPIs Count: {len(report.get('key_performance_indicators', []))}")
            print(f"   - Insights Count: {len(report.get('critical_insights', []))}")
            
            if 'error' in report:
                print(f"⚠️  Error noted: {report['error']}")
            else:
                print("✅ No errors in report generation")
                
        else:
            print("❌ Failed to generate report")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 Test completed!")

if __name__ == "__main__":
    test_reflector_first_run()

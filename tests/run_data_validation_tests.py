#!/usr/bin/env python3
"""
Data Validation Test Runner
==========================

This script runs comprehensive data validation tests for all view components
and generates a detailed report of test results.
"""

import unittest
import sys
import os
from io import StringIO
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_data_validation import (
    TestDataRequiredDecorator,
    TestRevenueViewDataValidation,
    TestForecastViewDataValidation,
    TestKPIViewDataValidation,
    TestOperationsViewDataValidation,
    TestDataIntegrityValidation,
    TestExcelDataImportValidation
)


class DataValidationTestRunner:
    """Custom test runner for data validation tests."""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
    
    def run_all_tests(self):
        """Run all data validation test suites."""
        print("\n" + "="*60)
        print("HOTEL DASHBOARD DATA VALIDATION TEST SUITE")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n")
        
        # Define test suites
        test_suites = [
            ('Data Required Decorator Tests', TestDataRequiredDecorator),
            ('Revenue View Data Validation', TestRevenueViewDataValidation),
            ('Forecast View Data Validation', TestForecastViewDataValidation),
            ('KPI View Data Validation', TestKPIViewDataValidation),
            ('Operations View Data Validation', TestOperationsViewDataValidation),
            ('Data Integrity Validation', TestDataIntegrityValidation),
            ('Excel Import Validation', TestExcelDataImportValidation)
        ]
        
        # Run each test suite
        for suite_name, test_class in test_suites:
            print(f"Running {suite_name}...")
            self._run_test_suite(suite_name, test_class)
            print()
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.failed_tests == 0 and self.error_tests == 0
    
    def _run_test_suite(self, suite_name, test_class):
        """Run a specific test suite."""
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Capture test output
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        
        # Process results
        suite_total = result.testsRun
        suite_failures = len(result.failures)
        suite_errors = len(result.errors)
        suite_passed = suite_total - suite_failures - suite_errors
        
        # Update totals
        self.total_tests += suite_total
        self.passed_tests += suite_passed
        self.failed_tests += suite_failures
        self.error_tests += suite_errors
        
        # Store results
        self.test_results.append({
            'suite_name': suite_name,
            'total': suite_total,
            'passed': suite_passed,
            'failed': suite_failures,
            'errors': suite_errors,
            'failures': result.failures,
            'error_details': result.errors
        })
        
        # Print suite results
        status = "✅ PASSED" if suite_failures == 0 and suite_errors == 0 else "❌ FAILED"
        print(f"  {status} - {suite_passed}/{suite_total} tests passed")
        
        if suite_failures > 0:
            print(f"  ⚠️  {suite_failures} test(s) failed")
        if suite_errors > 0:
            print(f"  🚨 {suite_errors} test(s) had errors")
    
    def _generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("TEST SUMMARY REPORT")
        print("="*60)
        
        # Overall statistics
        print(f"Total Tests Run: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ({self.passed_tests/self.total_tests*100:.1f}%)")
        print(f"Failed: {self.failed_tests} ({self.failed_tests/self.total_tests*100:.1f}%)")
        print(f"Errors: {self.error_tests} ({self.error_tests/self.total_tests*100:.1f}%)")
        
        overall_status = "✅ ALL TESTS PASSED" if self.failed_tests == 0 and self.error_tests == 0 else "❌ SOME TESTS FAILED"
        print(f"\nOverall Status: {overall_status}")
        
        # Detailed results by suite
        print("\n" + "-"*60)
        print("DETAILED RESULTS BY TEST SUITE")
        print("-"*60)
        
        for result in self.test_results:
            print(f"\n{result['suite_name']}:")
            print(f"  Total: {result['total']}, Passed: {result['passed']}, Failed: {result['failed']}, Errors: {result['errors']}")
            
            # Show failure details
            if result['failures']:
                print("  Failures:")
                for test, traceback in result['failures']:
                    print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
            
            # Show error details
            if result['error_details']:
                print("  Errors:")
                for test, traceback in result['error_details']:
                    print(f"    - {test}: {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'See details above'}")
        
        # Recommendations
        print("\n" + "-"*60)
        print("RECOMMENDATIONS")
        print("-"*60)
        
        if self.failed_tests > 0 or self.error_tests > 0:
            print("\n🔧 Action Items:")
            print("1. Review failed tests and implement proper data validation")
            print("2. Add error handling for missing or invalid data scenarios")
            print("3. Implement data cleaning and preprocessing steps")
            print("4. Add user-friendly error messages for data issues")
            print("5. Consider implementing data validation at the import stage")
        else:
            print("\n🎉 Excellent! All data validation tests are passing.")
            print("Your application handles data validation scenarios properly.")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    def save_report_to_file(self, filename="data_validation_report.txt"):
        """Save the test report to a file."""
        report_path = os.path.join(os.path.dirname(__file__), filename)
        
        with open(report_path, 'w') as f:
            f.write(f"Hotel Dashboard Data Validation Test Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"Total Tests: {self.total_tests}\n")
            f.write(f"Passed: {self.passed_tests}\n")
            f.write(f"Failed: {self.failed_tests}\n")
            f.write(f"Errors: {self.error_tests}\n\n")
            
            for result in self.test_results:
                f.write(f"{result['suite_name']}:\n")
                f.write(f"  Passed: {result['passed']}/{result['total']}\n")
                if result['failures'] or result['error_details']:
                    f.write(f"  Issues found - review implementation\n")
                f.write("\n")
        
        print(f"\n📄 Detailed report saved to: {report_path}")


def main():
    """Main function to run data validation tests."""
    runner = DataValidationTestRunner()
    
    try:
        success = runner.run_all_tests()
        runner.save_report_to_file()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n🚨 Error running tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
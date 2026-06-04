'''
Using unittest for testing
python -m unittest test_step2_logo_matcher.py
python -m unittest discover -p "test_*.py"
'''
import os
import sys
import unittest

if __package__:
    from .test_step1 import TestStep1LayoutDetectorSimple
    from .test_step2 import TestStep2LogoMatcher
    from .test_step3 import TestStep3CrpClassifier
    from .test_step4 import TestStep4DynamicAnalysis
else:
    # Allow running the module directly without -m
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from test_step1 import TestStep1LayoutDetectorSimple
    from test_step2 import TestStep2LogoMatcher
    from test_step3 import TestStep3CrpClassifier
    from test_step4 import TestStep4DynamicAnalysis

def run_all_tests():
    """Run tests for all steps"""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestStep1LayoutDetectorSimple,
        TestStep2LogoMatcher,
        TestStep3CrpClassifier,
        TestStep4DynamicAnalysis,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Ensure the tested modules can be imported
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run all tests
    success = run_all_tests()
    
    # Exit based on test results
    sys.exit(0 if success else 1)
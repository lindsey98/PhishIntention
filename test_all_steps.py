'''
使用unittest测试
python -m unittest test_step2_logo_matcher.py
python -m unittest discover -p "test_*.py"
'''
import unittest
import sys

# 导入所有测试类
from test_step1 import TestStep1LayoutDetectorSimple
from test_step2 import TestStep2LogoMatcher
from test_step3 import TestStep3CrpClassifier
from test_step4 import TestStep4DynamicAnalysis


def run_all_tests():
    """运行所有步骤的测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestStep1LayoutDetectorSimple,
        TestStep2LogoMatcher,
        TestStep3CrpClassifier,
        TestStep4DynamicAnalysis,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出总结
    print("\n" + "="*60)
    print("测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # 确保能导入被测试的模块
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 运行所有测试
    success = run_all_tests()
    
    # 根据测试结果退出
    sys.exit(0 if success else 1)
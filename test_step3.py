import unittest
from unittest.mock import Mock, patch
import numpy as np


class TestStep3CrpClassifier(unittest.TestCase):
    """测试_step3_crp_classifier函数的单元测试"""
    
    def setUp(self):
        """测试前准备"""
        # 导入被测试的类
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper
        
        # 创建模拟的wrapper实例
        self.wrapper = Mock(spec=PhishIntentionWrapper)
        self.wrapper.CRP_CLASSIFIER = Mock()
    
    def test_output_format_html_heuristic_noncrp(self):
        """测试HTML启发式返回nonCRP时的情况"""
        with patch('phishintention_wrapper.html_heuristic') as mock_html, \
             patch('phishintention_wrapper.credential_classifier_mixed') as mock_classifier:
            
            # 模拟HTML启发式返回1（nonCRP）
            mock_html.return_value = 1
            # 模拟分类器返回0（CRP）
            mock_classifier.return_value = 0
            
            # 创建真实wrapper实例
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
            
            # 测试数据
            screenshot_path = "test.png"
            html_path = "test.html"
            pred_boxes = np.array([[10, 20, 100, 150], [200, 300, 400, 500]])
            pred_classes = np.array([1, 2])
            
            # 执行测试
            cre_pred, crp_class_time = wrapper._step3_crp_classifier(
                screenshot_path, html_path, pred_boxes, pred_classes
            )
            
            # 验证返回类型
            self.assertIsInstance(cre_pred, int)
            self.assertIsInstance(crp_class_time, float)
            
            # 验证函数调用
            mock_html.assert_called_once_with(html_path)
            mock_classifier.assert_called_once_with(
                img=screenshot_path,
                coords=pred_boxes,
                types=pred_classes,
                model=wrapper.CRP_CLASSIFIER
            )
            
            # 验证返回结果
            self.assertEqual(cre_pred, 0)  # 分类器的结果
    
    def test_output_format_html_heuristic_crp(self):
        """测试HTML启发式返回CRP时的情况"""
        with patch('phishintention_wrapper.html_heuristic') as mock_html, \
             patch('phishintention_wrapper.credential_classifier_mixed') as mock_classifier:
            
            # 模拟HTML启发式返回0（CRP）
            mock_html.return_value = 0
            
            # 创建wrapper实例
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
            
            # 测试数据
            screenshot_path = "test.png"
            html_path = "test.html"
            pred_boxes = np.array([[10, 20, 100, 150]])
            pred_classes = np.array([1])
            
            # 执行测试
            cre_pred, crp_class_time = wrapper._step3_crp_classifier(
                screenshot_path, html_path, pred_boxes, pred_classes
            )
            
            # 验证返回类型
            self.assertIsInstance(cre_pred, int)
            self.assertIsInstance(crp_class_time, float)
            
            # 验证函数调用
            mock_html.assert_called_once_with(html_path)
            # HTML启发式返回CRP时，不应调用分类器
            mock_classifier.assert_not_called()
            
            # 验证返回结果
            self.assertEqual(cre_pred, 0)  # HTML启发式的结果
    
    def test_output_format_classifier_results(self):
        """测试不同分类器结果"""
        test_cases = [
            (1, 1, "HTML nonCRP, 分类器 nonCRP"),
            (1, 0, "HTML nonCRP, 分类器 CRP"),
            (0, None, "HTML CRP, 不调用分类器"),
        ]
        
        for html_result, classifier_result, description in test_cases:
            with self.subTest(description=description):
                with patch('phishintention_wrapper.html_heuristic') as mock_html, \
                     patch('phishintention_wrapper.credential_classifier_mixed') as mock_classifier:
                    
                    mock_html.return_value = html_result
                    if classifier_result is not None:
                        mock_classifier.return_value = classifier_result
                    
                    wrapper = self.wrapper_class()
                    wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
                    
                    cre_pred, crp_class_time = wrapper._step3_crp_classifier(
                        "test.png", "test.html", 
                        np.array([[0, 0, 100, 100]]), 
                        np.array([1])
                    )
                    
                    self.assertIsInstance(cre_pred, int)
                    self.assertIsInstance(crp_class_time, float)
                    
                    if html_result == 0:
                        expected = 0  # HTML返回CRP
                    else:
                        expected = classifier_result  # 分类器结果
                    
                    self.assertEqual(cre_pred, expected)
    
    def test_time_measurement(self):
        """测试时间测量功能"""
        import time
        
        with patch('phishintention_wrapper.html_heuristic') as mock_html, \
             patch('phishintention_wrapper.credential_classifier_mixed') as mock_classifier:
            
            # 模拟需要时间的处理
            def delayed_html(*args, **kwargs):
                time.sleep(0.03)
                return 1
            
            def delayed_classifier(*args, **kwargs):
                time.sleep(0.02)
                return 0
            
            mock_html.side_effect = delayed_html
            mock_classifier.side_effect = delayed_classifier
            
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
            
            _, crp_class_time = wrapper._step3_crp_classifier(
                "test.png", "test.html", 
                np.array([[0, 0, 100, 100]]), 
                np.array([1])
            )
            
            # 验证总时间（HTML + 分类器）
            self.assertGreaterEqual(crp_class_time, 0.05)


if __name__ == '__main__':
    unittest.main()
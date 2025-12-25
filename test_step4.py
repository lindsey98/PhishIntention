import unittest
from unittest.mock import Mock, patch, MagicMock


class TestStep4DynamicAnalysis(unittest.TestCase):
    """测试_step4_dynamic_analysis函数的单元测试"""
    
    def setUp(self):
        """测试前准备"""
        # 导入被测试的类
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper
        
        # 创建模拟的wrapper实例
        self.wrapper = Mock(spec=PhishIntentionWrapper)
        self.wrapper.CRP_CLASSIFIER = Mock()
        self.wrapper.AWL_MODEL = Mock()
        self.wrapper.CRP_LOCATOR_MODEL = Mock()
    
    def test_output_format_successful_analysis(self):
        """测试动态分析成功找到CRP的情况"""
        with patch('phishintention_wrapper.driver_loader') as mock_loader, \
             patch('phishintention_wrapper.crp_locator') as mock_locator:
            
            # 模拟driver
            mock_driver = MagicMock()
            mock_loader.return_value = mock_driver
            
            # 模拟crp_locator返回成功
            mock_locator.return_value = (
                "https://updated-url.com",  # url
                "/new/screenshot.png",      # screenshot_path
                True,                       # successful
                2.5                         # process_time
            )
            
            # 创建真实wrapper实例
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
            wrapper.CRP_LOCATOR_MODEL = self.wrapper.CRP_LOCATOR_MODEL
            
            # 测试数据
            url = "https://original-url.com"
            screenshot_path = "/original/screenshot.png"
            pred_boxes = [[10, 20, 100, 150]]
            pred_classes = [1]
            
            # 执行测试
            result = wrapper._step4_dynamic_analysis(
                url, screenshot_path, pred_boxes, pred_classes
            )
            
            # 验证返回结构
            self.assertEqual(len(result), 4)
            new_url, new_screenshot_path, successful, process_time = result
            
            # 验证返回类型
            self.assertIsInstance(new_url, str)
            self.assertIsInstance(new_screenshot_path, str)
            self.assertIsInstance(successful, bool)
            self.assertIsInstance(process_time, float)
            
            # 验证driver相关调用
            mock_loader.assert_called_once()
            mock_locator.assert_called_once_with(
                url=url,
                screenshot_path=screenshot_path,
                cls_model=wrapper.CRP_CLASSIFIER,
                ele_model=wrapper.AWL_MODEL,
                login_model=wrapper.CRP_LOCATOR_MODEL,
                driver=mock_driver
            )
            mock_driver.quit.assert_called_once()
            
            # 验证返回值
            self.assertEqual(new_url, "https://updated-url.com")
            self.assertEqual(new_screenshot_path, "/new/screenshot.png")
            self.assertTrue(successful)
            self.assertEqual(process_time, 2.5)
    
    def test_output_format_unsuccessful_analysis(self):
        """测试动态分析未找到CRP的情况"""
        with patch('phishintention_wrapper.driver_loader') as mock_loader, \
             patch('phishintention_wrapper.crp_locator') as mock_locator:
            
            # 模拟driver
            mock_driver = MagicMock()
            mock_loader.return_value = mock_driver
            
            # 模拟crp_locator返回失败
            mock_locator.return_value = (
                "https://original-url.com",  # url不变
                "/original/screenshot.png",  # screenshot_path不变
                False,                       # successful
                1.8                          # process_time
            )
            
            # 创建wrapper实例
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
            wrapper.CRP_LOCATOR_MODEL = self.wrapper.CRP_LOCATOR_MODEL
            
            # 执行测试
            new_url, new_screenshot_path, successful, process_time = \
                wrapper._step4_dynamic_analysis(
                    "https://original-url.com",
                    "/original/screenshot.png",
                    [[0, 0, 100, 100]],
                    [1]
                )
            
            # 验证返回类型
            self.assertIsInstance(new_url, str)
            self.assertIsInstance(new_screenshot_path, str)
            self.assertIsInstance(successful, bool)
            self.assertIsInstance(process_time, float)
            
            # 验证返回值
            self.assertEqual(new_url, "https://original-url.com")
            self.assertEqual(new_screenshot_path, "/original/screenshot.png")
            self.assertFalse(successful)
            self.assertEqual(process_time, 1.8)
    
    def test_driver_lifecycle(self):
        """测试driver的完整生命周期"""
        with patch('phishintention_wrapper.driver_loader') as mock_loader, \
             patch('phishintention_wrapper.crp_locator') as mock_locator:
            
            # 跟踪driver使用
            mock_driver = MagicMock()
            mock_driver.quit_called = False
            
            def mark_quit():
                mock_driver.quit_called = True
            
            mock_driver.quit.side_effect = mark_quit
            mock_loader.return_value = mock_driver
            mock_locator.return_value = ("url", "path", True, 1.0)
            
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
            wrapper.CRP_LOCATOR_MODEL = self.wrapper.CRP_LOCATOR_MODEL
            
            # 执行测试
            wrapper._step4_dynamic_analysis("url", "path", [], [])
            
            # 验证driver被正确初始化和清理
            mock_loader.assert_called_once()
            self.assertTrue(mock_driver.quit_called, "driver.quit()应该被调用")
    
    def test_time_measurement(self):
        """测试时间测量包含在返回中"""
        with patch('phishintention_wrapper.driver_loader') as mock_loader, \
             patch('phishintention_wrapper.crp_locator') as mock_locator:
            
            mock_driver = MagicMock()
            mock_loader.return_value = mock_driver
            
            # 设置不同的处理时间
            test_times = [0.5, 1.0, 2.0, 5.0]
            
            for process_time in test_times:
                with self.subTest(process_time=process_time):
                    mock_locator.return_value = ("url", "path", True, process_time)
                    
                    wrapper = self.wrapper_class()
                    wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
                    wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
                    wrapper.CRP_LOCATOR_MODEL = self.wrapper.CRP_LOCATOR_MODEL
                    
                    _, _, _, returned_time = wrapper._step4_dynamic_analysis(
                        "url", "path", [], []
                    )
                    
                    # 验证返回的时间与模拟的时间一致
                    self.assertEqual(returned_time, process_time)


if __name__ == '__main__':
    unittest.main()
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch


class TestStep1LayoutDetectorSimple(unittest.TestCase):
    """精简版单元测试，验证核心功能"""
    
    def setUp(self):
        """创建测试环境"""
        # 导入被测试的类
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper
        
        # 创建模拟的wrapper实例
        self.wrapper = Mock(spec=PhishIntentionWrapper)
        self.wrapper.AWL_MODEL = Mock()
    
    def test_output_format_with_detections(self):
        """测试有检测结果时的输出格式"""
        with patch('phishintention.pred_rcnn') as mock_pred, \
             patch('phishintention.vis') as mock_vis:
            
            # 设置模拟返回值
            mock_pred.return_value = (
                torch.tensor([[10, 20, 100, 150], [200, 300, 400, 500]]),
                torch.tensor([1, 2]),
                None
            )
            mock_vis.return_value = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # 创建真实wrapper实例并替换AWL_MODEL
            wrapper = self.wrapper_class()
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
            
            # 执行测试
            result = wrapper._step1_layout_detector("test.png")
            
            # 验证返回结构
            self.assertEqual(len(result), 4)
            boxes, classes, vis_img, time_used = result
            
            # 验证类型
            self.assertIsInstance(boxes, np.ndarray)
            self.assertIsInstance(classes, np.ndarray)
            self.assertIsInstance(vis_img, np.ndarray)
            self.assertIsInstance(time_used, float)
            
            # 验证形状
            self.assertEqual(boxes.shape, (2, 4))
            self.assertEqual(classes.shape, (2,))
    
    def test_output_format_no_detections(self):
        """测试无检测结果时的输出格式"""
        with patch('phishintention.pred_rcnn') as mock_pred, \
             patch('phishintention.vis') as mock_vis:
            
            # 设置无检测结果
            mock_pred.return_value = (None, None, None)
            mock_vis.return_value = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # 创建wrapper实例
            wrapper = self.wrapper_class()
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
            
            # 执行测试
            boxes, classes, vis_img, time_used = wrapper._step1_layout_detector("test.png")
            
            # 验证
            self.assertIsNone(boxes)
            self.assertIsNone(classes)
            self.assertIsInstance(vis_img, np.ndarray)
            self.assertIsInstance(time_used, float)
    
    def test_tensor_to_numpy_conversion(self):
        """验证tensor到numpy的转换"""
        with patch('phishintention.pred_rcnn') as mock_pred, \
             patch('phishintention.vis') as mock_vis:
            
            # 创建测试tensor
            test_boxes = torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]])
            test_classes = torch.tensor([0, 1])
            
            mock_pred.return_value = (test_boxes, test_classes, None)
            mock_vis.return_value = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # 执行测试
            wrapper = self.wrapper_class()
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
            
            boxes, classes, _, _ = wrapper._step1_layout_detector("test.png")
            
            # 验证转换
            self.assertIsInstance(boxes, np.ndarray)
            self.assertIsInstance(classes, np.ndarray)
            
            # 验证数据一致性
            np.testing.assert_array_equal(boxes, test_boxes.numpy())
            np.testing.assert_array_equal(classes, test_classes.numpy())


def run_tests():
    """运行测试的简单方法"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStep1LayoutDetectorSimple)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出简单结果
    print(f"\n测试结果: {len(result.failures)} 失败, {len(result.errors)} 错误")
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    # 直接运行
    success = run_tests()
    
    # 或者使用标准方式
    # unittest.main()
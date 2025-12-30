import unittest
from unittest.mock import Mock, patch
import numpy as np


class TestStep2LogoMatcher(unittest.TestCase):
    """测试_step2_logo_matcher函数的单元测试"""
    
    def setUp(self):
        """测试前准备"""
        # 导入被测试的类
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper
        
        # 创建模拟的wrapper实例
        self.wrapper = Mock(spec=PhishIntentionWrapper)
        self.wrapper.SIAMESE_MODEL = Mock()
        self.wrapper.OCR_MODEL = Mock()
        self.wrapper.LOGO_FEATS = []
        self.wrapper.LOGO_FILES = []
        self.wrapper.DOMAIN_MAP_PATH = '/mock/path/domain_map.pkl'
        self.wrapper.SIAMESE_THRE = 0.5
    
    def test_output_format_with_match(self):
        """测试匹配到品牌时的输出格式"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # 设置模拟返回值
            mock_check.return_value = (
                "Microsoft",          # pred_target
                "microsoft.com",      # matched_domain
                [100, 150, 300, 400],  # matched_coord
                0.85                   # siamese_conf
            )
            
            # 创建真实wrapper实例并替换属性
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE
            
            # 测试数据
            logo_pred_boxes = np.array([[50, 60, 200, 300]])
            url = "https://test-site.com"
            screenshot_path = "test.png"
            
            # 执行测试
            result = wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)
            
            # 验证返回结构
            self.assertEqual(len(result), 5)
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = result
            
            # 验证类型
            self.assertIsInstance(pred_target, str)
            self.assertIsInstance(matched_domain, str)
            self.assertIsInstance(matched_coord, list)
            self.assertIsInstance(siamese_conf, float)
            self.assertIsInstance(logo_match_time, float)
            
            # 验证函数调用
            mock_check.assert_called_once()
            
            # 验证调用参数
            args, kwargs = mock_check.call_args
            
            # 修复：正确比较NumPy数组
            # 使用np.array_equal来比较两个NumPy数组
            self.assertTrue(np.array_equal(kwargs['logo_boxes'], logo_pred_boxes))
            
            # 或者逐个元素比较（更详细）
            received_boxes = kwargs['logo_boxes']
            self.assertIsInstance(received_boxes, np.ndarray)
            self.assertEqual(received_boxes.shape, logo_pred_boxes.shape)
            for i in range(len(received_boxes)):
                for j in range(4):
                    self.assertEqual(received_boxes[i][j], logo_pred_boxes[i][j])
            
            self.assertEqual(kwargs['url'], url)
            self.assertEqual(kwargs['shot_path'], screenshot_path)
            self.assertEqual(kwargs['model'], wrapper.SIAMESE_MODEL)
            self.assertEqual(kwargs['ocr_model'], wrapper.OCR_MODEL)
            self.assertEqual(kwargs['logo_feat_list'], wrapper.LOGO_FEATS)
            self.assertEqual(kwargs['file_name_list'], wrapper.LOGO_FILES)
            self.assertEqual(kwargs['domain_map_path'], wrapper.DOMAIN_MAP_PATH)
            self.assertEqual(kwargs['ts'], wrapper.SIAMESE_THRE)
    
    def test_output_format_no_match(self):
        """测试没有匹配到品牌时的输出格式"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # 设置模拟返回值（没有匹配）
            mock_check.return_value = (None, None, None, None)
            
            # 创建wrapper实例
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE
            
            # 测试数据
            logo_pred_boxes = np.array([[50, 60, 200, 300]])
            url = "https://test-site.com"
            screenshot_path = "test.png"
            
            # 执行测试
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)
            
            # 验证返回None值
            self.assertIsNone(pred_target)
            self.assertIsNone(matched_domain)
            self.assertIsNone(matched_coord)
            self.assertIsNone(siamese_conf)
            self.assertIsInstance(logo_match_time, float)
    
    def test_time_measurement(self):
        """测试时间测量功能"""
        import time
        
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # 模拟需要时间的处理
            def delayed_check(*args, **kwargs):
                time.sleep(0.05)  # 50ms延迟
                return ("Brand", "brand.com", [0, 0, 100, 100], 0.9)
            
            mock_check.side_effect = delayed_check
            
            # 创建wrapper实例
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE
            
            # 执行测试
            _, _, _, _, logo_match_time = wrapper._step2_logo_matcher(
                np.array([[0, 0, 100, 100]]), 
                "https://test.com", 
                "test.png"
            )
            
            # 验证时间测量
            self.assertGreaterEqual(logo_match_time, 0.05)
    
    def test_empty_logo_boxes(self):
        """测试空logo boxes的情况"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # 创建wrapper实例
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE
            
            # 测试空数组
            logo_pred_boxes = np.array([])
            url = "https://test-site.com"
            screenshot_path = "test.png"
            
            # 执行测试
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)
            
            # 验证返回None值
            self.assertIsNone(pred_target)
            self.assertIsNone(matched_domain)
            self.assertIsNone(matched_coord)
            self.assertIsNone(siamese_conf)
            self.assertIsInstance(logo_match_time, float)
            
            # 验证mock函数没有被调用
            mock_check.assert_not_called()
    
    def test_single_logo_box_format(self):
        """测试单个logo box的格式处理"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # 设置模拟返回值
            mock_check.return_value = ("Brand", "brand.com", [0, 0, 100, 100], 0.8)
            
            # 创建wrapper实例
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE
            
            # 测试单个边界框（一维数组）
            logo_pred_boxes = np.array([0, 0, 100, 100])
            url = "https://test-site.com"
            screenshot_path = "test.png"
            
            # 执行测试
            result = wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)
            
            # 验证返回结果
            self.assertIsNotNone(result)
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = result
            
            # 验证mock被调用，并检查参数格式
            mock_check.assert_called_once()
            args, kwargs = mock_check.call_args
            
            # 验证logo_boxes参数是二维数组格式
            received_boxes = kwargs['logo_boxes']
            self.assertIsInstance(received_boxes, np.ndarray)
            self.assertEqual(received_boxes.shape, (1, 4))  # 应该是 (1, 4) 而不是 (4,)
    
    def test_multiple_logo_boxes(self):
        """测试多个logo boxes的情况"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # 设置模拟返回值
            mock_check.return_value = ("Brand", "brand.com", [50, 50, 150, 150], 0.9)
            
            # 创建wrapper实例
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE
            
            # 测试多个边界框
            logo_pred_boxes = np.array([
                [0, 0, 100, 100],
                [50, 50, 150, 150],
                [200, 200, 300, 300]
            ])
            url = "https://test-site.com"
            screenshot_path = "test.png"
            
            # 执行测试
            result = wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)
            
            # 验证mock被调用，并检查参数
            mock_check.assert_called_once()
            args, kwargs = mock_check.call_args
            
            # 验证logo_boxes参数
            received_boxes = kwargs['logo_boxes']
            self.assertIsInstance(received_boxes, np.ndarray)
            self.assertEqual(received_boxes.shape, (3, 4))
            self.assertTrue(np.array_equal(received_boxes, logo_pred_boxes))


if __name__ == '__main__':
    unittest.main()
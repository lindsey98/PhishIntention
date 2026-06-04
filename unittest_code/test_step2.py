import unittest
from unittest.mock import Mock, patch
import numpy as np


class TestStep2LogoMatcher(unittest.TestCase):
    """Unit tests for the _step2_logo_matcher function"""

    def setUp(self):
        """Set up before each test"""
        # Import the class under test
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper

        # Create a mock wrapper instance
        self.wrapper = Mock(spec=PhishIntentionWrapper)
        self.wrapper.SIAMESE_MODEL = Mock()
        self.wrapper.OCR_MODEL = Mock()
        self.wrapper.LOGO_FEATS = []
        self.wrapper.LOGO_FILES = []
        self.wrapper.DOMAIN_MAP_PATH = '/mock/path/domain_map.pkl'
        self.wrapper.SIAMESE_THRE = 0.5
    
    def test_output_format_with_match(self):
        """Test the output format when a brand is matched"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # Configure the mock return value
            mock_check.return_value = (
                "Microsoft",          # pred_target
                "microsoft.com",      # matched_domain
                [100, 150, 300, 400],  # matched_coord
                0.85                   # siamese_conf
            )
            
            # Create a real wrapper instance and replace its attributes
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE

            # Test data
            logo_pred_boxes = np.array([[50, 60, 200, 300]])
            url = "https://test-site.com"
            screenshot_path = "test.png"

            # Run the method under test
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)

            # Verify the types
            self.assertIsInstance(pred_target, str)
            self.assertIsInstance(matched_domain, str)
            self.assertIsInstance(matched_coord, list)
            self.assertIsInstance(siamese_conf, float)
            self.assertIsInstance(logo_match_time, float)

            # Verify the function call
            mock_check.assert_called_once()

            # Verify the call arguments
            args, kwargs = mock_check.call_args

            # Fix: compare NumPy arrays correctly
            # Use np.array_equal to compare two NumPy arrays
            self.assertTrue(np.array_equal(kwargs['logo_boxes'], logo_pred_boxes))

            # Or compare element by element (more detailed)
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
        """Test the output format when no brand is matched"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # Configure the mock return value (no match)
            mock_check.return_value = (None, None, None, None)

            # Create a wrapper instance
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE

            # Test data
            logo_pred_boxes = np.array([[50, 60, 200, 300]])
            url = "https://test-site.com"
            screenshot_path = "test.png"

            # Run the method under test
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)

            # Verify the returned None values
            self.assertIsNone(pred_target)
            self.assertIsNone(matched_domain)
            self.assertIsNone(matched_coord)
            self.assertIsNone(siamese_conf)
            self.assertIsInstance(logo_match_time, float)
    
    def test_time_measurement(self):
        """Test the timing measurement"""
        import time

        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # Simulate processing that takes time
            def delayed_check(*args, **kwargs):
                time.sleep(0.05)  # 50ms delay
                return ("Brand", "brand.com", [0, 0, 100, 100], 0.9)

            mock_check.side_effect = delayed_check

            # Create a wrapper instance
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE

            # Run the method under test
            _, _, _, _, logo_match_time = wrapper._step2_logo_matcher(
                np.array([[0, 0, 100, 100]]),
                "https://test.com",
                "test.png"
            )

            # Verify the timing measurement
            self.assertGreaterEqual(logo_match_time, 0.05)
    
    def test_empty_logo_boxes(self):
        """Test the case of empty logo boxes"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # Create a wrapper instance
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE

            # Test an empty array
            logo_pred_boxes = np.array([])
            url = "https://test-site.com"
            screenshot_path = "test.png"

            # Run the method under test
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)

            # Verify the returned None values
            self.assertIsNone(pred_target)
            self.assertIsNone(matched_domain)
            self.assertIsNone(matched_coord)
            self.assertIsNone(siamese_conf)
            self.assertIsInstance(logo_match_time, float)

            # Verify the mock function was not called
            mock_check.assert_not_called()
    
    def test_single_logo_box_format(self):
        """Test format handling for a single logo box"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # Configure the mock return value
            mock_check.return_value = ("Brand", "brand.com", [0, 0, 100, 100], 0.8)

            # Create a wrapper instance
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE
            
            # Test a single bounding box (1D array)
            logo_pred_boxes = np.array([0, 0, 100, 100])
            url = "https://test-site.com"
            screenshot_path = "test.png"

            # Run the method under test
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)

            # Verify the returned values are not None
            self.assertIsNotNone(pred_target)
            self.assertIsNotNone(matched_domain)
            self.assertIsNotNone(matched_coord)
            self.assertIsNotNone(siamese_conf)
            self.assertIsInstance(logo_match_time, float)

            # Verify the mock was called and check the argument format
            mock_check.assert_called_once()
            args, kwargs = mock_check.call_args

            # Verify the logo_boxes argument is a 2D array
            received_boxes = kwargs['logo_boxes']
            self.assertIsInstance(received_boxes, np.ndarray)
            self.assertEqual(received_boxes.shape, (1, 4))  # Should be (1, 4) rather than (4,)
    
    def test_multiple_logo_boxes(self):
        """Test the case of multiple logo boxes"""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            # Configure the mock return value
            mock_check.return_value = ("Brand", "brand.com", [50, 50, 150, 150], 0.9)

            # Create a wrapper instance
            wrapper = self.wrapper_class()
            wrapper.SIAMESE_MODEL = self.wrapper.SIAMESE_MODEL
            wrapper.OCR_MODEL = self.wrapper.OCR_MODEL
            wrapper.LOGO_FEATS = self.wrapper.LOGO_FEATS
            wrapper.LOGO_FILES = self.wrapper.LOGO_FILES
            wrapper.DOMAIN_MAP_PATH = self.wrapper.DOMAIN_MAP_PATH
            wrapper.SIAMESE_THRE = self.wrapper.SIAMESE_THRE
            
            # Test multiple bounding boxes
            logo_pred_boxes = np.array([
                [0, 0, 100, 100],
                [50, 50, 150, 150],
                [200, 200, 300, 300]
            ])
            url = "https://test-site.com"
            screenshot_path = "test.png"

            # Run the method under test
            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)

            # Verify the returned values are not None
            self.assertIsNotNone(pred_target)
            self.assertIsNotNone(matched_domain)
            self.assertIsNotNone(matched_coord)
            self.assertIsNotNone(siamese_conf)
            self.assertIsInstance(logo_match_time, float)

            # Verify the mock was called and check the arguments
            mock_check.assert_called_once()
            args, kwargs = mock_check.call_args

            # Verify the logo_boxes argument
            received_boxes = kwargs['logo_boxes']
            self.assertIsInstance(received_boxes, np.ndarray)
            self.assertEqual(received_boxes.shape, (3, 4))
            self.assertTrue(np.array_equal(received_boxes, logo_pred_boxes))


if __name__ == '__main__':
    unittest.main()
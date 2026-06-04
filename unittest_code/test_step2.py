import unittest
from unittest.mock import Mock, patch
import numpy as np


class TestStep2LogoMatcher(unittest.TestCase):
    """Unit tests for _step2_logo_matcher (OCR-aided Siamese logo matching)."""

    def setUp(self):
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper

    def _make_wrapper(self):
        wrapper = self.wrapper_class()
        wrapper.SIAMESE_MODEL = Mock()
        wrapper.OCR_MODEL = Mock()
        wrapper.LOGO_FEATS = []
        wrapper.LOGO_FILES = []
        wrapper.DOMAIN_MAP_PATH = '/mock/path/domain_map.pkl'
        wrapper.SIAMESE_THRE = 0.5
        return wrapper

    def test_passes_correct_arguments_to_matcher(self):
        """A detected logo box is forwarded to check_domain_brand_inconsistency with the expected arguments."""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            mock_check.return_value = ("Microsoft", "microsoft.com", [100, 150, 300, 400], 0.85)

            wrapper = self._make_wrapper()
            logo_pred_boxes = np.array([[50, 60, 200, 300]])
            url = "https://test-site.com"
            screenshot_path = "test.png"

            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(logo_pred_boxes, url, screenshot_path)

            self.assertEqual(pred_target, "Microsoft")
            self.assertEqual(matched_domain, "microsoft.com")
            self.assertIsInstance(logo_match_time, float)

            mock_check.assert_called_once()
            _, kwargs = mock_check.call_args
            self.assertTrue(np.array_equal(kwargs['logo_boxes'], logo_pred_boxes))
            self.assertEqual(kwargs['url'], url)
            self.assertEqual(kwargs['shot_path'], screenshot_path)
            self.assertEqual(kwargs['model'], wrapper.SIAMESE_MODEL)
            self.assertEqual(kwargs['ocr_model'], wrapper.OCR_MODEL)
            self.assertEqual(kwargs['logo_feat_list'], wrapper.LOGO_FEATS)
            self.assertEqual(kwargs['file_name_list'], wrapper.LOGO_FILES)
            self.assertEqual(kwargs['domain_map_path'], wrapper.DOMAIN_MAP_PATH)
            self.assertEqual(kwargs['ts'], wrapper.SIAMESE_THRE)

    def test_empty_logo_boxes_short_circuit(self):
        """With no logo boxes the matcher is never invoked and the result is None."""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            wrapper = self._make_wrapper()

            pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time = \
                wrapper._step2_logo_matcher(np.array([]), "https://test-site.com", "test.png")

            self.assertIsNone(pred_target)
            self.assertIsNone(matched_domain)
            self.assertIsNone(matched_coord)
            self.assertIsNone(siamese_conf)
            self.assertIsInstance(logo_match_time, float)
            mock_check.assert_not_called()

    def test_single_box_reshaped_to_2d(self):
        """A 1-D single box [x1, y1, x2, y2] is reshaped to (1, 4) before matching."""
        with patch('phishintention.check_domain_brand_inconsistency') as mock_check:
            mock_check.return_value = ("Brand", "brand.com", [0, 0, 100, 100], 0.8)

            wrapper = self._make_wrapper()
            wrapper._step2_logo_matcher(np.array([0, 0, 100, 100]), "https://test-site.com", "test.png")

            mock_check.assert_called_once()
            _, kwargs = mock_check.call_args
            received_boxes = kwargs['logo_boxes']
            self.assertIsInstance(received_boxes, np.ndarray)
            self.assertEqual(received_boxes.shape, (1, 4))


if __name__ == '__main__':
    unittest.main()

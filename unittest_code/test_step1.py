import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch


class TestStep1LayoutDetectorSimple(unittest.TestCase):
    """Unit tests for _step1_layout_detector (AWL layout detection)."""

    def setUp(self):
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper

    def test_output_format_with_detections(self):
        """Detections are returned as a 4-tuple with tensors converted to numpy."""
        with patch('phishintention.pred_rcnn') as mock_pred, \
             patch('phishintention.vis') as mock_vis:

            mock_pred.return_value = (
                torch.tensor([[10, 20, 100, 150], [200, 300, 400, 500]]),
                torch.tensor([1, 2]),
                None
            )
            mock_vis.return_value = np.zeros((600, 800, 3), dtype=np.uint8)

            wrapper = self.wrapper_class()
            wrapper.AWL_MODEL = Mock()

            boxes, classes, vis_img, time_used = wrapper._step1_layout_detector("test.png")

            self.assertIsInstance(boxes, np.ndarray)
            self.assertIsInstance(classes, np.ndarray)
            self.assertEqual(boxes.shape, (2, 4))
            self.assertEqual(classes.shape, (2,))
            np.testing.assert_array_equal(boxes, [[10, 20, 100, 150], [200, 300, 400, 500]])
            self.assertIsInstance(vis_img, np.ndarray)
            self.assertIsInstance(time_used, float)

    def test_output_format_no_detections(self):
        """When the detector finds nothing, boxes/classes are None (benign path)."""
        with patch('phishintention.pred_rcnn') as mock_pred, \
             patch('phishintention.vis') as mock_vis:

            mock_pred.return_value = (None, None, None)
            mock_vis.return_value = np.zeros((600, 800, 3), dtype=np.uint8)

            wrapper = self.wrapper_class()
            wrapper.AWL_MODEL = Mock()

            boxes, classes, vis_img, time_used = wrapper._step1_layout_detector("test.png")

            self.assertIsNone(boxes)
            self.assertIsNone(classes)
            self.assertIsInstance(vis_img, np.ndarray)
            self.assertIsInstance(time_used, float)


if __name__ == '__main__':
    unittest.main()

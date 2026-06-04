import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch


class TestStep1LayoutDetectorSimple(unittest.TestCase):
    """Lightweight unit tests validating core behavior"""

    def setUp(self):
        """Set up the test environment"""
        # Import the class under test
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper

        # Create a mock wrapper instance
        self.wrapper = Mock(spec=PhishIntentionWrapper)
        self.wrapper.AWL_MODEL = Mock()
    
    def test_output_format_with_detections(self):
        """Test the output format when detections are present"""
        with patch('phishintention.pred_rcnn') as mock_pred, \
             patch('phishintention.vis') as mock_vis:

            # Configure the mock return value
            mock_pred.return_value = (
                torch.tensor([[10, 20, 100, 150], [200, 300, 400, 500]]),
                torch.tensor([1, 2]),
                None
            )
            mock_vis.return_value = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # Create a real wrapper instance and replace AWL_MODEL
            wrapper = self.wrapper_class()
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL

            # Run the method under test
            result = wrapper._step1_layout_detector("test.png")

            # Verify the return structure
            self.assertEqual(len(result), 4)
            boxes, classes, vis_img, time_used = result

            # Verify the types
            self.assertIsInstance(boxes, np.ndarray)
            self.assertIsInstance(classes, np.ndarray)
            self.assertIsInstance(vis_img, np.ndarray)
            self.assertIsInstance(time_used, float)

            # Verify the shapes
            self.assertEqual(boxes.shape, (2, 4))
            self.assertEqual(classes.shape, (2,))
    
    def test_output_format_no_detections(self):
        """Test the output format when no detections are present"""
        with patch('phishintention.pred_rcnn') as mock_pred, \
             patch('phishintention.vis') as mock_vis:

            # Configure an empty detection result
            mock_pred.return_value = (None, None, None)
            mock_vis.return_value = np.zeros((600, 800, 3), dtype=np.uint8)

            # Create a wrapper instance
            wrapper = self.wrapper_class()
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL

            # Run the method under test
            boxes, classes, vis_img, time_used = wrapper._step1_layout_detector("test.png")

            # Verify
            self.assertIsNone(boxes)
            self.assertIsNone(classes)
            self.assertIsInstance(vis_img, np.ndarray)
            self.assertIsInstance(time_used, float)
    
    def test_tensor_to_numpy_conversion(self):
        """Verify the conversion from tensor to numpy"""
        with patch('phishintention.pred_rcnn') as mock_pred, \
             patch('phishintention.vis') as mock_vis:

            # Create test tensors
            test_boxes = torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]])
            test_classes = torch.tensor([0, 1])
            
            mock_pred.return_value = (test_boxes, test_classes, None)
            mock_vis.return_value = np.zeros((600, 800, 3), dtype=np.uint8)

            # Run the method under test
            wrapper = self.wrapper_class()
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL

            boxes, classes, _, _ = wrapper._step1_layout_detector("test.png")

            # Verify the conversion
            self.assertIsInstance(boxes, np.ndarray)
            self.assertIsInstance(classes, np.ndarray)

            # Verify data consistency
            np.testing.assert_array_equal(boxes, test_boxes.numpy())
            np.testing.assert_array_equal(classes, test_classes.numpy())


def run_tests():
    """Simple helper to run the tests"""
    # Create the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStep1LayoutDetectorSimple)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print a brief summary
    print(f"\nTest results: {len(result.failures)} failures, {len(result.errors)} errors")
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    # Run directly
    success = run_tests()

    # Alternatively, use the standard entry point
    # unittest.main()
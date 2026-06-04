import unittest
from unittest.mock import Mock, patch
import numpy as np


class TestStep3CrpClassifier(unittest.TestCase):
    """Unit tests for the _step3_crp_classifier function"""

    def setUp(self):
        """Set up before each test"""
        # Import the class under test
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper

        # Create a mock wrapper instance
        self.wrapper = Mock(spec=PhishIntentionWrapper)
        self.wrapper.CRP_CLASSIFIER = Mock()

    def test_output_format_html_heuristic_noncrp(self):
        """Test the case where the HTML heuristic returns nonCRP"""
        with patch('phishintention.html_heuristic') as mock_html, \
             patch('phishintention.credential_classifier_mixed') as mock_classifier:

            # Mock the HTML heuristic returning 1 (nonCRP)
            mock_html.return_value = 1
            # Mock the classifier returning 0 (CRP)
            mock_classifier.return_value = 0

            # Create a real wrapper instance
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER

            # Test data
            screenshot_path = "test.png"
            html_path = "test.html"
            pred_boxes = np.array([[10, 20, 100, 150], [200, 300, 400, 500]])
            pred_classes = np.array([1, 2])

            # Run the method under test
            cre_pred, crp_class_time = wrapper._step3_crp_classifier(
                screenshot_path, html_path, pred_boxes, pred_classes
            )

            # Verify the return types
            self.assertIsInstance(cre_pred, int)
            self.assertIsInstance(crp_class_time, float)

            # Verify the function calls
            mock_html.assert_called_once_with(html_path)
            mock_classifier.assert_called_once_with(
                img=screenshot_path,
                coords=pred_boxes,
                types=pred_classes,
                model=wrapper.CRP_CLASSIFIER
            )

            # Verify the result
            self.assertEqual(cre_pred, 0)  # The classifier's result
    
    def test_output_format_html_heuristic_crp(self):
        """Test the case where the HTML heuristic returns CRP"""
        with patch('phishintention.html_heuristic') as mock_html, \
             patch('phishintention.credential_classifier_mixed') as mock_classifier:

            # Mock the HTML heuristic returning 0 (CRP)
            mock_html.return_value = 0

            # Create a wrapper instance
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER

            # Test data
            screenshot_path = "test.png"
            html_path = "test.html"
            pred_boxes = np.array([[10, 20, 100, 150]])
            pred_classes = np.array([1])

            # Run the method under test
            cre_pred, crp_class_time = wrapper._step3_crp_classifier(
                screenshot_path, html_path, pred_boxes, pred_classes
            )

            # Verify the return types
            self.assertIsInstance(cre_pred, int)
            self.assertIsInstance(crp_class_time, float)

            # Verify the function calls
            mock_html.assert_called_once_with(html_path)
            # When the HTML heuristic returns CRP, the classifier should not be called
            mock_classifier.assert_not_called()

            # Verify the result
            self.assertEqual(cre_pred, 0)  # The HTML heuristic's result
    
    def test_output_format_classifier_results(self):
        """Test different classifier results"""
        test_cases = [
            (1, 1, "HTML nonCRP, classifier nonCRP"),
            (1, 0, "HTML nonCRP, classifier CRP"),
            (0, None, "HTML CRP, classifier not called"),
        ]
        
        for html_result, classifier_result, description in test_cases:
            with self.subTest(description=description):
                with patch('phishintention.html_heuristic') as mock_html, \
                     patch('phishintention.credential_classifier_mixed') as mock_classifier:
                    
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
                        expected = 0  # HTML returns CRP
                    else:
                        expected = classifier_result  # The classifier's result
                    
                    self.assertEqual(cre_pred, expected)
    
    def test_time_measurement(self):
        """Test the timing measurement"""
        import time

        with patch('phishintention.html_heuristic') as mock_html, \
             patch('phishintention.credential_classifier_mixed') as mock_classifier:

            # Simulate processing that takes time
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
            
            # Verify the total time (HTML + classifier)
            self.assertGreaterEqual(crp_class_time, 0.05)


if __name__ == '__main__':
    unittest.main()
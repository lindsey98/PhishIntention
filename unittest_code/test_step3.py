import unittest
from unittest.mock import Mock, patch
import numpy as np


class TestStep3CrpClassifier(unittest.TestCase):
    """Unit tests for _step3_crp_classifier (CRP detection)."""

    def setUp(self):
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper

    def test_html_noncrp_falls_back_to_classifier(self):
        """When the HTML heuristic returns nonCRP, the image classifier decides the result."""
        with patch('phishintention.html_heuristic') as mock_html, \
             patch('phishintention.credential_classifier_mixed') as mock_classifier:

            mock_html.return_value = 1   # nonCRP
            mock_classifier.return_value = 0   # CRP

            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = Mock()

            pred_boxes = np.array([[10, 20, 100, 150], [200, 300, 400, 500]])
            pred_classes = np.array([1, 2])
            cre_pred, crp_class_time = wrapper._step3_crp_classifier(
                "test.png", "test.html", pred_boxes, pred_classes
            )

            self.assertEqual(cre_pred, 0)
            self.assertIsInstance(crp_class_time, float)
            mock_html.assert_called_once_with("test.html")
            mock_classifier.assert_called_once_with(
                img="test.png", coords=pred_boxes, types=pred_classes, model=wrapper.CRP_CLASSIFIER
            )

    def test_html_crp_short_circuits_classifier(self):
        """When the HTML heuristic already reports CRP, the image classifier is skipped."""
        with patch('phishintention.html_heuristic') as mock_html, \
             patch('phishintention.credential_classifier_mixed') as mock_classifier:

            mock_html.return_value = 0   # CRP

            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = Mock()

            cre_pred, crp_class_time = wrapper._step3_crp_classifier(
                "test.png", "test.html", np.array([[10, 20, 100, 150]]), np.array([1])
            )

            self.assertEqual(cre_pred, 0)
            self.assertIsInstance(crp_class_time, float)
            mock_html.assert_called_once_with("test.html")
            mock_classifier.assert_not_called()


if __name__ == '__main__':
    unittest.main()

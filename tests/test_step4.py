import unittest
from unittest.mock import Mock, patch, MagicMock


class TestStep4DynamicAnalysis(unittest.TestCase):
    """Unit tests for _step4_dynamic_analysis (CRP locator / dynamic analysis)."""

    def setUp(self):
        from phishintention.pipeline import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper

    def test_invokes_crp_locator_and_releases_driver(self):
        """The driver is loaded, crp_locator is called with the right models, and the driver is always quit."""
        with patch('phishintention.pipeline.driver_loader') as mock_loader, \
             patch('phishintention.pipeline.crp_locator') as mock_locator:

            mock_driver = MagicMock()
            mock_loader.return_value = mock_driver
            mock_locator.return_value = ("https://updated-url.com", "/new/screenshot.png", True, 2.5)

            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = Mock()
            wrapper.AWL_MODEL = Mock()
            wrapper.CRP_LOCATOR_MODEL = Mock()

            url = "https://original-url.com"
            screenshot_path = "/original/screenshot.png"
            new_url, new_screenshot_path, successful, process_time = \
                wrapper._step4_dynamic_analysis(url, screenshot_path, [[10, 20, 100, 150]], [1])

            mock_loader.assert_called_once()
            mock_locator.assert_called_once_with(
                url=url,
                screenshot_path=screenshot_path,
                cls_model=wrapper.CRP_CLASSIFIER,
                ele_model=wrapper.AWL_MODEL,
                login_model=wrapper.CRP_LOCATOR_MODEL,
                driver=mock_driver,
            )
            mock_driver.quit.assert_called_once()

            self.assertEqual(new_url, "https://updated-url.com")
            self.assertEqual(new_screenshot_path, "/new/screenshot.png")
            self.assertTrue(successful)
            self.assertEqual(process_time, 2.5)


if __name__ == '__main__':
    unittest.main()

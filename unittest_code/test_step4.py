import unittest
from unittest.mock import Mock, patch, MagicMock


class TestStep4DynamicAnalysis(unittest.TestCase):
    """Unit tests for the _step4_dynamic_analysis function"""

    def setUp(self):
        """Set up before each test"""
        # Import the class under test
        from phishintention import PhishIntentionWrapper
        self.wrapper_class = PhishIntentionWrapper

        # Create a mock wrapper instance
        self.wrapper = Mock(spec=PhishIntentionWrapper)
        self.wrapper.CRP_CLASSIFIER = Mock()
        self.wrapper.AWL_MODEL = Mock()
        self.wrapper.CRP_LOCATOR_MODEL = Mock()

    def test_output_format_successful_analysis(self):
        """Test the case where dynamic analysis successfully finds a CRP"""
        with patch('phishintention.driver_loader') as mock_loader, \
             patch('phishintention.crp_locator') as mock_locator:

            # Mock the driver
            mock_driver = MagicMock()
            mock_loader.return_value = mock_driver

            # Mock crp_locator returning success
            mock_locator.return_value = (
                "https://updated-url.com",  # url
                "/new/screenshot.png",      # screenshot_path
                True,                       # successful
                2.5                         # process_time
            )

            # Create a real wrapper instance
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
            wrapper.CRP_LOCATOR_MODEL = self.wrapper.CRP_LOCATOR_MODEL

            # Test data
            url = "https://original-url.com"
            screenshot_path = "/original/screenshot.png"
            pred_boxes = [[10, 20, 100, 150]]
            pred_classes = [1]

            # Run the method under test
            result = wrapper._step4_dynamic_analysis(
                url, screenshot_path, pred_boxes, pred_classes
            )

            # Verify the return structure
            self.assertEqual(len(result), 4)
            new_url, new_screenshot_path, successful, process_time = result

            # Verify the return types
            self.assertIsInstance(new_url, str)
            self.assertIsInstance(new_screenshot_path, str)
            self.assertIsInstance(successful, bool)
            self.assertIsInstance(process_time, float)

            # Verify the driver-related calls
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

            # Verify the return values
            self.assertEqual(new_url, "https://updated-url.com")
            self.assertEqual(new_screenshot_path, "/new/screenshot.png")
            self.assertTrue(successful)
            self.assertEqual(process_time, 2.5)
    
    def test_output_format_unsuccessful_analysis(self):
        """Test the case where dynamic analysis does not find a CRP"""
        with patch('phishintention.driver_loader') as mock_loader, \
             patch('phishintention.crp_locator') as mock_locator:

            # Mock the driver
            mock_driver = MagicMock()
            mock_loader.return_value = mock_driver

            # Mock crp_locator returning failure
            mock_locator.return_value = (
                "https://original-url.com",  # url unchanged
                "/original/screenshot.png",  # screenshot_path unchanged
                False,                       # successful
                1.8                          # process_time
            )

            # Create a wrapper instance
            wrapper = self.wrapper_class()
            wrapper.CRP_CLASSIFIER = self.wrapper.CRP_CLASSIFIER
            wrapper.AWL_MODEL = self.wrapper.AWL_MODEL
            wrapper.CRP_LOCATOR_MODEL = self.wrapper.CRP_LOCATOR_MODEL

            # Run the method under test
            new_url, new_screenshot_path, successful, process_time = \
                wrapper._step4_dynamic_analysis(
                    "https://original-url.com",
                    "/original/screenshot.png",
                    [[0, 0, 100, 100]],
                    [1]
                )

            # Verify the return types
            self.assertIsInstance(new_url, str)
            self.assertIsInstance(new_screenshot_path, str)
            self.assertIsInstance(successful, bool)
            self.assertIsInstance(process_time, float)

            # Verify the return values
            self.assertEqual(new_url, "https://original-url.com")
            self.assertEqual(new_screenshot_path, "/original/screenshot.png")
            self.assertFalse(successful)
            self.assertEqual(process_time, 1.8)
    
    def test_driver_lifecycle(self):
        """Test the full lifecycle of the driver"""
        with patch('phishintention.driver_loader') as mock_loader, \
             patch('phishintention.crp_locator') as mock_locator:

            # Track driver usage
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

            # Run the method under test
            wrapper._step4_dynamic_analysis("url", "path", [], [])

            # Verify the driver is properly initialized and cleaned up
            mock_loader.assert_called_once()
            self.assertTrue(mock_driver.quit_called, "driver.quit() should be called")
    
    def test_time_measurement(self):
        """Test that the timing measurement is included in the return value"""
        with patch('phishintention.driver_loader') as mock_loader, \
             patch('phishintention.crp_locator') as mock_locator:

            mock_driver = MagicMock()
            mock_loader.return_value = mock_driver

            # Set up different processing times
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
                    
                    # Verify the returned time matches the mocked time
                    self.assertEqual(returned_time, process_time)


if __name__ == '__main__':
    unittest.main()
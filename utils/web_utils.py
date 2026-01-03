from selenium.common.exceptions import NoSuchElementException, TimeoutException, MoveTargetOutOfBoundsException, WebDriverException
from selenium import webdriver
import helium
import time
import re
import logging
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService

logger = logging.getLogger(__name__)

def initialize_chrome_settings():
    '''
    initialize chrome settings
    '''
    options = webdriver.ChromeOptions()

    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--ignore-certificate-errors')  # ignore errors
    options.add_argument('--ignore-ssl-errors')
    options.add_argument("--headless") # FIXME: do not disable browser (have some issues: https://github.com/mherrmann/selenium-python-helium/issues/47)
    options.add_argument('--no-proxy-server')
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")
    #options.add_argument('--no-proxy-server')
    #options.add_argument("--proxy-server=http://127.0.0.1:7897;https://127.0.0.1:7897")
    #options.add_argument("--proxy-bypass-list=<-loopback>")

    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920,1080')  # fix screenshot size
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    options.set_capability('unhandledPromptBehavior', 'dismiss')  # dismiss


    return options

def click_button(button_text, max_retries=2):
    '''
    Click a button with retry mechanism
    :param button_text: text of the button to click
    :param max_retries: maximum number of retry attempts
    :return: True if successful, False otherwise
    '''
    helium.Config.implicit_wait_secs = 2 # this is the implicit timeout for helium
    driver = helium.get_driver()
    driver.implicitly_wait(2)
    current_url = None
    
    for attempt in range(max_retries + 1):
        try:
            current_url = driver.current_url
            helium.click(helium.Button(button_text))
            logger.debug(f'Successfully clicked button "{button_text}" (URL: {current_url})')
            return True
        except (TimeoutException, NoSuchElementException) as e:
            if attempt < max_retries:
                logger.debug(f'Retry {attempt + 1}/{max_retries} for button "{button_text}" (URL: {current_url}). Error: {type(e).__name__}: {str(e)}')
                time.sleep(1)
                continue
            else:
                logger.warning(f'Failed to click button "{button_text}" after {max_retries + 1} attempts (URL: {current_url}). Error type: {type(e).__name__}, Message: {str(e)}')
                return False
        except WebDriverException as e:
            logger.error(f'WebDriver error clicking button "{button_text}" (URL: {current_url}). Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
            return False
        except Exception as e:
            logger.error(f'Unexpected error clicking button "{button_text}" (URL: {current_url}). Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
            return False
    return False

def get_page_text(driver):
    '''
    get body text from html
    :param driver: chromdriver
    :return: text
    '''
    current_url = None
    try:
        current_url = driver.current_url
        body = driver.find_element(By.TAG_NAME, value='body').text
    except NoSuchElementException as e: # if no body tag, just get all text
        logger.warning(f'No body tag found (URL: {current_url}), using page_source instead. Error: {str(e)}')
        try:
            body = driver.page_source
        except (TimeoutException, WebDriverException) as e:
            logger.error(f'Failed to get page source (URL: {current_url}). Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
            body = ''
        except Exception as e:
            logger.error(f'Unexpected error getting page source (URL: {current_url}). Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
            body = ''
    except (TimeoutException, WebDriverException) as e:
        current_url = getattr(driver, 'current_url', 'unknown')
        logger.error(f'WebDriver error getting body text (URL: {current_url}). Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
        try:
            body = driver.page_source
        except Exception as fallback_e:
            logger.error(f'Fallback to page_source also failed (URL: {current_url}). Error: {str(fallback_e)}', exc_info=True)
            body = ''
    return body

def click_text(text, max_retries=1):
    '''
    click the text's region with retry mechanism
    :param text: text to click
    :param max_retries: maximum number of retry attempts
    :return: True if successful, False otherwise
    '''
    helium.Config.implicit_wait_secs = 2 # this is the implicit timeout for helium
    driver = helium.get_driver()
    driver.implicitly_wait(2) # this is the implicit timeout for selenium
    current_url = None
    
    try:
        current_url = driver.current_url
        body = get_page_text(driver)
    except Exception as e:
        logger.error(f'Failed to get page text before clicking "{text}" (URL: {current_url or "unknown"}). Error: {type(e).__name__}: {str(e)}', exc_info=True)
        return False
    
    for attempt in range(max_retries + 1):
        try:
            helium.highlight(text) # highlight text for debugging
            time.sleep(1)
            if re.search(text, body, flags=re.I):
                helium.click(text)
                time.sleep(2) # wait until website is completely loaded
                logger.debug(f'Successfully clicked text "{text}" (URL: {current_url})')
                return True
            else:
                logger.debug(f'Text "{text}" not found in page body (URL: {current_url})')
                return False
        except TimeoutException as e:
            if attempt < max_retries:
                logger.debug(f'Retry {attempt + 1}/{max_retries} for text "{text}" (URL: {current_url}). Timeout: {str(e)}')
                time.sleep(1)
                continue
            else:
                logger.warning(f'Timeout when clicking text "{text}" after {max_retries + 1} attempts (URL: {current_url}). Error: {str(e)}')
                return False
        except LookupError as e:
            logger.warning(f'Text "{text}" not found (URL: {current_url}). Error: {str(e)}')
            return False
        except WebDriverException as e:
            logger.error(f'WebDriver error clicking text "{text}" (URL: {current_url}). Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
            return False
        except Exception as e:
            logger.error(f'Unexpected error clicking text "{text}" (URL: {current_url}). Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
            return False
    return False

def click_point(x, y):
    '''
    click on coordinate (x,y)
    :param x:
    :param y:
    :return:
    '''
    helium.Config.implicit_wait_secs = 2 # this is the implicit timeout for helium
    helium.get_driver().implicitly_wait(2) # this the implicit timeout for selenium
    try:
        helium.click(helium.Point(x, y))
        time.sleep(2) # wait until website is completely loaded
        # click_popup()
    except TimeoutException as e:
        logger.warning(f'Timeout when clicking point ({x}, {y}): {str(e)}')
    except MoveTargetOutOfBoundsException as e:
        logger.warning(f'Point ({x}, {y}) is out of bounds: {str(e)}')
    except LookupError as e:
        logger.warning(f'Lookup error when clicking point ({x}, {y}): {str(e)}')
    except AttributeError as e:
        logger.error(f'Attribute error when clicking point ({x}, {y}): {str(e)}', exc_info=True)
    except Exception as e:
        logger.error(f'Error clicking point ({x}, {y}): {str(e)}', exc_info=True)

def visit_url(driver, orig_url, max_retries=2):
    '''
    Visit a URL with retry mechanism
    :param driver: chromedriver
    :param orig_url: URL to visit
    :param max_retries: maximum number of retry attempts
    :return: (success: bool, driver)
    '''
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry {attempt}/{max_retries} visiting URL: {orig_url}")
            else:
                logger.info(f"Attempting to visit URL: {orig_url}")
            
            driver.get(orig_url)
            logger.info(f"Successfully loaded URL: {orig_url}")
            time.sleep(2)
            
            # Try to dismiss alert if present
            try:
                logger.debug(f"Attempting to dismiss alert if present (URL: {orig_url})")
                driver.switch_to.alert.dismiss()
                logger.debug(f"Alert dismissed successfully (URL: {orig_url})")
            except NoSuchElementException:
                # No alert present, this is normal
                logger.debug(f"No alert present (URL: {orig_url})")
            except TimeoutException:
                # Alert timeout, continue
                logger.debug(f"Alert timeout (URL: {orig_url})")
            
            return True, driver
            
        except TimeoutException as e:
            if attempt < max_retries:
                logger.warning(f'Timeout when visiting URL {orig_url} (attempt {attempt + 1}/{max_retries + 1}). Retrying... Error: {str(e)}')
                time.sleep(2)
                continue
            else:
                logger.error(f'Timeout when visiting URL {orig_url} after {max_retries + 1} attempts. Error: {str(e)}', exc_info=True)
                return False, driver
        except WebDriverException as e:
            logger.error(f'WebDriver error visiting URL {orig_url}. Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
            return False, driver
        except Exception as e:
            logger.error(f'Unexpected error visiting URL {orig_url}. Error type: {type(e).__name__}, Message: {str(e)}', exc_info=True)
            return False, driver
    
    return False, driver


def driver_loader():

    options = initialize_chrome_settings()
    try:
        service = ChromeService(executable_path="./chromedriver/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        logger.info(f'Failed to load driver, trying to load driver.exe: {str(e)}', exc_info=True)
        try:
            service = ChromeService(executable_path="./chromedriver/chromedriver.exe")
            driver = webdriver.Chrome(service=service, options=options)
        except Exception as e:
            logger.error(f'Failed to load driver.exe: {str(e)}', exc_info=True)
            raise e
            
    driver.set_page_load_timeout(60)  # set timeout to avoid wasting time
    driver.set_script_timeout(60)  # set timeout to avoid wasting time
    helium.set_driver(driver)
    return driver

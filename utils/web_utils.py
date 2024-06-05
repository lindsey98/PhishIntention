from selenium.common.exceptions import NoSuchElementException, TimeoutException, MoveTargetOutOfBoundsException, StaleElementReferenceException
from seleniumwire import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager
import helium
import time
import re
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

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

    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920,1080')  # fix screenshot size
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    options.set_capability('unhandledPromptBehavior', 'dismiss')  # dismiss


    return options

def click_button(button_text):
    helium.Config.implicit_wait_secs = 2 # this is the implicit timeout for helium
    helium.get_driver().implicitly_wait(2)
    try:
        helium.click(helium.Button(button_text))
        return True
    except:
        return False

def get_page_text(driver):
    '''
    get body text from html
    :param driver: chromdriver
    :return: text
    '''
    try:
        body = driver.find_element(By.TAG_NAME, value='body').text
    except NoSuchElementException as e: # if no body tag, just get all text
        print(e)
        try:
            body = driver.page_source
        except Exception as e:
            print(e)
            body = ''
    return body

def click_text(text):
    '''
    click the text's region
    :param text:
    :return:
    '''
    helium.Config.implicit_wait_secs = 2 # this is the implicit timeout for helium
    helium.get_driver().implicitly_wait(2) # this is the implicit timeout for selenium
    body = get_page_text(helium.get_driver())
    try:
        helium.highlight(text) # highlight text for debugging
        time.sleep(1)
        if re.search(text, body, flags=re.I):
            helium.click(text)
            time.sleep(2) # wait until website is completely loaded
    except TimeoutException as e:
        print(e)
    except LookupError as e:
        print(e)
    except Exception as e:
        print(e)

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
        print(e)
    except MoveTargetOutOfBoundsException as e:
        print(e)
    except LookupError as e:
        print(e)
    except AttributeError as e:
        print(e)
    except Exception as e:
        print(e)

def visit_url(driver, orig_url):
    '''
    Visit a URL
    :param driver: chromedriver
    :param orig_url: URL to visit
    :param popup: click popup window or not
    :param sleep: need sleep time or not
    :return: load url successful or not
    '''
    try:
        driver.get(orig_url)
        time.sleep(2)
        driver.switch_to.alert.dismiss()
        return True, driver
    except TimeoutException as e:
        print(str(e))
        return False, driver
    except Exception as e:
        print(str(e))
        print("no alert")
        return True, driver


def driver_loader():
    '''
    load chrome driver
    '''

    seleniumwire_options = {
        'seleniumwire_options': {
            'enable_console_log': True,
            'log_level': 'DEBUG',
        }
    }

    options = initialize_chrome_settings()
    capabilities = DesiredCapabilities.CHROME
    capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
    capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert
    capabilities["pageLoadStrategy"] = "eager"  # eager mode #FIXME: set eager mode, may load partial webpage

    # driver = webdriver.Chrome(ChromeDriverManager().install())
    service = Service(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(options=options, service=service, seleniumwire_options=seleniumwire_options)
    driver.set_page_load_timeout(60)  # set timeout to avoid wasting time
    driver.set_script_timeout(60)  # set timeout to avoid wasting time
    helium.set_driver(driver)
    return driver



from selenium.common.exceptions import NoSuchElementException, TimeoutException, MoveTargetOutOfBoundsException, StaleElementReferenceException
import helium
import time
import requests
from seleniumwire import webdriver


def initialize_chrome_settings(lang_txt:str):
    '''
    initialize chrome settings
    :return:
    '''
    # enable translation
    white_lists = {}

    with open(lang_txt) as langf:
        for i in langf.readlines():
            i = i.strip()
            text = i.split(' ')
            print(text)
            white_lists[text[1]] = 'en'
    prefs = {
        "translate": {"enabled": "true"},
        "translate_whitelists": white_lists
    }

    options = webdriver.ChromeOptions()

    options.add_experimental_option("prefs", prefs)
    options.add_argument('--ignore-certificate-errors') # ignore errors
    options.add_argument('--ignore-ssl-errors')
    options.add_argument("--headless") # disable browser

    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920,1080') # fix screenshot size
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    options.set_capability('unhandledPromptBehavior', 'dismiss') # dismiss

    return  options

def vt_scan(url_test):
    retry = 0
    api_key = "2b93fae94a62662be089e9aa067e672ac242e3276b0f6a1e44e298b4858d4cf8"
    url = 'https://www.virustotal.com/vtapi/v2/url/report'

    params = {'apikey': api_key, 'resource': url_test, 'scan':1}
    response = requests.get(url, params=params).json()

    # This means the url wasnt in VT's database, preparing a new scan
    while("total" not in response and "positives" not in response and retry < 3):
        print("[*] " + str(retry) + " try. Maximum of 3 tries with 30 seconds interval...")
        # Intentionally sleeping for 30 seconds before coming back to retrieve results
        time.sleep(30)
        response = requests.get(url, params=params).json()
        retry +=1

    # Getting out of the loop means either tried >= 3 times, or successfully gotten result
    try:
        positive = response['positives']
        total = response['total']
    except KeyError:
        positive = None
        total = None

    return positive, total

def get_page_text(driver):
    '''
    get body text from html
    :param driver:
    :return:
    '''
    try:
        body = driver.find_element_by_tag_name('body').text
    except NoSuchElementException as e:
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
    try:
        helium.click(text)
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
    try:
        helium.click(helium.Point(x, y))
    except TimeoutException as e:
        print(e)
    except MoveTargetOutOfBoundsException as e:
        print(e)
    except LookupError as e:
        print(e)
    except AttributeError as e:
        print(e)
    except Exception as e:
        # print(x, y)
        print(e)



def clean_up_window(driver):
    '''
    Close chrome tab properly
    :param driver:
    :return:
    '''
    try:
        current_window = driver.current_window_handle
        for i in driver.window_handles:
            if i != current_window:
                driver.switch_to_window(i)
                driver.close()
    except Exception as e: # unknown exception occurs
        pass


def writetxt(txtpath, contents):
    '''
    write into txt file with encoding utf-8
    :param txtpath:
    :param contents:
    :return:
    '''
    with open(txtpath, 'w', encoding='utf-8') as fw:
        fw.write(contents)


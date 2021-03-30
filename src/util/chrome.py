import helium
from seleniumwire import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager

def get_page_text(driver):
    '''
    get body text from html
    :param driver:
    :return:
    '''
    return driver.find_element_by_tag_name('body').text

def click_text(text):
    '''
    click the text's region
    :param text:
    :return:
    '''
    try:
        helium.click(text)
    except LookupError as e:
        print(e)

def click_point(x, y):
    '''
    click on coordinate (x,y)
    :param x:
    :param y:
    :return:
    '''
    helium.click(helium.Point(x, y))

# write in txt
def writetxt(txtpath, contents):
    with open(txtpath, 'w', encoding='utf-8') as fw:
        fw.write(contents)

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
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.add_argument("--headless") # diable browser

    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920,1080') # screenshot size
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    options.set_capability('unhandledPromptBehavior', 'dismiss')

    return  options



# load driver ONCE
options = initialize_chrome_settings(lang_txt='src/util/lang.txt')
capabilities = DesiredCapabilities.CHROME
capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert

driver = webdriver.Chrome(ChromeDriverManager().install(), desired_capabilities=capabilities,
                          chrome_options=options)
helium.set_driver(driver)

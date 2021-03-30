import helium
from seleniumwire import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
from webdriver_manager.chrome import ChromeDriverManager

white_lists = {}


def get_page_text(driver):
    return driver.find_element_by_tag_name('body').text

def click_text(text):
    helium.click(text)

if __name__ == '__main__':
    with open('src/util/lang.txt') as langf:
        for i in langf.readlines():
            i = i.strip()
            text = i.split(' ')
            print(text)
            white_lists[text[1]] = 'en'
    print(white_lists)
    prefs = {
        "translate": {"enabled": "true"},

        "translate_whitelists": white_lists
    }

    options = webdriver.ChromeOptions()
    # options.add_argument("--start-maximized")
    capabilities = DesiredCapabilities.CHROME
    # capabilities["loggingPrefs"] = {"performance": "ALL"}  # chromedriver < ~75
    capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
    # options.add_experimental_option("excludeSwitches", ["disable-popup-blocking"])

    options.add_experimental_option("prefs", prefs)
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')

    options.add_argument("--start-maximized")
    # options.add_argument('--no-sandbox')
    #   options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')


    driver = webdriver.Chrome(ChromeDriverManager().install(), desired_capabilities=capabilities,
                                      chrome_options=options)
    helium.set_driver(driver)

    driver.get("https://www.google.com.sg")
    print("getting url")

    page_text = get_page_text(driver)
    print(page_text)
    page_text = page_text.split('\n')
    for i in page_text:
        if 'sign' in i.lower():
            print("found")
            click_text(i)

    time.sleep(1)
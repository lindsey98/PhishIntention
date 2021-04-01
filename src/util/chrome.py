from selenium.common.exceptions import NoSuchElementException, TimeoutException, MoveTargetOutOfBoundsException, StaleElementReferenceException
import helium
import time
import requests

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


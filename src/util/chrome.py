from selenium.common.exceptions import NoSuchElementException, TimeoutException, MoveTargetOutOfBoundsException, StaleElementReferenceException
import helium

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
        body = driver.page_source
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
        print(x, y)
        print(e)



def clean_up_window(driver):
    '''
    Close chrome tab properly
    :param driver:
    :return:
    '''
    current_window = driver.current_window_handle
    for i in driver.window_handles:
        if i != current_window:
            driver.switch_to_window(i)
            driver.close()

# write in txt
def writetxt(txtpath, contents):
    with open(txtpath, 'w', encoding='utf-8') as fw:
        fw.write(contents)


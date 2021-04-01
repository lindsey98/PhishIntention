from lxml import html
import io
import os
import numpy as np

def read_html(html_path):
    '''
    read html and parse into tree
    :param html_path: path to html.txt
    '''
    done = False
    tree_list = None
    
    # check if html path exist
    if not os.path.exists(html_path):
        print('Path not exists: {}'.format(html_path))
        return tree_list
    
    # parse html text
    try:
        with io.open(html_path,'r',encoding='ISO-8859-1') as f:
            page = f.read()
            tree = html.fromstring(page)
            tree_list = tree
            done = True
    except Exception as e:
        pass

    # try another encoding
    if not done:
        try:
            with open(html_path, 'r', encoding="utf-8") as f:
                page = f.read()
                tree = html.fromstring(page)
                tree_list = tree
                done = True
        except Exception as e:
            pass

    # try another encoding
    if not done:
        try:
            with open(html_path,'r',encoding='ANSI') as f:
                page = f.read()
                tree = html.fromstring(page)
                tree_list = tree
                done = True
        except Exception as e:
            pass
    

    return tree_list

def proc_tree(tree):
    '''
    returns number of forms, type of forms in a list, number of inputs in each form, number of password field in each form
    :param tree: Element html object
    '''
    
    if tree is None: # parsing into tree failed
        return 0, [], [], [], []
    forms = tree.xpath('.//form') # find form
    if len(forms) == 0 : # no form
        return 0, [], [], [], []
    else:
        methods  = []
        count_inputs = []
        count_password = []
        count_username = []
        
        for form in forms:
            count = 0
            methods.append(form.get('method')) # get method of form "post"/"get"
            
            inputs = form.xpath('.//input')
            count_inputs.append(len(inputs)) # get number if inputs
            inputs = form.xpath('.//input[@type="password"]') # get number of password fields
            count_password.append(len(inputs))
            
            usernames = form.xpath('.//input[@type="username"]') # get number of username fields
            count_username.append(len(usernames))
            
        return len(forms), methods, count_inputs, count_password, count_username
            
        
def check_post(x, version=1):
    
    '''
    check whether html contains postform/user name input field/ password input field
    :param x: Tuple object (len(forms):int, methods:List[str|float], count_inputs:List[int], count_password:List[int], count_username:List[int])
    :return:
    '''

    num_form, methods, num_inputs, num_password, num_username = x
#     print(num_password, num_username)
    
    if len(methods) == 0:
        have_postform = 0
    else:
        have_postform = (len([y for y in [x for x in methods if x is not None] if y.lower() == 'post']) > 0)

    if len(num_password) == 0:
        have_password = 0
    else:
        have_password = (np.sum(num_password) > 0)

    if len(num_username) == 0:
        have_username = 0
    else:
        have_username = (np.sum(num_username) > 0)

    # CRP = 0, nonCRP = 1
    if version == 1:
        return 0 if (have_postform) else 1
    elif version == 2:
        return 0 if (have_password | have_username) else 1
    elif version == 3:
        return 0 if (have_postform | have_password | have_username) else 1



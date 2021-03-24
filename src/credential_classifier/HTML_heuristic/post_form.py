from lxml import html
import io
import os


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
            with open(html_path,'r',encoding='ANSI') as f:
                page = f.read()
                tree = html.fromstring(page)
                tree_list = tree
                done = True
        except Exception as e:
            pass
    
    # try another encoding
    if not done:
        try:
            with open(html_path, 'r', encoding="UTF-8") as f:
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
        return 0, [], [],[]
    forms = tree.xpath('.//form') # find form
    if len(forms) == 0 : # no form
        return 0, [], [],[] 
    else:
        methods  = []
        count_inputs = []
        count_password = []
        for form in forms:
            count = 0
            methods.append(form.get('method')) # get method of form "post"/"get"
            inputs = form.xpath('.//input')
            count_inputs.append(len(inputs)) # get number if inputs
            inputs = form.xpath('.//input[@type="password"]') # get number of password fields
            count_password.append(len(inputs))
        return len(forms), methods, count_inputs, count_password
            
        
def check_post(x):
    
    '''
    check whether html contains postform
    :param x: Tuple object (len(forms), methods, count_inputs, count_password)
    '''
    if x[0] == 0: # if no form
        return 1 # nonCRP?
    for i in x[1]: # method fields
        if i is None:
            continue
        if i.lower() =='post': # check if any method is post
            return 0 # CRP?
    return 1 # nonCRP?



# if __name__ == '__main__':
#     proc_data = [] 
#     for html_f, tree in tree_list:
#         proc_data.append((html_f,proc_tree(tree)))   

#     len(list(filter(lambda x: not check_post(x), proc_data)))






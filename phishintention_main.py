# from seleniumwire import webdriver
from phishintention_config import *
import os
import argparse
from gsheets import gwrapper
# from src.utils import *
from src.element_detector import vis
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
# from login_cv import WebTester
# from login_cv import test_wrapper
import time

def main(url, screenshot_path):
    '''
    Get phishing prediction 
    params url: url
    params screenshot_path: path to screenshot
    returns phish_category: 0 for benign, 1 for phish
    returns phish_target: None/brand name
    '''
    
    waive_crp_classifier = False
    siamese_conf = None
    
    while True:
        # 0 for benign, 1 for phish, default is benign
        phish_category = 0
        pred_target = None
        print("entering phishpedia")

        ####################### Step1: element detector ##############################################
        pred_classes, pred_boxes, pred_scores = element_recognition(img=screenshot_path, model=ele_model)
        plotvis = vis(screenshot_path, pred_boxes, pred_classes)
        print("plot")
        # If no element is reported
        if len(pred_boxes) == 0:
            print('No element is detected, report as benign')
            return phish_category, pred_target, plotvis, siamese_conf
        print('entering siamese')

        ######################## Step2: Siamese (logo matcher) ########################################
        pred_target, matched_coord, siamese_conf = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                        domain_map_path=domain_map_path,
                                        model=pedia_model, 
                                        logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                        url=url,
                                        shot_path=screenshot_path,
                                        ts=siamese_ts) 

        if pred_target is None:
            print('Did not match to any brand, report as benign')
            return phish_category, pred_target, plotvis, siamese_conf
        print(pred_target)

        ######################## Step3: CRP checker (if a target is reported) #################################
        if waive_crp_classifier: # only run dynamic analysis ONCE
            break
            
        if pred_target is not None:
            # dir = os.path.dirname(screenshot_path)
            # CRP classifier + heuristic
            # test_wrapper(url, dir, webTester)
            # break

            # CRP HTML heuristic
            html_path = screenshot_path.replace("shot.png", "html.txt")
            cre_pred = html_heuristic(html_path)
            if cre_pred == 1: # if HTML heuristic report as nonCRP
                # CRP classifier
                cre_pred, cred_conf, _  = credential_classifier_mixed_al(img=screenshot_path, coords=pred_boxes,
                                                                         types=pred_classes, model=cls_model)
#
#           ######################## Step4: Dynamic analysis #################################
            if cre_pred == 1:
                print('Non-CRP, enter dynamic analysis')
#
                # update url and screenshot path
                url, screenshot_path, successful = dynamic_analysis(url=url, screenshot_path=screenshot_path,
                                                                    cls_model=cls_model, ele_model=ele_model, login_model=login_model,
                                                                    driver=driver)
#
                waive_crp_classifier = True # only run dynamic analysis ONCE

                if successful == False:
                    print('Dynamic analysis cannot find any link redirected to a CRP page, report as benign')
                    return phish_category, None, plotvis, None

            else: # already a CRP page
                print('Already a CRP, continue')
                break
#
    ######################## Step5: Return #################################
    if pred_target is not None:
        phish_category = 1
        # Visualize
        cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                    (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
    return phish_category, pred_target, plotvis, siamese_conf



if __name__ == "__main__":

    # white_lists = {}
    #
    # with open('src/util/lang.txt') as langf:
    #     for i in langf.readlines():
    #         i = i.strip()
    #         text = i.split(' ')
    #         print(text)
    #         white_lists[text[1]] = 'en'
    # print(white_lists)
    # prefs = {
    #     "translate": {"enabled": "true"},
    #
    #     "translate_whitelists": white_lists
    # }
    #
    # base_save = 'latest_model/alexa2'
    # if not os.path.exists(base_save):
    #     os.mkdir(base_save)
    # options = webdriver.ChromeOptions()
    # # options.add_argument("--start-maximized")
    # capabilities = DesiredCapabilities.CHROME
    # # capabilities["loggingPrefs"] = {"performance": "ALL"}  # chromedriver < ~75
    # capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
    # # options.add_experimental_option("excludeSwitches", ["disable-popup-blocking"])
    #
    # options.add_experimental_option("prefs", prefs)
    # options.add_argument('--ignore-certificate-errors')
    # options.add_argument('--ignore-ssl-errors')
    #
    # options.add_argument("--start-maximized")
    # # options.add_argument('--no-sandbox')
    # #   options.add_argument('--disable-dev-shm-usage')
    # options.add_argument('--window-size=1920,1080')
    # options.add_argument("--disable-blink-features=AutomationControlled")
    # # options.add_experimental_option("excludeSwitches", ["enable-automation"])
    # options.add_experimental_option('useAutomationExtension', False)
    # options.add_argument(
    #     'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    # webTester = WebTester(options, capabilities)

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", help='Input folder path to parse', required=True)
    parser.add_argument('-r', "--results", help='Input results file name', required=True)
    args = parser.parse_args()
    date = args.folder.split('/')[-1]    
    directory = args.folder 
    results_path = args.results

    if not os.path.exists(args.results):
        with open(args.results, "w+") as f:
            f.write("url" +"\t")
            f.write("phish" +"\t")
            f.write("prediction" + "\t") # write top1 prediction only
            f.write("vt_result" +"\n")


    done = []
    # while True:
    for item in os.listdir(directory):
        if item in done:
            continue

        try:
            print(item)
            full_path = os.path.join(directory, item)

            screenshot_path = os.path.join(full_path, "shot.png")
            url = open(os.path.join(full_path, 'info.txt'), encoding='utf-8').read()

            if not os.path.exists(screenshot_path):
                continue

            else:
                phish_category, phish_target, plotvis, siamese_conf = main(url=url, screenshot_path=screenshot_path)

                vt_result = "None"
                if phish_target is not None:
                    try:
                        if vt_scan(url) is not None:
                            positive, total = vt_scan(url)
                            print("Positive VT scan!")
                            vt_result = str(positive) + "/" + str(total)
                        else:
                            print("Negative VT scan!")
                            vt_result = "None"

                    except Exception as e:
                        print('VTScan is not working...')
                        vt_result = "error"

                with open(args.results, "a+") as f:
                    f.write(url +"\t")
                    f.write(str(phish_category) +"\t")
                    f.write(str(phish_target) + "\t") # write top1 prediction only
                    f.write(str(siamese_conf) + "\t")
                    f.write(vt_result +"\n")

                cv2.imwrite(os.path.join(full_path, "predict.png"), plotvis)

        except Exception as e:
            print(str(e))
      #  raise(e)
    time.sleep(15)

    driver.quit()


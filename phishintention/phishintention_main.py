from phishintention.phishintention_config import *
import os
import argparse
from phishintention.src.AWL_detector import vis
import time
# import os
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#####################################################################################################################
# ** Step 1: Enter Layout detector, get predicted elements
# ** Step 2: Enter Siamese, siamese match a phishing target, get phishing target

# **         If Siamese report no target, Return Benign, None
# **         Else Siamese report a target, Enter CRP classifier(and HTML heuristic)

# ** Step 3: If CRP classifier(and heuristic) report it is non-CRP, go to step 4: Dynamic analysis, go back to step1
# **         Else CRP classifier(and heuristic) reports its a CRP page

# ** Step 5: If reach a CRP + Siamese report target: Return Phish, Phishing target
# ** Else: Return Benign
#####################################################################################################################


def test(url, screenshot_path, AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH):
    '''
    Phish-discovery main script
    :param url: URL
    :param screenshot_path: path to screenshot
    :return phish_category: 0 for benign, 1 for phish
    :return phish_target: None/brand name
    :return plotvis: predicted image
    :return siamese_conf: matching confidence reported by siamese
    :return dynamic: go through dynamic analysis or not
    :return time breakdown
    '''
    
    waive_crp_classifier = False
    dynamic = False
    ele_detector_time = 0
    siamese_time = 0
    crp_time = 0
    dynamic_time = 0
    process_time = 0


    while True:
        # 0 for benign, 1 for phish, default is benign
        phish_category = 0
        pred_target = None
        siamese_conf = None
        print("Entering phishintention")

        ####################### Step1: layout detector ##############################################
        start_time = time.time()
        pred_classes, pred_boxes, pred_scores = element_recognition(img=screenshot_path, model=AWL_MODEL)
        ele_detector_time = time.time() - start_time
        plotvis = vis(screenshot_path, pred_boxes, pred_classes)
        print("plot")

        # If no element is reported
        if pred_boxes is None or len(pred_boxes) == 0:
            print('No element is detected, report as benign')
            return phish_category, pred_target, plotvis, siamese_conf, dynamic, str(ele_detector_time)+'|'+str(siamese_time)+'|'+str(crp_time)+'|'+str(dynamic_time)+'|'+str(process_time), pred_boxes, pred_classes
        print('Entering siamese')

        # domain already in targetlist
        query_domain = tldextract.extract(url).domain
        with open(DOMAIN_MAP_PATH, 'rb') as handle:
            domain_map = pickle.load(handle)
        existing_brands = domain_map.keys()
        existing_domains = [y for x in list(domain_map.values()) for y in x]
        if query_domain in existing_brands or query_domain in existing_domains:
            return phish_category, pred_target, plotvis, siamese_conf, dynamic, str(ele_detector_time)+'|'+str(siamese_time)+'|'+str(crp_time)+'|'+str(dynamic_time)+'|'+str(process_time), pred_boxes, pred_classes

        ######################## Step2: Siamese (logo matcher) ########################################
        start_time = time.time()
        pred_target, matched_coord, siamese_conf = phishpedia_classifier_OCR(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                        domain_map_path=DOMAIN_MAP_PATH, model=SIAMESE_MODEL,
                                        ocr_model = OCR_MODEL,
                                        logo_feat_list=LOGO_FEATS, file_name_list=LOGO_FILES,
                                        url=url, shot_path=screenshot_path,
                                        ts=SIAMESE_THRE)
        siamese_time = time.time() - start_time

        if pred_target is None:
            print('Did not match to any brand, report as benign')
            return phish_category, pred_target, plotvis, siamese_conf, dynamic, \
                   str(ele_detector_time)+'|'+str(siamese_time)+'|'+str(crp_time)+'|'+str(dynamic_time)+'|'+str(process_time),\
                   pred_boxes, pred_classes

        ######################## Step3: CRP checker (if a target is reported) #################################
        print('A target is reported by siamese, enter CRP classifier')
        if waive_crp_classifier: # only run dynamic analysis ONCE
            break
            
        if pred_target is not None:
            # CRP HTML heuristic
            html_path = screenshot_path.replace("shot.png", "html.txt")
            start_time = time.time()
            cre_pred = html_heuristic(html_path)
            if cre_pred == 1: # if HTML heuristic report as nonCRP
                # CRP classifier
                cre_pred, cred_conf, _  = credential_classifier_mixed_al(img=screenshot_path, coords=pred_boxes,
                                                                         types=pred_classes, model=CRP_CLASSIFIER)
            crp_time = time.time() - start_time
#
#           ######################## Step4: Dynamic analysis #################################
            if cre_pred == 1:
                print('It is a Non-CRP page, enter dynamic analysis')
                # update url and screenshot path
                # # load driver ONCE!
                driver = driver_loader()
                print('Finish loading webdriver')
                # load chromedriver
                start_time = time.time()
                url, screenshot_path, successful, process_time = dynamic_analysis(url=url, screenshot_path=screenshot_path,
                                                                    cls_model=CRP_CLASSIFIER, ele_model=AWL_MODEL, login_model=CRP_LOCATOR_MODEL,
                                                                    driver=driver)
                dynamic_time = time.time() - start_time
                driver.quit()

                waive_crp_classifier = True # only run dynamic analysis ONCE

                # If dynamic analysis did not reach a CRP
                if successful == False:
                    print('Dynamic analysis cannot find any link redirected to a CRP page, report as benign')
                    return phish_category, None, plotvis, None, dynamic, str(ele_detector_time)+'|'+str(siamese_time)+'|'+str(crp_time)+'|'+str(dynamic_time)+'|'+str(process_time), pred_boxes, pred_classes
                else: # dynamic analysis successfully found a CRP
                    dynamic = True
                    print('Dynamic analysis found a CRP, go back to layout detector')

            else: # already a CRP page
                print('Already a CRP, continue')
                break
#
    ######################## Step5: Return #################################
    if pred_target is not None:
        print('Phishing is found!')
        phish_category = 1
        # Visualize, add annotations
        cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                    (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
    return phish_category, pred_target, plotvis, siamese_conf, dynamic, \
           str(ele_detector_time)+'|'+str(siamese_time)+'|'+str(crp_time)+'|'+str(dynamic_time)+'|'+str(process_time), \
            pred_boxes, pred_classes

def runit(folder, results, AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH):
    date = folder.split('/')[-1]
    directory = folder
    results_path = results

    if not os.path.exists(results):
        with open(results, "w+") as f:
            f.write("folder" + "\t")
            f.write("url" +"\t")
            f.write("phish" +"\t")
            f.write("prediction" + "\t") # write top1 prediction only
            f.write("siamese_conf" + "\t")
            f.write("vt_result" +"\t")
            f.write("dynamic" + "\t")
            f.write("runtime (layout detector|siamese|crp classifier|login finder total|login finder process)" + "\t")
            f.write("total_runtime" + "\n")


    for item in tqdm(os.listdir(directory)):

        if item in [x.split('\t')[0] for x in open(results, encoding='ISO-8859-1').readlines()]:
            continue # have been predicted

        # try:
        print(item)
        full_path = os.path.join(directory, item)
        if item == '' or not os.path.exists(full_path):  # screenshot not exist
            continue
        screenshot_path = os.path.join(full_path, "shot.png")
        info_path = os.path.join(full_path, 'info.txt')
        if not os.path.exists(screenshot_path):  # screenshot not exist
            continue
        try:
            url = open(info_path, encoding='ISO-8859-1').read()
        except:
            url = 'https://www' + item


        start_time = time.time()
        phish_category, phish_target, plotvis, siamese_conf, dynamic, time_breakdown, _, _ = test(url=url, screenshot_path=screenshot_path,
                                                                                            AWL_MODEL=AWL_MODEL, CRP_CLASSIFIER=CRP_CLASSIFIER, CRP_LOCATOR_MODEL=CRP_LOCATOR_MODEL,
                                                                                            SIAMESE_MODEL=SIAMESE_MODEL, OCR_MODEL=OCR_MODEL,
                                                                                            SIAMESE_THRE=SIAMESE_THRE, LOGO_FEATS=LOGO_FEATS, LOGO_FILES=LOGO_FILES,
                                                                                            DOMAIN_MAP_PATH=DOMAIN_MAP_PATH)
        end_time = time.time()

        # FIXME: call VTScan only when phishintention report it as positive
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

        # write results as well as predicted image
        try:
            with open(results, "a+", encoding='ISO-8859-1') as f:
                f.write(item + "\t")
                f.write(url +"\t")
                f.write(str(phish_category) +"\t")
                f.write(str(phish_target) + "\t") # write top1 prediction only
                f.write(str(siamese_conf) + "\t")
                f.write(vt_result +"\t")
                f.write(str(dynamic) + "\t")
                f.write(time_breakdown + "\t")
                f.write(str(end_time - start_time) + "\n")

            if plotvis is not None:
                cv2.imwrite(os.path.join(full_path, "predict_intention.png"), plotvis)
        except UnicodeEncodeError:
            continue


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", help='Input folder path to parse',  default='./datasets/test_sites')
    parser.add_argument('-r', "--results", help='Input results file name', default='./test_intention.txt')
    parser.add_argument('-c', "--config", help='Config file path', default=None)
    parser.add_argument('-d', '--device', help='device', choices=['cuda', 'cpu'], required=True)
    args = parser.parse_args()

    AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(args.config,
                                                                                                                                                device=args.device)
    runit(args.folder, args.results, AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)








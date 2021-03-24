
from phishintention_config import *
import os
import argparse
# from gsheets import gwrapper
# from src.utils import *
from src.element_detector import vis

def main(url, screenshot_path):
    
    '''
    Get phishing prediction 
    params url: url
    params screenshot_path: path to screenshot
    returns phish_category: 0 for benign, 1 for phish
    returns phish_target: None/brand name
    '''
    
    waive_crp_classifier = False
    
    while True:
        # 0 for benign, 1 for phish, default is benign
        phish_category = 0 

        ####################### Step1: element detector ##############################################
        pred_classes, pred_boxes, pred_scores = element_recognition(img=screenshot_path, model=ele_model)
        plotvis = vis(screenshot_path, pred_boxes, pred_classes)
        
        # If no element is reported
        if len(pred_boxes) == 0:
            print('No element is detected, report as benign')
            return phish_category, None, plotvis
        
        ######################## Step2: siamese (logo matcher) ########################################
        pred_target, matched_coord = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                        domain_map_path=domain_map_path,
                                        model=pedia_model, 
                                        logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                        url=url,
                                        shot_path=screenshot_path,
                                        ts=siamese_ts) 

        if pred_target is None:
            print('Did not match to any brand, report as benign')
            return phish_category, None, plotvis

        ######################## Step3: a target is reported, check CRP #################################
        if waive_crp_classifier: # only run dynamic analysis once
            break
            
        if pred_target is not None:
            # CRP HTML heuristic
            cre_pred = html_heuristic(html_path)
            if cre_pred == 1: # if HTML heuristic report as nonCRP
                # CRP classifier
                cre_pred, cred_conf, _  = credential_classifier_mixed_al(img=screenshot_path, coords=pred_boxes, 
                                                                 types=pred_classes, model=cls_model)
            
            if cre_pred == 1: # non-CRP page
                print('Non-CRP, enter dynamic analysis')
                
                ###### Dynamic analysis here ##############
                # update url and screenshot path
                url, screenshot_path, successful = dynamic_analysis(url, screenshot_path, cls_model)
                ###########################################
                
                waive_crp_classifier = True # only run dynamic analysis once
                
                if successful == False:
                    print('Dynamic analysis cannot find any link redirected to a CRP page, report as benign')
                    return phish_category, None, plotvis
#                 return phish_category, None, plotvis #TODO: remove this
                
            else: # already a CRP page
                print('Already a CRP, continue')
                break
        
    if pred_target is not None:
        phish_category = 1
        # Visualize
        cv2.putText(plotvis, "Target: %s" % pred_target, (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
    return phish_category, pred_target, plotvis



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", help='Input folder path to parse', required=True)
    parser.add_argument('-r', "--results", help='Input results file name', required=True)
    args = parser.parse_args()
    date = args.folder.split('/')[-1]    
    directory = args.folder 
    results_path = args.results
    with open(args.results, "w+") as f:
        f.write("url" +"\t")
        f.write("phish" +"\t")
        f.write("prediction" + "\t") # write top1 prediction only
        f.write("vt_result" +"\n")
    

    for item in os.listdir(directory):
        try:
            print(item)
            full_path = os.path.join(directory, item)
            
            screenshot_path = os.path.join(full_path, "shot.png")
            url = open(os.path.join(full_path, 'info.txt')).read()
            
            if not os.path.exists(screenshot_path):
                continue
            else:
                phish_category, phish_target, plotvis = main(url=url, screenshot_path=screenshot_path)
                
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
                    f.write(vt_result +"\n")
                    
                cv2.imwrite(os.path.join(full_path, "predict.png"), plotvis)
                
        except Exception as e:
            print(str(e))
          #  raise(e)
        
#     sheets = gwrapper()
#     sheets.update_file(results_path, date) 


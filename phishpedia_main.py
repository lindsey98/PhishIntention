from phishintention_config import *
import os
import argparse
from src.element_detector import vis
import time


#####################################################################################################################
# ** Step 1: Enter Layout detector, get predicted elements
# ** Step 2: Enter Siamese, siamese match a phishing target, get phishing target

# **         If Siamese report no target, Return Benign, None
# **         Else Siamese report a target, Return Phish, phishing target
#####################################################################################################################


def main(url, screenshot_path):
    '''
    Get phishing prediction
    params url: url
    params screenshot_path: path to screenshot
    returns phish_category: 0 for benign, 1 for phish
    returns phish_target: None/brand name
    '''

    # 0 for benign, 1 for phish, default is benign
    phish_category = 0
    pred_target = None
    siamese_conf = None
    print("Entering phishpedia")

    ####################### Step1: layout detector ##############################################
    pred_classes, pred_boxes, pred_scores = element_recognition(img=screenshot_path, model=ele_model)
    plotvis = vis(screenshot_path, pred_boxes, pred_classes)
    print("plot")
    # If no element is reported
    if len(pred_boxes) == 0:
        print('No element is detected, report as benign')
        return phish_category, pred_target, plotvis, siamese_conf
    print('Entering siamese')

    ######################## Step2: Siamese (logo matcher) ########################################
    pred_target, matched_coord, siamese_conf = phishpedia_classifier(pred_classes=pred_classes,
                                                                     pred_boxes=pred_boxes,
                                                                     domain_map_path=domain_map_path,
                                                                     model=pedia_model,
                                                                     logo_feat_list=logo_feat_list,
                                                                     file_name_list=file_name_list,
                                                                     url=url,
                                                                     shot_path=screenshot_path,
                                                                     ts=siamese_ts)

    if pred_target is None:
        print('Did not match to any brand, report as benign')
        return phish_category, pred_target, plotvis, siamese_conf

    else:
        phish_category = 1
        # Visualize, add annotations
        cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                    (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return phish_category, pred_target, plotvis, siamese_conf


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", help='Input folder path to parse', required=True)
    parser.add_argument('-r', "--results", help='Input results file name', required=True)
    args = parser.parse_args()
    date = args.folder.split('/')[-1]
    directory = args.folder
    results_path = args.results.split('.txt')[0] + "_pedia.txt"

    if not os.path.exists(results_path):
        with open(results_path, "w+") as f:
            f.write("folder" + "\t")
            f.write("url" + "\t")
            f.write("phish" + "\t")
            f.write("prediction" + "\t")  # write top1 prediction only
            f.write("siamese_conf" + "\t")
            f.write("vt_result" + "\t")
            f.write("runtime" + "\n")

    for item in tqdm(os.listdir(directory)):
        start_time = time.time()

        if item in open(results_path).read():
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

                with open(results_path, "a+") as f:
                    f.write(item + "\t")
                    f.write(url + "\t")
                    f.write(str(phish_category) + "\t")
                    f.write(str(phish_target) + "\t")  # write top1 prediction only
                    f.write(str(siamese_conf) + "\t")
                    f.write(vt_result + "\t")
                    f.write(str(round(time.time() - start_time, 4)) + "\n")

                cv2.imwrite(os.path.join(full_path, "predict.png"), plotvis)

        except Exception as e:
            print(str(e))

    #  raise(e)
    time.sleep(2)




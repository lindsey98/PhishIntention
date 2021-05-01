import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
import shutil

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('./datasets/gsheetdownload.json', scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
sheet = client.open("test").sheet1

# Extract and print all of the values
list_of_hashes = sheet.get_all_records()
df = pd.DataFrame(list_of_hashes)

# Get breakdown statistics
pedia_tp = []
pedia_fp = []
pedia_unsure = []

intention_tp = []
intention_miss = []
intention_fp = []
intention_unsure = []

for index, row in df.iterrows():
    path = row['date'] + '/' + row['foldername']
    if row['yes'] > 0:
        # phishpedia get it correct
        pedia_tp.append(path)
        if os.path.exists(os.path.join('./datasets/PhishDiscovery/PhishIntention', path)): # intention also get it correct
            intention_tp.append(path)
        else: # intention miss it
            intention_miss.append(path)
    elif row['no'] > 0:
        # mistaken
        pedia_fp.append(path)
        if os.path.exists(os.path.join('./datasets/PhishDiscovery/PhishIntention', path)): # intention also has this FP
            intention_fp.append(path)
        else:
            pass # intention suppress this fp
    elif row['unsure'] > 0:
        # unsure, ignore
        print('ignore')
        pedia_unsure.append(path)
        if os.path.exists(os.path.join('./datasets/PhishDiscovery/PhishIntention', path)): # intention also has this FP
            intention_unsure.append(path)
    else:
        print('Not labelled yet {}'.format(path))

# print out
print('Pedia TP = {}, FP = {}, Unsure = {}'.format(str(len(pedia_tp)), str(len(pedia_fp)), str(len(pedia_unsure))))
print('Intention TP = {}, FP = {}, Miss(FN) = {}, Unsure = {}'.format(str(len(intention_tp)), str(len(intention_fp)), str(len(intention_miss)), str(len(intention_unsure))))

# try:
#     shutil.rmtree('./datasets/PhishDiscovery/phishintention_miss')
# except Exception as e:
#     print(e)
#     pass
# os.makedirs('./datasets/PhishDiscovery/phishintention_miss', exist_ok=True)
# for folder in intention_miss:
#     try:
#         shutil.copytree(os.path.join('./datasets/PhishDiscovery/Phishpedia/', folder),
#                     os.path.join('./datasets/PhishDiscovery/phishintention_miss/', folder.split('/')[1]))
#     except Exception as e:
#         print(e)
#
# try:
#     shutil.rmtree('./datasets/PhishDiscovery/phishintention_fp')
# except Exception as e:
#     print(e)
#     pass
# os.makedirs('./datasets/PhishDiscovery/phishintention_fp', exist_ok=True)
# for folder in intention_fp:
#     try:
#         shutil.copytree(os.path.join('./datasets/PhishDiscovery/Phishpedia/', folder),
#                     os.path.join('./datasets/PhishDiscovery/phishintention_fp/', folder.split('/')[1]))
#     except Exception as e:
#         print(e)


# from src.credential import html_heuristic, credential_classifier_mixed_al, credential_config
# from src.element_detector import element_recognition, element_config, vis
# import cv2
#
# # element recognition model
# ele_cfg, ele_model = element_config(rcnn_weights_path = './src/element_detector/output/website_lr0.001/model_final.pth',
#                                     rcnn_cfg_path='./src/element_detector/configs/faster_rcnn_web.yaml')
# # # # CRP classifier -- mixed version
# cls_model = credential_config(checkpoint='./src/credential_classifier/output/hybrid/hybrid_lr0.005/BiT-M-R50x1V2_0.005.pth.tar',
#                               model_type='mixed')
#
# r_element = 0
# element_folder = []
#
# for folder in intention_miss:
#     screenshot_path = 'D:\\ruofan\\PhishIntention\\datasets\\PhishDiscovery\\phishintention_miss\\{}\\shot.png'.format(folder.split('/')[-1]) # secure.terratopmail.xyz, facebookadcenter.acierbuildingtech.com
#     html_path = screenshot_path.replace("shot.png", "html.txt")
#     url = open(screenshot_path.replace("shot.png", "info.txt")).read()
#     cre_pred = html_heuristic(html_path)
#     print('HTML heuristic:', cre_pred)
#     pred_classes, pred_boxes, pred_scores = element_recognition(img=screenshot_path, model=ele_model)
#     plotvis = vis(screenshot_path, pred_boxes, pred_classes)
#     cv2.imwrite('debug.png', plotvis)
#
#     if cre_pred == 1:  # if HTML heuristic report as nonCRP
#         # CRP classifier
#         cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=screenshot_path, coords=pred_boxes,
#                                                                 types=pred_classes, model=cls_model)
#         print(cre_pred, cred_conf)
#
#     if cre_pred == 0:
#         r_element += 1
#         element_folder.append(folder.split('/')[-1])
#
# print(r_element)


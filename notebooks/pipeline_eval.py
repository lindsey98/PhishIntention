import os
os.chdir('..')
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # use all devices

from tqdm import tqdm
import time

from src.layout import *
from src.siamese import *
from src.element_detector import *
from src.credential import *

def evaluate_html_classifier(datadir, mode, write_txt, classifier_type):

    assert mode in ['phish', 'benign', 'discovery']
    assert classifier_type in ['layout', 'screenshot', 'mixed']
    
    for folder in tqdm(os.listdir(datadir)):

        phish_category = 0 # 0 for benign, 1 for phish
        pred_target = None # predicted target, default is None

        img_path = os.path.join(datadir, folder, 'shot.png')
        html_path = os.path.join(datadir, folder, 'html.txt')
        if mode != 'phish':
            url = open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read()
        else:
            url = eval(open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read())['url']

        # Element recognition module
        start_time = time.time()
        pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0 # Report as benign

        # If at least one element is reported
        else:
            # Credential heuristic module
            start_time = time.time()
            cred_conf = None
            # CRP HTML heuristic
            cre_pred = html_heuristic(html_path)
            # Credential classifier module
            if cre_pred == 1: # if HTML heuristic report as nonCRP
                cre_pred, cred_conf, _  = credential_classifier_mixed_al(img=img_path, coords=pred_boxes, 
                                                                     types=pred_classes, model=cls_model)
            credential_cls_time = time.time() - start_time

            # Non-credential page
            if cre_pred == 1: # if non-CRP
                phish_category = 0 # Report as benign

            # Credential page
            else: 
                # Phishpedia module
                start_time = time.time()
                pred_target, _, _ = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                                    domain_map_path=domain_map_path,
                                                    model=pedia_model, 
                                                    logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                                    url=url,
                                                    shot_path=img_path,
                                                    ts=0.83)
                siamese_time = time.time() - start_time

                # Phishpedia reports target 
                if pred_target is not None:
                    phish_category = 1 # Report as suspicious

                # Phishpedia does not report target
                else: # Report as benign
                    phish_category = 0


        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder+'\t')
            f.write(brand_converter(folder.split('+')[0])+'\t') # true brand
            f.write(str(phish_category)+'\t') # phish/benign/suspicious
            f.write(brand_converter(pred_target)+'\t') if pred_target is not None else f.write('\t')# phishing target
            # element recognition time
            f.write(str(ele_recog_time)+'\t') 
            # credential classifier/heuristic time
            f.write(str(credential_cls_time)+'\t') if 'credential_cls_time' in locals() else f.write('\t')
            # siamese time
            f.write(str(siamese_time)+'\t') if 'siamese_time' in locals() else f.write('\t') 
            # layout time
            f.write(str(layout_time)+'\n') if 'layout_time' in locals() else f.write('\n') 

        # delete time variables
        try:
            del ele_recog_time
            del credential_cls_time
            del siamese_time
            del layout_time
        except:
            pass

    
    
    
    
    
    
    
def evaluate_rule_classifier_rule_matcher(datadir, mode, write_txt, classifier_type):
    assert mode in ['phish', 'benign', 'discovery']
    assert classifier_type in ['layout', 'screenshot', 'mixed']
    
    for folder in tqdm(os.listdir(datadir)):

        phish_category = 0 # 0 for benign, 1 for phish
        pred_target = None # predicted target, default is None

        img_path = os.path.join(datadir, folder, 'shot.png')
        if mode != 'phish':
            url = open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read()
        else:
            url = eval(open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read())['url']

        # Element recognition module
        start_time = time.time()
        pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0 # Report as benign

        # If at least one element is reported
        else:
            # Credential heuristic module
            start_time = time.time()
            pattern_ct, len_input = layout_heuristic(pred_boxes, pred_classes)
            if len_input == 0:
                cre_pred = 1
            elif pattern_ct >= 2:
                cre_pred = 0
            else:
            # Credential classifier module
            if classifier_type == 'screenshot':
                cre_pred, cred_conf, _  = credential_classifier_al_screenshot(img=img_path, model=cls_model)
            elif classifier_type == 'layout':
                cre_pred, cred_conf, _ = credential_classifier_al(img=img_path, coords=pred_boxes, 
                                                 types=pred_classes, model=cls_model)
            elif classifier_type == 'mixed':
                cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=img_path, coords=pred_boxes, 
                                                 types=pred_classes, model=cls_model)
                
            credential_cls_time = time.time() - start_time

            # Credential page
            if cre_pred == 0: 
                # Phishpedia module
                start_time = time.time()
                pred_target, _ = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                                    domain_map_path=domain_map_path,
                                                    model=pedia_model, 
                                                    logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                                    url=url,
                                                    shot_path=img_path,
                                                    ts=0.83)
                siamese_time = time.time() - start_time

                # Phishpedia reports target 
                if pred_target is not None:
                    # Layout module is only built w.r.t specific brands (social media brands)
                    if pred_target not in ['Amazon', 'Facebook', 'Google', 'Instagram', 
                                           'LinkedIn Corporation', 'ms_skype', 'Twitter, Inc.']:
                        phish_category = 2 # Report as phish

                    elif pattern_ct >= 2: # layout heuristic adopted from crp heuristic
                        phish_category = 2 # Report as phish

                    else: 
                        # Layout template matching
                        layout_cfg, gt_coords_arr, gt_clses, gt_files_arr, gt_shot_size_arr = layout_config(cfg_dir=layout_cfg_dir,  ref_dir=layout_ref_dir,  matched_brand=pred_target, ele_model=ele_model)
                        start_time = time.time()
                        max_s, max_site = layout_matcher(pred_boxes=pred_boxes, pred_clses=pred_classes, 
                                                        img=img_path, 
                                                        gt_coords_arr=gt_coords_arr, gt_clses=gt_clses, 
                                                        gt_files_arr=gt_files_arr, gt_shot_size_arr=gt_shot_size_arr,
                                                        cfg=layout_cfg)
                        layout_time = time.time() - start_time

                        # Success layout match
                        if max_s >= layout_ts: 
                            phish_category = 2 # Report as phish

                        # Un-successful layout match
                        else: 
                            phish_category = 1 # Report as suspicious


                # Phishpedia does not report target
                else: # Report as benign
                    phish_category = 0

            # Non-credential page
            elif cre_pred == 1: 
                phish_category = 0 # Report as benign

        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder+'\t')
            f.write(brand_converter(folder.split('+')[0])+'\t') # true brand
            f.write(str(phish_category)+'\t') # phish/benign/suspicious
            f.write(brand_converter(pred_target)+'\t') if pred_target is not None else f.write('\t')# phishing target
            # element recognition time
            f.write(str(ele_recog_time)+'\t') 
            # credential classifier/heuristic time
            f.write(str(credential_cls_time)+'\t') if 'credential_cls_time' in locals() else f.write('\t') 
            # siamese time
            f.write(str(siamese_time)+'\t') if 'siamese_time' in locals() else f.write('\t') 
            # layout time
            f.write(str(layout_time)+'\n') if 'layout_time' in locals() else f.write('\n') 

        # delete time variables
        try:
            del ele_recog_time
            del credential_cls_time
            del siamese_time
            del layout_time
        except:
            pass
        
def evaluate_rule_classifier_matcher(datadir, mode, write_txt, classifier_type):
    assert mode in ['phish', 'benign', 'discovery']
    assert classifier_type in ['layout', 'screenshot', 'mixed']
    
    for folder in tqdm(os.listdir(datadir)):

        phish_category = 0 # 0 for benign, 1 for phish
        pred_target = None # predicted target, default is None

        img_path = os.path.join(datadir, folder, 'shot.png')
        if mode != 'phish':
            url = open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read()
        else:
            url = eval(open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read())['url']

        # Element recognition module
        start_time = time.time()
        pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0 # Report as benign

        # If at least one element is reported
        else:
            # Credential heuristic module
            start_time = time.time()
            pattern_ct, len_input = layout_heuristic(pred_boxes, pred_classes)
            if len_input == 0:
                cre_pred = 1
            elif pattern_ct >= 2:
                cre_pred = 0
            else:
            # Credential classifier module
            if classifier_type == 'screenshot':
                cre_pred, cred_conf, _  = credential_classifier_al_screenshot(img=img_path, model=cls_model)
            elif classifier_type == 'layout':
                cre_pred, cred_conf, _ = credential_classifier_al(img=img_path, coords=pred_boxes, 
                                                 types=pred_classes, model=cls_model)
            elif classifier_type == 'mixed':
                cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=img_path, coords=pred_boxes, 
                                                 types=pred_classes, model=cls_model)
                
            credential_cls_time = time.time() - start_time

            # Credential page
            if cre_pred == 0: 
                # Phishpedia module
                start_time = time.time()
                pred_target, _ = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                                    domain_map_path=domain_map_path,
                                                    model=pedia_model, 
                                                    logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                                    url=url,
                                                    shot_path=img_path,
                                                    ts=0.83)
                siamese_time = time.time() - start_time

                # Phishpedia reports target 
                if pred_target is not None:
                    # Layout module is only built w.r.t specific brands (social media brands)
                    if pred_target not in ['Amazon', 'Facebook', 'Google', 'Instagram', 
                                           'LinkedIn Corporation', 'ms_skype', 'Twitter, Inc.']:
                        phish_category = 2 # Report as phish

                    else: 
                        # Layout template matching
                        layout_cfg, gt_coords_arr, gt_clses, gt_files_arr, gt_shot_size_arr = layout_config(cfg_dir=layout_cfg_dir,  ref_dir=layout_ref_dir,  matched_brand=pred_target, ele_model=ele_model)
                        start_time = time.time()
                        max_s, max_site = layout_matcher(pred_boxes=pred_boxes, pred_clses=pred_classes, 
                                                        img=img_path, 
                                                        gt_coords_arr=gt_coords_arr, gt_clses=gt_clses, 
                                                        gt_files_arr=gt_files_arr, gt_shot_size_arr=gt_shot_size_arr,
                                                        cfg=layout_cfg)
                        layout_time = time.time() - start_time

                        # Success layout match
                        if max_s >= layout_ts: 
                            phish_category = 2 # Report as phish

                        # Un-successful layout match
                        else: 
                            phish_category = 1 # Report as suspicious


                # Phishpedia does not report target
                else: # Report as benign
                    phish_category = 0

            # Non-credential page
            elif cre_pred == 1: 
                phish_category = 0 # Report as benign

        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder+'\t')
            f.write(brand_converter(folder.split('+')[0])+'\t') # true brand
            f.write(str(phish_category)+'\t') # phish/benign/suspicious
            f.write(brand_converter(pred_target)+'\t') if pred_target is not None else f.write('\t')# phishing target
            # element recognition time
            f.write(str(ele_recog_time)+'\t') 
            # credential classifier/heuristic time
            f.write(str(credential_cls_time)+'\t') if 'credential_cls_time' in locals() else f.write('\t') 
            # siamese time
            f.write(str(siamese_time)+'\t') if 'siamese_time' in locals() else f.write('\t') 
            # layout time
            f.write(str(layout_time)+'\n') if 'layout_time' in locals() else f.write('\n') 

        # delete time variables
        try:
            del ele_recog_time
            del credential_cls_time
            del siamese_time
            del layout_time
        except:
            pass


        
def evaluate_classifier_matcher(datadir, mode, write_txt, classifier_type):
    assert mode in ['phish', 'benign', 'discovery']
    assert classifier_type in ['layout', 'screenshot', 'mixed']
    
    for folder in tqdm(os.listdir(datadir)):

        phish_category = 0 # 0 for benign, 1 for phish
        pred_target = None # predicted target, default is None

        img_path = os.path.join(datadir, folder, 'shot.png')
        if mode != 'phish':
            url = open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read()
        else:
            url = eval(open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read())['url']

        # Element recognition module
        start_time = time.time()
        pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0 # Report as benign

        # If at least one element is reported
        else:
            start_time = time.time()
            # Credential classifier module
            if classifier_type == 'screenshot':
                cre_pred, cred_conf, _  = credential_classifier_al_screenshot(img=img_path, model=cls_model)
            elif classifier_type == 'layout':
                cre_pred, cred_conf, _ = credential_classifier_al(img=img_path, coords=pred_boxes, 
                                                 types=pred_classes, model=cls_model)
            elif classifier_type == 'mixed':
                cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=img_path, coords=pred_boxes, 
                                                 types=pred_classes, model=cls_model)
                
            credential_cls_time = time.time() - start_time

            # Credential page
            if cre_pred == 0: 
                # Phishpedia module
                start_time = time.time()
                pred_target, _ = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                                    domain_map_path=domain_map_path,
                                                    model=pedia_model, 
                                                    logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                                    url=url,
                                                    shot_path=img_path,
                                                    ts=0.83)
                siamese_time = time.time() - start_time

                # Phishpedia reports target 
                if pred_target is not None:
                    # Layout module is only built w.r.t specific brands (social media brands)
                    if pred_target not in ['Amazon', 'Facebook', 'Google', 'Instagram', 
                                           'LinkedIn Corporation', 'ms_skype', 'Twitter, Inc.']:
                        phish_category = 2 # Report as phish

#                     elif pattern_ct >= 2: # layout heuristic adopted from crp heuristic
#                         phish_category = 2 # Report as phish

                    else: 
                        # Layout template matching
                        layout_cfg, gt_coords_arr, gt_clses, gt_files_arr, gt_shot_size_arr = layout_config(cfg_dir=layout_cfg_dir,  ref_dir=layout_ref_dir,  matched_brand=pred_target, ele_model=ele_model)
                        start_time = time.time()
                        max_s, max_site = layout_matcher(pred_boxes=pred_boxes, pred_clses=pred_classes, 
                                                        img=img_path, 
                                                        gt_coords_arr=gt_coords_arr, gt_clses=gt_clses, 
                                                        gt_files_arr=gt_files_arr, gt_shot_size_arr=gt_shot_size_arr,
                                                        cfg=layout_cfg)
                        layout_time = time.time() - start_time

                        # Success layout match
                        if max_s >= layout_ts: 
                            phish_category = 2 # Report as phish

                        # Un-successful layout match
                        else: 
                            phish_category = 1 # Report as suspicious


                # Phishpedia does not report target
                else: # Report as benign
                    phish_category = 0

            # Non-credential page
            elif cre_pred == 1: 
                phish_category = 0 # Report as benign

        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder+'\t')
            f.write(brand_converter(folder.split('+')[0])+'\t') # true brand
            f.write(str(phish_category)+'\t') # phish/benign/suspicious
            f.write(brand_converter(pred_target)+'\t') if pred_target is not None else f.write('\t')# phishing target
            # element recognition time
            f.write(str(ele_recog_time)+'\t') 
            # credential classifier/heuristic time
            f.write(str(credential_cls_time)+'\t') if 'credential_cls_time' in locals() else f.write('\t') 
            # siamese time
            f.write(str(siamese_time)+'\t') if 'siamese_time' in locals() else f.write('\t') 
            # layout time
            f.write(str(layout_time)+'\n') if 'layout_time' in locals() else f.write('\n') 

        # delete time variables
        try:
            del ele_recog_time
            del credential_cls_time
            del siamese_time
            del layout_time
        except:
            pass



def evaluate_classifier(datadir, mode, write_txt, classifier_type):
    assert mode in ['phish', 'benign', 'discovery']
    assert classifier_type in ['layout', 'screenshot', 'mixed']
    
    for folder in tqdm(os.listdir(datadir)):

        phish_category = 0 # 0 for benign, 1 for phish
        pred_target = None # predicted target, default is None

        img_path = os.path.join(datadir, folder, 'shot.png')
        if mode != 'phish':
            url = open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read()
        else:
            url = eval(open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read())['url']

        # Element recognition module
        start_time = time.time()
        pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0 # Report as benign

        # If at least one element is reported
        else:
            start_time = time.time()
            # Credential classifier module
            if classifier_type == 'screenshot':
                cre_pred, cred_conf, _  = credential_classifier_al_screenshot(img=img_path, model=cls_model)
            elif classifier_type == 'layout':
                cre_pred, cred_conf, _ = credential_classifier_al(img=img_path, coords=pred_boxes, 
                                                 types=pred_classes, model=cls_model)
            elif classifier_type == 'mixed':
                cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=img_path, coords=pred_boxes, 
                                                 types=pred_classes, model=cls_model)
                
            credential_cls_time = time.time() - start_time

            # Credential page
            if cre_pred == 0: 
                # Phishpedia module
                start_time = time.time()
                pred_target, _ = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                                    domain_map_path=domain_map_path,
                                                    model=pedia_model, 
                                                    logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                                    url=url,
                                                    shot_path=img_path,
                                                    ts=0.83)
                siamese_time = time.time() - start_time

                # Phishpedia reports target 
                if pred_target is not None:
                    phish_category = 1 # Report as suspicious


                # Phishpedia does not report target
                else: # Report as benign
                    phish_category = 0

            # Non-credential page
            elif cre_pred == 1: 
                phish_category = 0 # Report as benign

        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder+'\t')
            f.write(brand_converter(folder.split('+')[0])+'\t') # true brand
            f.write(str(phish_category)+'\t') # phish/benign/suspicious
            f.write(brand_converter(pred_target)+'\t') if pred_target is not None else f.write('\t')# phishing target
            # element recognition time
            f.write(str(ele_recog_time)+'\t') 
            # credential classifier/heuristic time
            f.write(str(credential_cls_time)+'\t') if 'credential_cls_time' in locals() else f.write('\t') 
            # siamese time
            f.write(str(siamese_time)+'\t') if 'siamese_time' in locals() else f.write('\t') 
            # layout time
            f.write(str(layout_time)+'\n') if 'layout_time' in locals() else f.write('\n') 

        # delete time variables
        try:
            del ele_recog_time
            del credential_cls_time
            del siamese_time
            del layout_time
        except:
            pass


def evaluate_baseline(datadir, mode, write_txt):
    assert mode in ['phish', 'benign', 'discovery']
    
    for folder in tqdm(os.listdir(datadir)):

        phish_category = 0 # 0 for benign, 1 for phish
        pred_target = None # predicted target, default is None

        img_path = os.path.join(datadir, folder, 'shot.png')
        if mode != 'phish':
            url = open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read()
        else:
            url = eval(open(os.path.join(datadir, folder, 'info.txt'), encoding = "ISO-8859-1").read())['url']

        # Element recognition module
        start_time = time.time()
        pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0 # Report as benign

        # If at least one element is reported
        else:

            start_time = time.time()
            credential_cls_time = time.time() - start_time

            # Phishpedia module
            start_time = time.time()
            pred_target, _ = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes, 
                                                domain_map_path=domain_map_path,
                                                model=pedia_model, 
                                                logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                                url=url,
                                                shot_path=img_path,
                                                ts=0.83)
            siamese_time = time.time() - start_time

            # Phishpedia reports target 
            if pred_target is not None:
                phish_category = 1 # Report as suspicious

            # Phishpedia does not report target
            else: # Report as benign
                phish_category = 0


        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder+'\t')
            f.write(brand_converter(folder.split('+')[0])+'\t') # true brand
            f.write(str(phish_category)+'\t') # phish/benign/suspicious
            f.write(brand_converter(pred_target)+'\t') if pred_target is not None else f.write('\t')# phishing target
            # element recognition time
            f.write(str(ele_recog_time)+'\t') 
            # credential classifier/heuristic time
            f.write(str(credential_cls_time)+'\t') if 'credential_cls_time' in locals() else f.write('\t') 
            # siamese time
            f.write(str(siamese_time)+'\t') if 'siamese_time' in locals() else f.write('\t') 
            # layout time
            f.write(str(layout_time)+'\n') if 'layout_time' in locals() else f.write('\n') 

        # delete time variables
        try:
            del ele_recog_time
            del credential_cls_time
            del siamese_time
            del layout_time
        except:
            pass
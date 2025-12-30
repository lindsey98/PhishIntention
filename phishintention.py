import time
from datetime import datetime, timezone
import argparse
import os
import json
import torch
import cv2
import logging
from configs import load_config
from modules.awl_detector import pred_rcnn, vis, find_element_type
from modules.logo_matching import check_domain_brand_inconsistency
from modules.crp_classifier import credential_classifier_mixed, html_heuristic
from modules.crp_locator import crp_locator
from utils.web_utils import driver_loader
from tqdm import tqdm
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
RESULT_VERSION = "1.0"


class PhishIntentionWrapper:
    _caller_prefix = "PhishIntentionWrapper"
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self._load_config()

    def _load_config(self):
        self.AWL_MODEL, self.CRP_CLASSIFIER, self.CRP_LOCATOR_MODEL, self.SIAMESE_MODEL, self.OCR_MODEL, \
            self.SIAMESE_THRE, self.LOGO_FEATS, self.LOGO_FILES, self.DOMAIN_MAP_PATH = load_config()
        logger.info(f'Length of reference list = {len(self.LOGO_FEATS)}')

    def _step1_layout_detector(self, screenshot_path):
        """Step 1: Layout detection with AWL detector"""
        start_time = time.time()
        pred_boxes, pred_classes, _ = pred_rcnn(im=screenshot_path, predictor=self.AWL_MODEL)
        awl_detect_time = time.time() - start_time

        if pred_boxes is not None:
            pred_boxes = pred_boxes.numpy()
            pred_classes = pred_classes.numpy()
        
        plotvis = vis(screenshot_path, pred_boxes, pred_classes)
        
        return pred_boxes, pred_classes, plotvis, awl_detect_time

    def _step2_logo_matcher(self, logo_pred_boxes, url, screenshot_path):
        """Step 2: Logo matching with Siamese network"""
        start_time = time.time()
        pred_target, matched_domain, matched_coord, siamese_conf = check_domain_brand_inconsistency(
            logo_boxes=logo_pred_boxes,
            domain_map_path=self.DOMAIN_MAP_PATH,
            model=self.SIAMESE_MODEL,
            ocr_model=self.OCR_MODEL,
            logo_feat_list=self.LOGO_FEATS,
            file_name_list=self.LOGO_FILES,
            url=url,
            shot_path=screenshot_path,
            ts=self.SIAMESE_THRE
        )
        logo_match_time = time.time() - start_time
        
        return pred_target, matched_domain, matched_coord, siamese_conf, logo_match_time

    def _step3_crp_classifier(self, screenshot_path, html_path, pred_boxes, pred_classes):
        """Step 3: CRP classifier for credential pages"""
        start_time = time.time()
        cre_pred = html_heuristic(html_path)
        
        if cre_pred == 1:  # if HTML heuristic report as nonCRP
            cre_pred = credential_classifier_mixed(
                img=screenshot_path,
                coords=pred_boxes,
                types=pred_classes,
                model=self.CRP_CLASSIFIER
            )
        
        crp_class_time = time.time() - start_time
        return cre_pred, crp_class_time

    def _step4_dynamic_analysis(self, url, screenshot_path, pred_boxes, pred_classes):
        """Step 4: Dynamic analysis to find CRP pages"""
        # load driver ONCE!
        driver = driver_loader()
        logger.info('Finish loading webdriver')
        
        # load chromedriver
        url, screenshot_path, successful, process_time = crp_locator(
            url=url,
            screenshot_path=screenshot_path,
            cls_model=self.CRP_CLASSIFIER,
            ele_model=self.AWL_MODEL,
            login_model=self.CRP_LOCATOR_MODEL,
            driver=driver
        )
        
        driver.quit()
        return url, screenshot_path, successful, process_time

    def _step5_result_processing(self, plotvis, pred_target, siamese_conf, matched_coord):
        """Step 5: Final result processing and visualization"""
        phish_category = "benign"
        if pred_target is not None:
            logger.warning('Phishing is found!')
            phish_category = "phish"
            # Visualize, add annotations
            cv2.putText(plotvis, 
                       f"Target: {pred_target} with confidence {siamese_conf:.4f}",
                       (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return phish_category, plotvis

    def test_orig_phishintention(self, url, screenshot_path):
        """Main method to run PhishIntention pipeline"""
        waive_crp_classifier = False
        phish_category = "benign"  # "benign" or "phish", default is benign
        pred_target = None
        matched_domain = None
        matched_coord = None
        siamese_conf = None
        awl_detect_time = 0
        logo_match_time = 0
        crp_class_time = 0
        crp_locator_time = 0
        
        logger.info("Entering PhishIntention")

        while True:
            # Step 1: Layout detector
            pred_boxes, pred_classes, plotvis, step1_time = self._step1_layout_detector(screenshot_path)
            awl_detect_time += step1_time

            # If no element is detected
            if pred_boxes is None or len(pred_boxes) == 0:
                logger.info('No element detected, reported as benign')
                return self._build_return_result(
                    phish_category, pred_target, matched_domain, plotvis, siamese_conf,
                    awl_detect_time, logo_match_time, crp_class_time, crp_locator_time,
                    pred_boxes, pred_classes
                )

            # Check for logo elements
            logo_pred_boxes, _ = find_element_type(pred_boxes, pred_classes, bbox_type='logo')
            if logo_pred_boxes is None or len(logo_pred_boxes) == 0:
                logger.info('No logo detected, reported as benign')
                return self._build_return_result(
                    phish_category, pred_target, matched_domain, plotvis, siamese_conf,
                    awl_detect_time, logo_match_time, crp_class_time, crp_locator_time,
                    pred_boxes, pred_classes
                )

            logger.info('Entering logo matching (Siamese network)')

            # Step 2: Siamese (Logo matcher)
            pred_target, matched_domain, matched_coord, siamese_conf, step2_time = self._step2_logo_matcher(
                logo_pred_boxes, url, screenshot_path
            )
            logo_match_time += step2_time

            if pred_target is None:
                logger.info('No brand matched, reported as benign')
                return self._build_return_result(
                    phish_category, pred_target, matched_domain, plotvis, siamese_conf,
                    awl_detect_time, logo_match_time, crp_class_time, crp_locator_time,
                    pred_boxes, pred_classes
                )

            # Step 3: CRP classifier (if a target is reported)
            logger.info(f'Target brand detected: {pred_target}, entering CRP classifier')
            if waive_crp_classifier:  # only run dynamic analysis ONCE
                break

            html_path = screenshot_path.replace("shot.png", "html.txt")
            cre_pred, step3_time = self._step3_crp_classifier(screenshot_path, html_path, pred_boxes, pred_classes)
            crp_class_time += step3_time

            # Step 4: Dynamic analysis
            if cre_pred == 1:
                logger.info('Non-CRP page detected, entering dynamic analysis')
                url, screenshot_path, successful, step4_time = self._step4_dynamic_analysis(
                    url, screenshot_path, pred_boxes, pred_classes
                )
                crp_locator_time += step4_time
                waive_crp_classifier = True  # only run dynamic analysis ONCE

                # If dynamic analysis did not reach a CRP
                if not successful:
                    logger.info('Dynamic analysis: no CRP page found via link redirection, reported as benign')
                    return self._build_return_result(
                        phish_category, pred_target, matched_domain, plotvis, siamese_conf,
                        awl_detect_time, logo_match_time, crp_class_time, crp_locator_time,
                        pred_boxes, pred_classes
                    )
                else:  # dynamic analysis successfully found a CRP
                    logger.info('Dynamic analysis: CRP page found, returning to layout detector')
            else:  # already a CRP page
                logger.info('CRP page confirmed, proceeding to final analysis')
                break

        # Step 5: Final result processing
        phish_category, plotvis = self._step5_result_processing(plotvis, pred_target, siamese_conf, matched_coord)
        
        return self._build_return_result(
            phish_category, pred_target, matched_domain, plotvis, siamese_conf,
            awl_detect_time, logo_match_time, crp_class_time, crp_locator_time,
            pred_boxes, pred_classes
        )

    def _build_return_result(self, phish_category, pred_target, matched_domain, plotvis, siamese_conf,
                            awl_detect_time, logo_match_time, crp_class_time, crp_locator_time,
                            pred_boxes, pred_classes):
        """Helper method to build the return result tuple"""
        runtime_breakdown = f"{awl_detect_time:.4f}|{logo_match_time:.4f}|{crp_class_time:.4f}|{crp_locator_time:.4f}"
        return (phish_category, pred_target, matched_domain, plotvis, siamese_conf,
                runtime_breakdown, pred_boxes, pred_classes)


def _is_forbidden_url(url):
    """Check if URL has forbidden file extensions"""
    _forbidden_suffixes = r"\.(mp3|wav|wma|ogg|mkv|zip|tar|xz|rar|z|deb|bin|iso|csv|tsv|dat|txt|css|log|sql|xml|sql|mdb|apk|bat|bin|exe|jar|wsf|fnt|fon|otf|ttf|ai|bmp|gif|ico|jp(e)?g|png|ps|psd|svg|tif|tiff|cer|rss|key|odp|pps|ppt|pptx|c|class|cpp|cs|h|java|sh|swift|vb|odf|xlr|xls|xlsx|bak|cab|cfg|cpl|cur|dll|dmp|drv|icns|ini|lnk|msi|sys|tmp|3g2|3gp|avi|flv|h264|m4v|mov|mp4|mp(e)?g|rm|swf|vob|wmv|doc(x)?|odt|rtf|tex|txt|wks|wps|wpd)$"
    return re.search(_forbidden_suffixes, url, re.IGNORECASE)


def _load_url_from_info(folder, request_dir):
    """Load URL from info.txt file or use folder name as fallback"""
    info_path = os.path.join(request_dir, folder, 'info.txt')
    if os.path.exists(info_path):
        return open(info_path).read()
    else:
        return "https://" + folder


def _parse_runtime_breakdown(runtime_breakdown):
    """Split runtime string into numeric components."""
    parts = str(runtime_breakdown).split("|") if runtime_breakdown else []
    times = [0.0, 0.0, 0.0, 0.0]
    for idx, part in enumerate(parts[:4]):
        try:
            times[idx] = float(part)
        except (TypeError, ValueError):
            continue
    return tuple(times)


def _write_result_to_file(result_json, folder, url, phish_category, pred_target, 
                         matched_domain, siamese_conf, runtime_breakdown):
    """Write result to JSON format with metadata."""
    awl_time, logo_time, crp_class_time, crp_locator_time = _parse_runtime_breakdown(runtime_breakdown)
    
    processed_at = datetime.now(timezone.utc).isoformat()
    
    os.makedirs(os.path.dirname(result_json) or ".", exist_ok=True)
    
    # Read existing results or create new file
    file_exists = os.path.exists(result_json)
    results = []
    file_created_at = None
    
    if file_exists:
        try:
            with open(result_json, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    # Handle both list format and dict format
                    if isinstance(data, list):
                        results = data
                        # Try to get file_created_at from first entry's metadata
                        if results and isinstance(results[0], dict) and "metadata" in results[0]:
                            file_created_at = results[0]["metadata"].get("file_created_at")
                    elif isinstance(data, dict):
                        results = [data]
                        if "metadata" in data:
                            file_created_at = data["metadata"].get("file_created_at")
        except (json.JSONDecodeError, IOError):
            results = []
    
    # Set file_created_at only if file is new
    if file_created_at is None:
        file_created_at = datetime.now(timezone.utc).isoformat()
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to Python native types."""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        return obj
    
    # Build result entry
    result_entry = {
        "folder": folder,
        "url": url,
        "phish_category": phish_category,
        "pred_target": convert_to_native(pred_target),
        "matched_domain": matched_domain,
        "siamese_conf": convert_to_native(siamese_conf),
        "runtime": {
            "awl_detect_time": float(awl_time),
            "logo_match_time": float(logo_time),
            "crp_class_time": float(crp_class_time),
            "crp_locator_time": float(crp_locator_time),
            "total_time": float(awl_time + logo_time + crp_class_time + crp_locator_time)
        },
        "processed_at": processed_at,
        "metadata": {
            "version": RESULT_VERSION,
            "file_created_at": file_created_at
        }
    }
    
    # Append new result
    results.append(result_entry)
    
    # Write all results back to file
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _save_visualization_if_phishing(phish_category, plotvis, request_dir, folder):
    """Save visualization image if phishing is detected"""
    if phish_category == "phish":
        os.makedirs(os.path.join(request_dir, folder), exist_ok=True)
        cv2.imwrite(os.path.join(request_dir, folder, "predict.png"), plotvis)


def _process_single_folder(folder, request_dir, phishintention_cls, output_fn, stats):
    """Process a single folder with PhishIntention"""
    #html_path = os.path.join(request_dir, folder, "html.txt")
    screenshot_path = os.path.join(request_dir, folder, "shot.png")
    
    if not os.path.exists(screenshot_path):
        stats['skipped'] += 1
        logger.debug(f"Skipping {folder}: screenshot not found")
        return None

    url = _load_url_from_info(folder, request_dir)
    
    # Check if URL already processed
    if os.path.exists(output_fn):
        try:
            with open(output_fn, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    results = json.loads(content)
                    if isinstance(results, list):
                        if any(entry.get("url") == url for entry in results):
                            stats['skipped'] += 1
                            logger.debug(f"Skipping {folder}: URL already processed")
                            return None
                    elif isinstance(results, dict) and results.get("url") == url:
                        stats['skipped'] += 1
                        logger.debug(f"Skipping {folder}: URL already processed")
                        return None
        except (json.JSONDecodeError, IOError):
            pass
    
    # Check for forbidden URL suffixes
    if _is_forbidden_url(url):
        stats['skipped'] += 1
        logger.debug(f"Skipping {folder}: forbidden URL suffix")
        return None

    stats['processed'] += 1
    logger.info(f"Processing folder: {folder} | URL: {url}")
    
    # Run PhishIntention
    phish_category, pred_target, matched_domain, plotvis, siamese_conf, \
        runtime_breakdown, pred_boxes, pred_classes = phishintention_cls.test_orig_phishintention(url, screenshot_path)

    # Update statistics
    if phish_category == "phish":
        stats['phish'] += 1
        logger.warning(f"⚠️  Phishing detected in {folder} | Target: {pred_target} | Confidence: {siamese_conf:.4f}")
    else:
        stats['benign'] += 1
        logger.info(f"✓ Benign site: {folder}")

    # Write results
    _write_result_to_file(output_fn, folder, url, phish_category, pred_target, 
                         matched_domain, siamese_conf, runtime_breakdown)
    
    # Save visualization if phishing
    _save_visualization_if_phishing(phish_category, plotvis, request_dir, folder)
    
    return phish_category


if __name__ == '__main__':
    today = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str)
    parser.add_argument("--output_fn", default=f'{today}_results.json', help="Output JSON path")
    args = parser.parse_args()

    request_dir = args.folder
    phishintention_cls = PhishIntentionWrapper()
    output_fn = args.output_fn

    os.makedirs(request_dir, exist_ok=True)

    # Initialize statistics
    stats = {
        'total': 0,
        'processed': 0,
        'skipped': 0,
        'phish': 0,
        'benign': 0
    }

    # Get all folders
    folders = [f for f in os.listdir(request_dir) if os.path.isdir(os.path.join(request_dir, f))]
    stats['total'] = len(folders)
    
    logger.info("=" * 60)
    logger.info("PhishIntention Processing Started")
    logger.info(f"Input directory: {request_dir}")
    logger.info(f"Output file: {output_fn}")
    logger.info(f"Total folders to process: {stats['total']}")
    logger.info("=" * 60)

    # Process all folders
    for folder in tqdm(folders, desc="Processing sites", unit="site"):
        _process_single_folder(folder, request_dir, phishintention_cls, output_fn, stats)

    # Print summary statistics
    logger.info("=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Total folders: {stats['total']}")
    logger.info(f"  ├─ Processed: {stats['processed']}")
    logger.info(f"  ├─ Skipped: {stats['skipped']}")
    logger.info("Results:")
    logger.info(f"  ├─ Phishing sites: {stats['phish']}")
    logger.info(f"  └─ Benign sites: {stats['benign']}")
    if stats['processed'] > 0:
        phish_rate = (stats['phish'] / stats['processed']) * 100
        logger.info(f"Phishing detection rate: {phish_rate:.2f}%")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_fn}")
    logger.info("=" * 60)
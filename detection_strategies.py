from abc import ABC, abstractmethod  
import time  
import cv2  
  
class DetectionStrategy(ABC):  
    """检测策略基类"""  
      
    @abstractmethod  
    def execute(self, context: DetectionContext) -> DetectionResult:  
        pass  
  
class LayoutDetectionStrategy(DetectionStrategy):  
    """Stage 1: 布局检测"""  
      
    def __init__(self, awl_model):  
        self.awl_model = awl_model  
      
    def execute(self, context: DetectionContext) -> DetectionResult:  
        start_time = time.time()  
          
        # 执行布局检测  
        pred_boxes, pred_classes, _ = pred_rcnn(im=context.screenshot_path, predictor=self.awl_model)  
        context.awl_detect_time += time.time() - start_time  
          
        # 处理检测结果  
        if pred_boxes is not None:  
            pred_boxes = pred_boxes.numpy()  
            pred_classes = pred_classes.numpy()  
          
        plotvis = vis(context.screenshot_path, pred_boxes, pred_classes)  
        context.update_from_stage1(pred_boxes, pred_classes, plotvis)  
          
        # 检查是否检测到元素  
        if pred_boxes is None or len(pred_boxes) == 0:  
            return DetectionResult.benign(context)  
          
        # 提取logo区域  
        logo_boxes, _ = find_element_type(pred_boxes, pred_classes, bbox_type='logo')  
        if logo_boxes is None or len(logo_boxes) == 0:  
            return DetectionResult.benign(context)  
          
        return DetectionResult.continue_to_next_stage(context, "logo_matching")  
  
class LogoMatchingStrategy(DetectionStrategy):  
    """Stage 2: Logo匹配"""  
      
    def __init__(self, siamese_model, ocr_model, logo_feats, logo_files, domain_map_path, siamese_thre):  
        self.siamese_model = siamese_model  
        self.ocr_model = ocr_model  
        self.logo_feats = logo_feats  
        self.logo_files = logo_files  
        self.domain_map_path = domain_map_path  
        self.siamese_thre = siamese_thre  
      
    def execute(self, context: DetectionContext) -> DetectionResult:  
        start_time = time.time()  
          
        # 执行logo匹配  
        pred_target, matched_domain, matched_coord, siamese_conf = check_domain_brand_inconsistency(  
            logo_boxes=context.logo_boxes,  
            domain_map_path=self.domain_map_path,  
            model=self.siamese_model,  
            ocr_model=self.ocr_model,  
            logo_feat_list=self.logo_feats,  
            file_name_list=self.logo_files,  
            url=context.url,  
            shot_path=context.screenshot_path,  
            ts=self.siamese_thre  
        )  
        context.logo_match_time += time.time() - start_time  
          
        context.update_from_stage2(context.logo_boxes, pred_target, matched_domain, matched_coord, siamese_conf)  
          
        # 检查是否匹配到品牌  
        if pred_target is None:  
            return DetectionResult.benign(context)  
          
        return DetectionResult.continue_to_next_stage(context, "crp_classification")  
  
class CRPClassificationStrategy(DetectionStrategy):  
    """Stage 3: CRP分类"""  
      
    def __init__(self, crp_classifier):  
        self.crp_classifier = crp_classifier  
      
    def execute(self, context: DetectionContext) -> DetectionResult:  
        start_time = time.time()  
          
        # 检查是否需要跳过  
        if context.waive_crp_classifier:  
            return DetectionResult.continue_to_next_stage(context, "final_classification")  
          
        # 执行CRP分类  
        html_path = context.screenshot_path.replace("shot.png", "html.txt")  
        cre_pred = html_heuristic(html_path)  
          
        if cre_pred == 1:  # 如果HTML启发式报告为非CRP  
            cre_pred = credential_classifier_mixed(  
                img=context.screenshot_path,  
                coords=context.pred_boxes,  
                types=context.pred_classes,  
                model=self.crp_classifier  
            )  
          
        context.crp_class_time += time.time() - start_time  
        context.update_from_stage3(cre_pred)  
          
        if cre_pred == 1:  
            return DetectionResult.continue_to_next_stage(context, "dynamic_analysis")  
        else:  
            return DetectionResult.continue_to_next_stage(context, "final_classification")  
  
class DynamicAnalysisStrategy(DetectionStrategy):  
    """Stage 4: 动态分析"""  
      
    def __init__(self, crp_locator_model, awl_model, crp_classifier):  
        self.crp_locator_model = crp_locator_model  
        self.awl_model = awl_model  
        self.crp_classifier = crp_classifier  
      
    def execute(self, context: DetectionContext) -> DetectionResult:  
        driver = driver_loader()  
          
        url, screenshot_path, successful, process_time = crp_locator(  
            url=context.url,  
            screenshot_path=context.screenshot_path,  
            cls_model=self.crp_classifier,  
            ele_model=self.awl_model,  
            login_model=self.crp_locator_model,  
            driver=driver  
        )  
          
        context.crp_locator_time += process_time  
        driver.quit()  
        context.waive_crp_classifier = True  
          
        context.update_from_stage4(url, screenshot_path, successful, process_time)  
          
        if not successful:  
            return DetectionResult.benign(context)  
          
        # 成功找到CRP，需要重新循环  
        return DetectionResult.continue_to_next_stage(context, "layout_detection")
from dataclasses import dataclass, field  
from typing import Optional, Tuple, Any  
import numpy as np  
  
@dataclass  
class DetectionContext:  
    """检测上下文，携带各阶段所需的数据"""  
    # 输入数据  
    url: str  
    screenshot_path: str  
    html_path: Optional[str] = None  
      
    # Stage 1 输出  
    pred_boxes: Optional[np.ndarray] = None  
    pred_classes: Optional[np.ndarray] = None  
    plotvis: Optional[np.ndarray] = None  
      
    # Stage 2 输出  
    logo_boxes: Optional[np.ndarray] = None  
    pred_target: Optional[str] = None  
    matched_domain: Optional[str] = None  
    matched_coord: Optional[Tuple] = None  
    siamese_conf: Optional[float] = None  
      
    # Stage 3 输出  
    cre_pred: Optional[int] = None  
      
    # Stage 4 输出  
    successful: Optional[bool] = None  
    process_time: Optional[float] = None  
      
    # 控制标志  
    waive_crp_classifier: bool = False  
      
    # 性能统计  
    awl_detect_time: float = 0.0  
    logo_match_time: float = 0.0  
    crp_class_time: float = 0.0  
    crp_locator_time: float = 0.0  
      
    def update_from_stage1(self, pred_boxes, pred_classes, plotvis):  
        """更新Stage 1结果"""  
        self.pred_boxes = pred_boxes  
        self.pred_classes = pred_classes  
        self.plotvis = plotvis  
      
    def update_from_stage2(self, logo_boxes, pred_target, matched_domain, matched_coord, siamese_conf):  
        """更新Stage 2结果"""  
        self.logo_boxes = logo_boxes  
        self.pred_target = pred_target  
        self.matched_domain = matched_domain  
        self.matched_coord = matched_coord  
        self.siamese_conf = siamese_conf  
      
    def update_from_stage3(self, cre_pred):  
        """更新Stage 3结果"""  
        self.cre_pred = cre_pred  
      
    def update_from_stage4(self, url, screenshot_path, successful, process_time):  
        """更新Stage 4结果"""  
        self.url = url  
        self.screenshot_path = screenshot_path  
        self.successful = successful  
        self.process_time = process_time
        
@dataclass  
class DetectionResult:  
    """检测结果，统一的返回格式"""  
    # 最终结果  
    is_phishing: bool  
    phish_category: int  # 0 for benign, 1 for phish  
    pred_target: Optional[str]  
    matched_domain: Optional[str]  
    siamese_conf: Optional[float]  
      
    # 可视化数据  
    plotvis: Optional[np.ndarray]  
    pred_boxes: Optional[np.ndarray]  
    pred_classes: Optional[np.ndarray]  
      
    # 性能数据  
    runtime_breakdown: str  
      
    # 控制信息  
    should_stop: bool = False  
    should_continue_loop: bool = False  
    next_stage: Optional[str] = None  
      
    @classmethod  
    def benign(cls, context: DetectionContext):  
        """创建良性结果"""  
        return cls(  
            is_phishing=False,  
            phish_category=0,  
            pred_target=None,  
            matched_domain=None,  
            siamese_conf=None,  
            plotvis=context.plotvis,  
            pred_boxes=context.pred_boxes,  
            pred_classes=context.pred_classes,  
            runtime_breakdown=f"{context.awl_detect_time}|{context.logo_match_time}|{context.crp_class_time}|{context.crp_locator_time}",  
            should_stop=True  
        )  
      
    @classmethod  
    def phishing(cls, context: DetectionContext):  
        """创建钓鱼结果"""  
        return cls(  
            is_phishing=True,  
            phish_category=1,  
            pred_target=context.pred_target,  
            matched_domain=context.matched_domain,  
            siamese_conf=context.siamese_conf,  
            plotvis=context.plotvis,  
            pred_boxes=context.pred_boxes,  
            pred_classes=context.pred_classes,  
            runtime_breakdown=f"{context.awl_detect_time}|{context.logo_match_time}|{context.crp_class_time}|{context.crp_locator_time}",  
            should_stop=True  
        )  
      
    @classmethod  
    def continue_to_next_stage(cls, context: DetectionContext, next_stage: str):  
        """创建继续到下一阶段的结果"""  
        return cls(  
            is_phishing=False,  
            phish_category=0,  
            pred_target=None,  
            matched_domain=None,  
            siamese_conf=None,  
            plotvis=context.plotvis,  
            pred_boxes=context.pred_boxes,  
            pred_classes=context.pred_classes,  
            runtime_breakdown="",  
            should_stop=False,  
            should_continue_loop=True,  
            next_stage=next_stage  
        )
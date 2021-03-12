# https://github.com/lolipopshock/Detectron2_AL/tree/7eb444e165f1aea6b3e1930ba0097dcaadf4c705/src/detectron2_al

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import ProposalNetwork, GeneralizedRCNN
import torch

__all__ = ['ActiveLearningRCNN']


@META_ARCH_REGISTRY.register()
class ActiveLearningRCNN(GeneralizedRCNN):

    def _estimate_feature_proposal(self, batched_inputs):
        
        with torch.no_grad():
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            if self.proposal_generator:
                rpn_proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                rpn_proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        return features, rpn_proposals

    def generate_object_al_scores(self, batched_inputs, feature_refs, do_postprocess=False):
        
        with torch.no_grad():
            features, rpn_proposals = self._estimate_feature_proposal(batched_inputs)

        return  self.roi_heads.generate_object_scores(features, rpn_proposals, feature_refs)
    
    def generate_image_al_scores(self, batched_inputs, feature_refs):
        """
        Returns: List[Tuple]
            A list of (image_score, image_id) tuple
        """
        with torch.no_grad():
            features, rpn_proposals = self._estimate_feature_proposal(batched_inputs)

        image_scores = self.roi_heads.generate_image_scores(features, rpn_proposals, feature_refs)

        return [(score, gt['image_id']) for score, gt in zip(image_scores, batched_inputs)]

    def forward_al(self, batched_inputs, feature_refs, do_postprocess=True):

        """
        Run inference on the given inputs with active learning.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        
        with torch.no_grad():
            features, rpn_proposals = self._estimate_feature_proposal(batched_inputs)

        detection_results = self.roi_heads.generate_object_scores(features, rpn_proposals, feature_refs)
    
        if do_postprocess:
            detection_results = GeneralizedRCNN._postprocess(detection_results, batched_inputs, images.image_sizes) 
        
        return detection_results
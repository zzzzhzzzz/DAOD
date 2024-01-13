import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss
from torch.distributed import all_reduce
from torchvision.ops.boxes import nms
import math
from scipy.optimize import linear_sum_assignment

from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
from utils.distributed_utils import is_dist_avail_and_initialized, get_world_size
from collections import defaultdict


class HungarianMatcher(nn.Module):

    def __init__(self,
                 coef_class: float = 2,
                 coef_bbox: float = 5,
                 coef_giou: float = 2):
        super().__init__()
        self.coef_class = coef_class
        self.coef_bbox = coef_bbox
        self.coef_giou = coef_giou
        assert coef_class != 0 or coef_bbox != 0 or coef_giou != 0, "all costs cant be 0"

    def forward(self, pred_logits, pred_boxes, annotations):
        with torch.no_grad():
            bs, num_queries = pred_logits.shape[:2]
            # We flatten to compute the cost matrices in a batch
            pred_logits = pred_logits.flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_class]
            pred_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]
            gt_class = torch.cat([anno["labels"] for anno in annotations]).to(pred_logits.device)
            gt_boxes = torch.cat([anno["boxes"] for anno in annotations]).to(pred_logits.device)
            # Compute the classification cost.
            alpha, gamma = 0.25, 2.0
            neg_cost_class = (1 - alpha) * (pred_logits ** gamma) * (-(1 - pred_logits + 1e-8).log())
            pos_cost_class = alpha * ((1 - pred_logits) ** gamma) * (-(pred_logits + 1e-8).log())
            cost_class = pos_cost_class[:, gt_class] - neg_cost_class[:, gt_class]
            # Compute the L1 cost between boxes
            cost_boxes = torch.cdist(pred_boxes, gt_boxes, p=1)
            # Compute the giou cost between boxes
            cost_giou = - generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))
            # Final cost matrix
            cost = self.coef_bbox * cost_boxes + self.coef_class * cost_class + self.coef_giou * cost_giou
            cost = cost.view(bs, num_queries, -1).cpu()
            sizes = [len(anno["boxes"]) for anno in annotations]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):

    def __init__(self,
                 num_classes=9,
                 coef_class=2,
                 coef_boxes=5,
                 coef_giou=2,
                 coef_domain=1.0,
                 coef_domain_bac=0.3,
                 alpha_focal=0.25,
                 device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.matcher = HungarianMatcher()
        self.coef_class = coef_class
        self.coef_boxes = coef_boxes
        self.coef_giou = coef_giou
        self.coef_domain = coef_domain
        self.coef_domain_bac = coef_domain_bac
        self.alpha_focal = alpha_focal
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=device) for _ in range(num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=device) for _ in range(num_classes)]

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
        pass

    def loss_class(self, pred_logits, annotations, indices, num_boxes):
        pass

    def loss_boxes(self, pred_boxes, annotations, indices, num_boxes):
        pass

    def loss_giou(self, pred_boxes, annotations, indices, num_boxes):
        pass

    def loss_domains(self, out, domain_label):
        pass

    

    def forward(self, out, annotations=None, domain_label=None, enable_mae=False):
        # Implement here
        return loss, loss_dict


@torch.no_grad()
def post_process(pred_logits, pred_boxes, image_sizes, topk=100):
    assert len(pred_logits) == len(image_sizes)
    assert image_sizes.shape[1] == 2
    prob = pred_logits.sigmoid()
    prob = prob.view(pred_logits.shape[0], -1)
    topk_values, topk_indexes = torch.topk(prob, topk, dim=1)
    topk_boxes = torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='trunc')
    labels = topk_indexes % pred_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(pred_boxes)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    # From relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = image_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(topk_values, labels, boxes)]
    return results


def get_pseudo_labels(pred_logits, pred_boxes, thresholds, nms_threshold=0.7):
    probs = pred_logits.sigmoid()
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    pseudo_labels = []
    thresholds_tensor = torch.tensor(thresholds, device=pred_logits.device)
    for scores, labels, pred_box in zip(scores_batch, labels_batch, pred_boxes):
        larger_idx = torch.gt(scores, thresholds_tensor[labels]).nonzero()[:, 0]
        scores, labels, boxes = scores[larger_idx], labels[larger_idx], pred_box[larger_idx, :]
        nms_idx = nms(box_cxcywh_to_xyxy(boxes), scores, iou_threshold=nms_threshold)
        scores, labels, boxes = scores[nms_idx], labels[nms_idx], boxes[nms_idx, :]
        pseudo_labels.append({'scores': scores, 'labels': labels, 'boxes': boxes})
    return pseudo_labels

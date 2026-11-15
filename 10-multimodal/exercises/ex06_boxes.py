"""Exercise 06 - Box metrics for visual grounding: IoU, matching, average precision.

TASK
----
Boxes are (x1, y1, x2, y2) with x1 < x2 and y1 < y2. Implement (numpy and math only):

    box_area(b) -> float
        Area, or 0.0 for a degenerate box (x2 <= x1 or y2 <= y1).

    iou(a, b) -> float
        Intersection over union. 0.0 when the union is 0.

    denormalize_box(box, width, height, scale=1000) -> tuple[float, float, float, float]
        Many VLMs emit boxes in a 0..scale integer grid. Convert to pixels:
        x * width / scale and y * height / scale.

    iou_matrix(preds, gts) -> np.ndarray
        Shape (len(preds), len(gts)), entry [i][j] = iou(preds[i], gts[j]).

    match_predictions(preds, scores, gts, thr=0.5) -> list[bool]
        Process predictions in descending score order (ties: lower index first). Each one
        is matched to the unmatched ground-truth box with the highest IoU if that IoU is at
        least thr, and counts as a true positive; otherwise a false positive. Return the
        flags in the ORIGINAL prediction order.

    average_precision(preds, scores, gts, thr=0.5) -> float
        Single-class AP with all-point interpolation: sort by score, compute the cumulative
        precision and recall at each prediction, replace precision by its running maximum
        from the right (the envelope), and sum precision * (recall step) over recall steps.
        Return 0.0 if there are no ground-truth boxes.

    grounding_accuracy(pred_boxes, gt_boxes, thr=0.5) -> float
        One predicted box per query: fraction with IoU >= thr (the "Acc@0.5" of
        referring-expression benchmarks).

HINTS
-----
- Intersection width is max(0, min(x2) - max(x1)), the same for height.
- Greedy matching means a duplicate prediction of an already matched object is a false
  positive.
- Example: two ground truths and predictions (TP, FP, TP) in score order give precision
  1, 1/2, 2/3 and recall 1/2, 1/2, 1. AP = 0.5 * 1 + 0.5 * 2/3 = 5/6.

WHAT YOU SHOULD LEARN
---------------------
- How localisation is scored, and why a box can be "roughly right" and still fail at 0.5.
- Why coordinate conventions (normalised grids versus pixels) are a common silent bug.
- How precision and recall combine into AP when predictions are ranked by confidence.
"""
from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


def box_area(b: Box) -> float:
    raise NotImplementedError("TODO")


def iou(a: Box, b: Box) -> float:
    raise NotImplementedError("TODO")


def denormalize_box(box: Box, width: float, height: float, scale: float = 1000) -> Box:
    raise NotImplementedError("TODO")


def iou_matrix(preds: list[Box], gts: list[Box]) -> np.ndarray:
    raise NotImplementedError("TODO")


def match_predictions(preds: list[Box], scores: list[float], gts: list[Box], thr: float = 0.5) -> list[bool]:
    raise NotImplementedError("TODO")


def average_precision(preds: list[Box], scores: list[float], gts: list[Box], thr: float = 0.5) -> float:
    raise NotImplementedError("TODO")


def grounding_accuracy(pred_boxes: list[Box], gt_boxes: list[Box], thr: float = 0.5) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    a = (0, 0, 10, 10)
    assert box_area(a) == 100.0 and box_area((5, 5, 5, 9)) == 0.0 and box_area((6, 0, 2, 4)) == 0.0
    assert iou(a, a) == 1.0
    assert iou(a, (20, 20, 30, 30)) == 0.0                       # disjoint
    assert abs(iou(a, (5, 0, 15, 10)) - 50 / 150) < 1e-12        # half overlap
    assert abs(iou(a, (2, 2, 8, 8)) - 36 / 100) < 1e-12          # contained box
    assert iou(a, (10, 0, 20, 10)) == 0.0                        # touching edges
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0                # degenerate boxes
    assert abs(iou(a, (5, 0, 15, 10)) - iou((5, 0, 15, 10), a)) < 1e-15

    # shifting a 10x10 box by 2 px keeps IoU high, by 4 px already drops below 0.5
    assert abs(iou(a, (2, 0, 12, 10)) - 80 / 120) < 1e-12
    assert iou(a, (4, 0, 14, 10)) < 0.5 + 1e-9 and abs(iou(a, (4, 0, 14, 10)) - 60 / 140) < 1e-12

    # coordinate conventions: a 0..1000 grid on a 640 x 480 image
    assert denormalize_box((250, 500, 750, 1000), 640, 480) == (160.0, 240.0, 480.0, 480.0)
    # forgetting to denormalise gives IoU 0 against a pixel-space ground truth
    gt_px = (160.0, 240.0, 480.0, 480.0)
    assert iou((250, 500, 750, 1000), gt_px) < 0.5
    assert iou(denormalize_box((250, 500, 750, 1000), 640, 480), gt_px) == 1.0

    m = iou_matrix([a, (20, 20, 30, 30)], [a, (5, 0, 15, 10)])
    assert m.shape == (2, 2) and m[0, 0] == 1.0 and abs(m[0, 1] - 1 / 3) < 1e-12 and np.all(m[1] == 0)

    # matching: the second prediction duplicates an already claimed object
    gts = [(0, 0, 10, 10), (50, 50, 60, 60)]
    preds = [(0, 0, 10, 10), (1, 1, 10, 10), (50, 50, 60, 61)]
    scores = [0.9, 0.8, 0.7]
    assert match_predictions(preds, scores, gts) == [True, False, True]
    # the order of scores decides which duplicate wins
    assert match_predictions(preds, [0.8, 0.9, 0.7], gts) == [False, True, True]
    # threshold matters
    assert match_predictions([(0, 0, 10, 10)], [1.0], [(6, 0, 16, 10)], thr=0.5) == [False]
    assert match_predictions([(0, 0, 10, 10)], [1.0], [(6, 0, 16, 10)], thr=0.25) == [True]

    # AP: TP, FP, TP over 2 ground truths = 5/6
    assert abs(average_precision(preds, scores, gts) - 5 / 6) < 1e-12
    # perfect detector
    assert abs(average_precision([gts[0], gts[1]], [0.9, 0.8], gts) - 1.0) < 1e-12
    # a confident false positive ranked first lowers AP: FP, TP, TP -> precisions 0, 1/2, 2/3
    ap = average_precision([(80, 80, 90, 90), gts[0], gts[1]], [0.99, 0.9, 0.8], gts)
    assert abs(ap - (0.5 * (2 / 3) + 0.5 * (2 / 3))) < 1e-12
    assert average_precision([], [], gts) == 0.0
    assert average_precision(preds, scores, []) == 0.0
    # missing one object caps recall at 0.5 and so AP at 0.5
    assert abs(average_precision([gts[0]], [0.9], gts) - 0.5) < 1e-12

    # grounding accuracy over four queries: hits, near miss, miss, hit
    pb = [(0, 0, 10, 10), (0, 0, 10, 10), (30, 30, 40, 40), (5, 5, 15, 15)]
    gb = [(0, 0, 10, 10), (4, 0, 14, 10), (0, 0, 10, 10), (5, 5, 15, 15)]
    assert grounding_accuracy(pb, gb) == 0.5
    assert grounding_accuracy(pb, gb, thr=0.4) == 0.75


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")

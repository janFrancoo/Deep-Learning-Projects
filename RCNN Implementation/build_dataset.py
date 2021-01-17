import os
import cv2
from utils import config
from bs4 import BeautifulSoup
from utils.intersection_over_union import compute_iou

for dir_path in (config.POSITIVE_PATH, config.NEGATIVE_PATH):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

image_paths = [os.path.sep.join([config.ORIG_IMAGES, f]) for f in os.listdir(config.ORIG_IMAGES)]
total_positive = 0
total_negative = 0

for i, image_path in enumerate(image_paths):
    print("Processing image {}/{}...".format(i + 1, len(image_paths)))
    file_name = image_path.split(os.path.sep)[-1]
    file_name = file_name[:file_name.rfind(".")]
    annot_path = os.path.sep.join([config.ORIG_ANNOTS, "{}.xml".format(file_name)])

    contents = open(annot_path).read()
    soup = BeautifulSoup(contents, "html.parser")

    gt_boxes = []
    w = int(soup.find("width").string)
    h = int(soup.find("height").string)

    for o in soup.find_all("object"):
        x_min = max(0, int(o.find("xmin").string))
        y_min = max(0, int(o.find("ymin").string))
        x_max = min(w, int(o.find("xmax").string))
        y_max = min(h, int(o.find("ymax").string))
        gt_boxes.append((x_min, y_min, x_max, y_max))

    image = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposed_rects = []

    for x, y, w, h in rects:
        proposed_rects.append((x, y, x + w, y + h))

    positive_rois = 0
    negative_rois = 0

    for proposed_rect in proposed_rects[:config.MAX_PROPOSALS]:
        prop_start_x, prop_start_y, prop_end_x, prop_end_y = proposed_rect
        for gt_box in gt_boxes:
            roi = None
            output_path = None
            iou_val = compute_iou(gt_box, proposed_rect)

            if iou_val > 0.7 and positive_rois <= config.MAX_POSITIVE:
                roi = image[prop_start_y:prop_end_y, prop_start_x:prop_end_x]
                file_name = "{}.png".format(total_positive)
                output_path = os.path.sep.join([config.POSITIVE_PATH, file_name])
                positive_rois += 1
                total_positive += 1

            gt_start_x, gt_start_y, gt_end_x, gt_end_y = gt_box
            full_overlap = prop_start_x >= gt_start_x
            full_overlap = full_overlap and prop_start_y >= gt_start_y
            full_overlap = full_overlap and prop_end_x <= gt_end_x
            full_overlap = full_overlap and prop_end_y <= gt_end_y

            if not full_overlap and iou_val < 0.05 and negative_rois <= config.MAX_NEGATIVE:
                roi = image[prop_start_y:prop_end_y, prop_start_x:prop_end_x]
                file_name = "{}.png".format(total_negative)
                output_path = os.path.sep.join([config.NEGATIVE_PATH, file_name])
                negative_rois += 1
                total_negative += 1

            if roi is not None and output_path is not None:
                roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(output_path, roi)

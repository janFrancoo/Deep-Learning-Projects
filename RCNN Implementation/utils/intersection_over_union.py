def compute_iou(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_up = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_down = min(box_a[3], box_b[3])

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_down - y_up + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    return intersection_area / float(box_a_area + box_b_area - intersection_area)

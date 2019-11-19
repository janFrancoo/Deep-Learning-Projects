import cv2
import numpy as np

path = "C:/Users/ErenS/Desktop/cc/"


def order_points(pts):
    rect_coordinates = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect_coordinates[0] = pts[np.argmin(s)]
    rect_coordinates[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect_coordinates[1] = pts[np.argmin(diff)]
    rect_coordinates[3] = pts[np.argmax(diff)]

    return rect_coordinates


def four_point_transform(img, pts):

    rect_coordinates = order_points(pts)
    (tl, tr, br, bl) = rect_coordinates

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    m = cv2.getPerspectiveTransform(rect_coordinates, dst)
    warped = cv2.warpPerspective(img, m, (max_width, max_height))

    return warped


def doc_scan(image):

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (image.shape[0], 500))
    orig_height, orig_width = image.shape[:2]
    Original_Area = orig_height * orig_width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)

    _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        area = cv2.contourArea(c)
        if area < (Original_Area / 3):
            print("Error Image Invalid")
            return ("ERROR")
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    cv2.resize(warped, (640, 403), interpolation=cv2.INTER_AREA)
    cv2.imwrite("credit_card_color.jpg", warped)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = warped.astype("uint8") * 255
    cv2.imshow("Extracted Credit Card", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warped


image = cv2.imread(path + 'test_card.jpg')
image = doc_scan(image)

region = [(55, 210), (640, 290)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

roi = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
cv2.imshow("Region", roi)
cv2.imwrite(path + "credit_card_extracted_digits.jpg", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

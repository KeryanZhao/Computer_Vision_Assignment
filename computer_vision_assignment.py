import numpy as np
import imutils
import cv2 as cv
import argparse

class Sticher:
    def harries_corner_dector(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        dst = cv.dilate(dst, None)
        threshold = 0.01 * dst.max()
        corner_mask = dst > threshold
        corners = np.argwhere(corner_mask)
        for corner in corners:
            y, x = corner
            cv.circle(image, (x, y), 1, (0, 0, 255), -1)
        return corners, image

    def SIFT_points_dector(self, image):
        # 将图像转换为灰度图像
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 初始化SIFT检测器
        sift = cv.SIFT_create()
        # 使用SIFT检测关键点和描述符
        kp, des = sift.detectAndCompute(gray, None)
        # 绘制特征点
        image_with_keypoints = cv.drawKeypoints(image, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoints_info = [(kp.pt, kp.angle) for kp in kp]

        return image_with_keypoints, keypoints_info


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="path to the first image")
ap.add_argument("-s", "--second", required=True, help="path to the second image")
args = vars(ap.parse_args())

imageA = cv.imread(args["first"])
imageB = cv.imread(args["second"])
# imageA = imutils.resize(imageA, width=400)
# imageB = imutils.resize(imageB, width=400)
stitcher = Sticher()
imageA_corners, imageA_harried = stitcher.harries_corner_dector(imageA.copy())
imageB_corners, imageB_harried = stitcher.harries_corner_dector(imageB.copy())
imageA_SIFT, imageA_SIFT_info = stitcher.SIFT_points_dector(imageA.copy())
imageB_SIFT, imageB_SIFT_info = stitcher.SIFT_points_dector(imageB.copy())

cv.imshow("Image A", imageA_harried)
cv.imshow("Image B", imageB_harried)
cv.imshow("Image A SIFT", imageA_SIFT)
cv.imshow("Image B SIFT", imageB_SIFT)
cv.waitKey(0)
cv.destroyAllWindows()

# print("ImageA Harris Corners Locations:")
# for corner in imageA_corners:
#     y, x = corner
#     print(f"({x}, {y})")
#
# print("ImageB Harris Corners Locations:")
# for corner in imageB_corners:
#     y, x = corner
#     print(f"({x}, {y})")
# print("SIFT关键点坐标和方向（图像A）:")
# for pt, angle in imageB_SIFT_info:
#     print(f"location: ({pt[0]}, {pt[1]}), orientation: {angle}")
#
# print("SIFT关键点坐标和方向（图像B）:")
# for pt, angle in imageB_SIFT_info:
#     print(f"location: ({pt[0]}, {pt[1]}), orientation: {angle}")
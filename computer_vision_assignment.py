import numpy as np
import imutils
import cv2 as cv
# import argparse

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
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        # 绘制特征点
        image_with_keypoints = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoints_info = [(kp.pt, kp.angle) for kp in keypoints]

        return image_with_keypoints, keypoints, keypoints_info, descriptors

    def ORB_points_dector(self, image):
        # 将图像转换为灰度图像
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        image_with_keypoints = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return image_with_keypoints, keypoints, descriptors

    def compute_ssd(self, descriptor1, descriptor2):
        ssd = (np.linalg.norm(descriptor1 - descriptor2)) ** 2
        return ssd

    def match_descriptors_ssd(self, descriptors1, descriptors2):
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance**2)
        return matches[:800]

    def ratio_test(self, descriptors1, descriptors2):
        """
        应用比率测试筛选匹配对
        :param matches: 所有匹配对的索引和距离
        :param ratio: 比率测试阈值
        :return: 通过比率测试的匹配对
        """
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            m_ssd = m.distance ** 2
            n_ssd = n.distance ** 2
            if m_ssd < 0.75 * n_ssd:
                good_matches.append(m)
        return good_matches

    def stitch_images(self, images):
        (imageB, imageA) = images
        # 使用SIFT特征检测和描述
        _, keypointsA, _, descriptorsA = self.SIFT_points_dector(imageA)
        _, keypointsB, _, descriptorsB = self.SIFT_points_dector(imageB)

        # 使用比率测试进行特征点匹配
        matches = self.ratio_test(descriptorsA, descriptorsB)

        src_pts = np.float32([keypointsA[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypointsB[m.trainIdx].pt for m in matches])

        # 计算单应性矩阵
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # # 使用单应性矩阵变换图像
        # heightA, widthA = imageA.shape[:2]
        # heightB, widthB = imageB.shape[:2]
        #
        # # 计算变换后图像的尺寸
        # result_width = widthA + widthB
        # result_height = max(heightA, heightB)
        #
        # # 变换图像A
        # result = cv.warpPerspective(imageA, H, (result_width, result_height))
        #
        # # # 将图像B拷贝到结果图像中
        # # result[0:heightB, 0:widthB] = imageB
        result = cv.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        return result



# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True, help="path to the first image")
# ap.add_argument("-s", "--second", required=True, help="path to the second image")
# args = vars(ap.parse_args())
#
# imageA = cv.imread(args["first"])
# imageB = cv.imread(args["second"])
# # imageA = imutils.resize(imageA, width=400)
# # imageB = imutils.resize(imageB, width=400)
# stitcher = Sticher()
# imageA_corners, imageA_harried = stitcher.harries_corner_dector(imageA.copy())
# imageB_corners, imageB_harried = stitcher.harries_corner_dector(imageB.copy())
# imageA_SIFT, imageA_SIFT_keypoints, imageA_SIFT_info, imageA_SIFT_descriptors = stitcher.SIFT_points_dector(imageA.copy())
# imageB_SIFT, imageB_SIFT_keypoints, imageB_SIFT_info, imageB_SIFT_descriptors = stitcher.SIFT_points_dector(imageB.copy())
# imageA_ORB, imageA_ORB_keypoints, imageA_ORB_descriptors = stitcher.ORB_points_dector(imageA.copy())
# imageB_ORB, imageB_ORB_keypoints, imageB_ORB_descriptors = stitcher.ORB_points_dector(imageB.copy())
# SIFT_ssd_matches = stitcher.match_descriptors_ssd(imageA_SIFT_descriptors, imageB_SIFT_descriptors)
# SIFT_ssd_match_img = cv.drawMatches(imageA, imageA_SIFT_keypoints, imageB, imageB_SIFT_keypoints, SIFT_ssd_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# SIFT_ratio_matches = stitcher.ratio_test(imageA_SIFT_descriptors, imageB_SIFT_descriptors)
# SIFT_ratio_match_img = cv.drawMatches(imageA, imageA_SIFT_keypoints, imageB, imageB_SIFT_keypoints, SIFT_ratio_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# stitched_image = stitcher.stitch_images([imageA, imageB])
# cv.imshow("Image A", imageA_harried)
# cv.imshow("Image B", imageB_harried)
# cv.imshow("Image A SIFT", imageA_SIFT)
# cv.imshow("Image B SIFT", imageB_SIFT)
# cv.imshow("Image A ORB", imageA_ORB)
# cv.imshow("SIFT_ssd_match_img", SIFT_ssd_match_img)
# cv.imshow("SIFT_ratio_match_img", SIFT_ratio_match_img)
# cv.imshow("stitched_image ", stitched_image)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # print("ImageA Harris Corners Locations:")
# # for corner in imageA_corners:
# #     y, x = corner
# #     print(f"({x}, {y})")
# #
# # print("ImageB Harris Corners Locations:")
# # for corner in imageB_corners:
# #     y, x = corner
# #     print(f"({x}, {y})")
# # print("SIFT关键点坐标和方向（图像A）:")
# # for pt, angle in imageB_SIFT_info:
# #     print(f"location: ({pt[0]}, {pt[1]}), orientation: {angle}")
# #
# # print("SIFT关键点坐标和方向（图像B）:")
# # for pt, angle in imageB_SIFT_info:
# #     print(f"location: ({pt[0]}, {pt[1]}), orientation: {angle}")
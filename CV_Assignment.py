# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# # filename = 'left.png'
# # img = cv.imread(filename)
# # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # gray = np.float32(gray)
# # dst = cv.cornerHarris(gray, 2, 3, 0.04)
# #
# # # result is dilated for marking the corners, not important
# # dst = cv.dilate(dst, None)
# #
# # # Threshold for an optimal value, it may vary depending on the image.
# # img[dst > 0.01 * dst.max()] = [0, 0, 255]
# #
# # cv.imshow('dst', img)
# # if cv.waitKey(0) & 0xff == 27:
# #     cv.destroyAllWindows()
#
# # 使用SIFT检测特征点并计算描述符
# # sift = cv.SIFT_create()
# # kp = sift.detect(gray, None)
# # kp,des = sift.compute(gray,kp)
# # print(des[0])
#
# #
# # img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# #
# # cv.imwrite('sift_keypoints.jpg', img)
#
#
# # 使用ORB描述符进行特征检测和描述
# # 初始化ORB检测器
# # orb = cv.ORB_create()
# #
# # # 检测特征点并计算描述符
# # keypoints, descriptors = orb.detectAndCompute(gray, None)
# #
# # # 绘制特征点
# # image_with_keypoints = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# #
# # # 显示图像
# # cv.imshow('ORB Keypoints', image_with_keypoints)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
# img1 = cv.imread('left.png')  # queryImage
# img2 = cv.imread('right.png')  # trainImage
#
# # Initiate SIFT detector
# sift = cv.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# # BFMatcher with default params
# bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
# # matches1 = bf.knnMatch(des1, des2, k = 1)
# matches2 = bf.knnMatch(des1, des2, k=2)
#
# # Apply ratio test
# good = []
# for m, n in matches2:
#     m_ssd = m.distance ** 2
#     n_ssd = n.distance ** 2
#     if m_ssd < 0.1 * n_ssd:
#         good.append([m])
#
# # cv.drawMatchesKnn expects list of lists as matches.
# # img2 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches1, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
# # 保存匹配结果图片
# cv.imwrite('ssd_ratio_matched_result.png', img3)
# # cv.imwrite('ssd_matched_result.png', img2)
#
#
# # def compute_ssd(descriptor1, descriptor2):
# #     """
# #     计算两个描述符向量之间的SSD距离
# #     :param descriptor1: 第一个描述符向量
# #     :param descriptor2: 第二个描述符向量
# #     :return: SSD距离
# #     """
# #     return np.sum((descriptor1 - descriptor2) ** 2)
# #
# # def match_descriptors_ssd(descriptors1, descriptors2):
# #     """
# #     使用SSD距离匹配两个描述符集合
# #     :param descriptors1: 第一个描述符集合
# #     :param descriptors2: 第二个描述符集合
# #     :return: 所有匹配对的索引和距离
# #     """
# #     matches = []
# #     for i, desc1 in enumerate(descriptors1):
# #         min_distance = float('inf')
# #         best_match_index = -1
# #         for j, desc2 in enumerate(descriptors2):
# #             distance = compute_ssd(desc1, desc2)
# #             if distance < min_distance:
# #                 min_distance = distance
# #                 best_match_index = j
# #         matches.append(cv.DMatch(i, best_match_index, min_distance))
# #     return matches
# #
# # def ratio_test(matches, ratio=0.75):
# #     """
# #     应用比率测试筛选匹配对
# #     :param matches: 所有匹配对的索引和距离
# #     :param ratio: 比率测试阈值
# #     :return: 通过比率测试的匹配对
# #     """
# #     good_matches = []
# #     for match in matches:
# #         i, best_match_index, best_distance, second_best_distance = match
# #         if best_distance < ratio * second_best_distance:
# #             good_matches.append(cv.DMatch(i, best_match_index, best_distance))
# #     return good_matches
# #
# # # 读取图像并转换为灰度图像
# # img1 = cv.imread('left.png')
# # img2 = cv.imread('right.png')
# #
# # # 初始化SIFT检测器
# # sift = cv.SIFT_create()
# #
# # # 检测特征点和计算描述符
# # kp1, des1 = sift.detectAndCompute(img1, None)
# # kp2, des2 = sift.detectAndCompute(img2, None)
# #
# # # 使用SSD匹配描述符
# # matches = match_descriptors_ssd(des1, des2)
# #
# # # 应用比率测试筛选匹配对
# # good_matches = ratio_test(matches)
# #
# # # 绘制匹配结果
# # img_matches1 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# # img_matches2 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# #
# # # 保存匹配结果图片
# # cv.imwrite('ssd_matches.png', img_matches1)
# # cv.imwrite('ssd_ratio_matches.png', img_matches2)
#
# import the necessary packages

import numpy as np
import imutils
import cv2

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homography matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homography could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

# 从命令行参数获取图像路径
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="path to the first image")
ap.add_argument("-s", "--second", required=True, help="path to the second image")
args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# filename = 'left.png'
# img = cv.imread(filename)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# gray = np.float32(gray)
# dst = cv.cornerHarris(gray, 2, 3, 0.04)
#
# # result is dilated for marking the corners, not important
# dst = cv.dilate(dst, None)
#
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst > 0.01 * dst.max()] = [0, 0, 255]
#
# cv.imshow('dst', img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()

# 使用SIFT检测特征点并计算描述符
# sift = cv.SIFT_create()
# kp = sift.detect(gray, None)
# kp,des = sift.compute(gray,kp)
# print(des[0])
#
# img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# cv.imwrite('sift_keypoints.jpg', img)


# 使用ORB描述符进行特征检测和描述
# 初始化ORB检测器
# orb = cv.ORB_create()
#
# # 检测特征点并计算描述符
# keypoints, descriptors = orb.detectAndCompute(gray, None)
#
# # 绘制特征点
# image_with_keypoints = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # 显示图像
# cv.imshow('ORB Keypoints', image_with_keypoints)
# cv.waitKey(0)
# cv.destroyAllWindows()

img1 = cv.imread('left.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('right.png', cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3), plt.show()
import numpy as np
import cv2 as cv
import time
class Sticher:

    def timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
            return result

        return wrapper
    def harries_corner_dector(self, image):
        '''
        :param image: input image
        :return: Coordinates of corner points and images of labelled corner points
        '''
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        dst = cv.dilate(dst, None) # Make corner points more visible
        threshold = 0.01 * dst.max() # set threshold
        corner_mask = dst > threshold
        corners = np.argwhere(corner_mask) # Get the coordinates of the corner point
        for corner in corners: # Plotting corner points on the image
            y, x = corner
            cv.circle(image, (x, y), 1, (0, 0, 255), -1)
        return corners, image

    def SIFT_points_dector(self, image):
        # Converting images to greyscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        # Detecting keypoints and descriptors using SIFT
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        image_with_keypoints = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoints_info = [(kp.pt, kp.angle) for kp in keypoints]

        return image_with_keypoints, keypoints, keypoints_info, descriptors

    def ORB_points_dector(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Initialise the ORB detector
        orb = cv.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        image_with_keypoints = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return image_with_keypoints, keypoints, descriptors

    def compute_ssd(self, descriptor1, descriptor2):
        ssd = np.sum((descriptor1 - descriptor2) ** 2)
        return ssd

    @timer
    def match_descriptors_ssd(self, descriptors1, descriptors2, ORB = False):
        '''
        :param descriptors1: descriptors of left image
        :param descriptors2: descriptors of right image
        :return:
        '''
        if ORB:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            return matches[:800]

        else:
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            return matches[:800]
            # matches = []
            # for i, desc1 in enumerate(descriptors1):
            #     best_match = None
            #     min_distance = float('inf')
            #     for j, desc2 in enumerate(descriptors2):
            #         distance = self.compute_ssd(desc1, desc2)
            #         if distance < min_distance:
            #             min_distance = distance
            #             best_match = cv.DMatch(i, j, distance)
            #     if best_match is not None:
            #         matches.append(best_match)
            # return matches
    @timer
    def ratio_test(self, descriptors1, descriptors2, ORB = False):
        """
        Apply ratio tests to filter matching pairs
        :param descriptors1: descriptors of left image
        :param descriptors2: descriptors of right image
        :return Matches that pass the ratio test
        """
        if ORB:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            return good_matches[:800]

        else:
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                m_ssd = m.distance ** 2
                n_ssd = n.distance ** 2
                if m_ssd < 0.75 * n_ssd:
                    good_matches.append(m)
            return good_matches[:800]

    def stitch_images(self, images):
        (imageB, imageA) = images
        # Detection and description using SIFT features
        _, keypointsA, _, descriptorsA = self.SIFT_points_dector(imageA)
        _, keypointsB, _, descriptorsB = self.SIFT_points_dector(imageB)

        # Feature point matching using ratio tests
        matches = self.ratio_test(descriptorsA, descriptorsB)

        src_pts = np.float32([keypointsA[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypointsB[m.trainIdx].pt for m in matches])

        # Calculate the homography
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # Perform a perspective transformation on an image using the homography
        # and stitch the two images together
        result = cv.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        return result


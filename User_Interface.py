import tkinter as tk
from tkinter import filedialog, Toplevel, Text
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
from computer_vision_assignment import Sticher


class ImageUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Uploader")

        # Set window size
        self.root.geometry("800x494")  # width x Height

        self.sticher = Sticher()

        self.left_image_label = tk.Label(root, text="Left Image: No image uploaded")
        self.left_image_label.pack(pady=10)

        left_button_frame = tk.Frame(root)
        left_button_frame.pack(pady=5)
        self.left_upload_button = tk.Button(left_button_frame, text="Upload Left Image", command=self.upload_left_image)
        self.left_upload_button.pack(side=tk.LEFT, padx=5)
        self.left_harris_button = tk.Button(left_button_frame, text="Process Left Image Harris", command=self.process_left_image_harris)
        self.left_harris_button.pack(side=tk.LEFT, padx=5)
        self.left_sift_button = tk.Button(left_button_frame, text="Process Left Image SIFT", command=self.process_left_image_sift)
        self.left_sift_button.pack(side=tk.LEFT, padx=5)

        self.right_image_label = tk.Label(root, text="Right Image: No image uploaded")
        self.right_image_label.pack(pady=10)

        right_button_frame = tk.Frame(root)
        right_button_frame.pack(pady=5)
        self.right_upload_button = tk.Button(right_button_frame, text="Upload Right Image", command=self.upload_right_image)
        self.right_upload_button.pack(side=tk.LEFT, padx=5)
        self.right_harris_button = tk.Button(right_button_frame, text="Process Right Image Harris", command=self.process_right_image_harris)
        self.right_harris_button.pack(side=tk.LEFT, padx=5)
        self.right_sift_button = tk.Button(right_button_frame, text="Process Right Image SIFT", command=self.process_right_image_sift)
        self.right_sift_button.pack(side=tk.LEFT, padx=5)

        # self.right_image_label = tk.Label(root, text="")
        # self.right_image_label.pack(pady=20)
        tk.Label(root).pack(pady=10)

        matches_button_frame = tk.Frame(root)
        matches_button_frame.pack(pady=5)
        self.sift_ssd_button = tk.Button(matches_button_frame, text="SIFT_SSD_Matches", command=self.process_sift_ssd_matches)
        self.sift_ssd_button.pack(side=tk.LEFT,pady=5)
        self.sift_ratio_button = tk.Button(matches_button_frame, text="SIFT_Ratio_Matches", command=self.process_sift_ratio_matches)
        self.sift_ratio_button.pack(side=tk.LEFT,pady=5)
        self.orb_ssd_button = tk.Button(matches_button_frame, text="ORB_Matches", command=self.process_orb_ssd_matches)
        self.orb_ssd_button.pack(side=tk.LEFT,pady=5)
        self.orb_ratio_button = tk.Button(matches_button_frame, text="ORB_Ratio_Matches", command=self.process_orb_ratio_matches)
        self.orb_ratio_button.pack(side=tk.LEFT,pady=5)

        self.stitch_button = tk.Button(root, text="Stitch Images", command=self.stitch_images)
        self.stitch_button.pack(pady=5)

        self.left_image_path = None
        self.right_image_path = None

    def upload_left_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.left_image_path = file_path
            self.left_image_label.config(text=f"Left Image: {file_path}")

    def upload_right_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.right_image_path = file_path
            self.right_image_label.config(text=f"Right Image: {file_path}")

    def process_left_image_harris(self):
        if self.left_image_path:
            image = cv.imread(self.left_image_path)
            corners, processed_image = self.sticher.harries_corner_dector(image.copy())
            self.display_processed_image(processed_image, "Processed Left Image")
            self.display_corners(corners, "Left Image Corners")

    def process_right_image_harris(self):
        if self.right_image_path:
            image = cv.imread(self.right_image_path)
            corners, processed_image = self.sticher.harries_corner_dector(image.copy())
            self.display_processed_image(processed_image, "Processed Right Image")
            self.display_corners(corners, "Right Image Corners")

    def process_left_image_sift(self):
        if self.left_image_path:
            image = cv.imread(self.left_image_path)
            image_with_keypoints, keypoints, keypoints_info, _ = self.sticher.SIFT_points_dector(image.copy())
            self.display_processed_image(image_with_keypoints, "Processed Left Image SIFT")
            self.display_keypoints(keypoints_info, "Left Image SIFT Keypoints")

    def process_right_image_sift(self):
        if self.right_image_path:
            image = cv.imread(self.right_image_path)
            image_with_keypoints, keypoints, keypoints_info, _ = self.sticher.SIFT_points_dector(image.copy())
            self.display_processed_image(image_with_keypoints, "Processed Right Image SIFT")
            self.display_keypoints(keypoints_info, "Right Image SIFT Keypoints")

    def process_sift_ssd_matches(self):
        if self.left_image_path and self.right_image_path:
            image_left = cv.imread(self.left_image_path)
            image_right = cv.imread(self.right_image_path)
            _, keypoints_left, _, descriptors_left = self.sticher.SIFT_points_dector(image_left.copy())
            _, keypoints_right, _, descriptors_right = self.sticher.SIFT_points_dector(image_right.copy())
            matches = self.sticher.match_descriptors_ssd(descriptors_left, descriptors_right, ORB=False)

            matched_image = cv.drawMatches(image_left.copy(), keypoints_left, image_right.copy(), keypoints_right, matches, None,
                                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.display_processed_image(matched_image, "SIFT SSD Matches")

    def process_sift_ratio_matches(self):
        if self.left_image_path and self.right_image_path:
            image_left = cv.imread(self.left_image_path)
            image_right = cv.imread(self.right_image_path)
            _, keypoints_left, _, descriptors_left = self.sticher.SIFT_points_dector(image_left.copy())
            _, keypoints_right, _, descriptors_right = self.sticher.SIFT_points_dector(image_right.copy())
            matches = self.sticher.ratio_test(descriptors_left, descriptors_right, ORB=False)

            matched_image = cv.drawMatches(image_left.copy(), keypoints_left, image_right.copy(), keypoints_right, matches, None,
                                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.display_processed_image(matched_image, "SIFT Ratio Matches")

    def process_orb_ssd_matches(self):
        if self.left_image_path and self.right_image_path:
            image_left = cv.imread(self.left_image_path)
            image_right = cv.imread(self.right_image_path)
            _, keypoints_left, descriptors_left = self.sticher.ORB_points_dector(image_left.copy())
            _, keypoints_right, descriptors_right = self.sticher.ORB_points_dector(image_right.copy())
            matches = self.sticher.match_descriptors_ssd(descriptors_left, descriptors_right, ORB = True)

            matched_image = cv.drawMatches(image_left.copy(), keypoints_left, image_right.copy(), keypoints_right, matches, None,
                                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.display_processed_image(matched_image, "ORB Matches")

    def process_orb_ratio_matches(self):
        if self.left_image_path and self.right_image_path:
            image_left = cv.imread(self.left_image_path)
            image_right = cv.imread(self.right_image_path)
            _, keypoints_left, descriptors_left = self.sticher.ORB_points_dector(image_left.copy())
            _, keypoints_right, descriptors_right = self.sticher.ORB_points_dector(image_right.copy())
            matches = self.sticher.ratio_test(descriptors_left, descriptors_right, ORB=True)

            matched_image = cv.drawMatches(image_left.copy(), keypoints_left, image_right.copy(), keypoints_right,
                                           matches, None,
                                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.display_processed_image(matched_image, "ORB Ratio Matches")

    def stitch_images(self):
        if self.left_image_path and self.right_image_path:
            image_left = cv.imread(self.left_image_path)
            image_right = cv.imread(self.right_image_path)
            stitched_image = self.sticher.stitch_images([image_left.copy(), image_right.copy()])
            self.display_processed_image(stitched_image, "Stitched Image")

    def display_processed_image(self, image, title):
        new_window = Toplevel(self.root)
        new_window.title(title)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((1000, 1000))
        photo = ImageTk.PhotoImage(image)

        label = tk.Label(new_window, image=photo)
        label.image = photo  # Keep a reference to avoid garbage collection
        label.pack()

    def display_corners(self, corners, title):
        new_window = Toplevel(self.root)
        new_window.title(title)

        text_widget = Text(new_window, wrap='word')
        corners_text = "\n".join([f"({x}, {y})" for y, x in corners])
        text_widget.insert('1.0', f"Corners:\n{corners_text}")
        text_widget.pack(expand=True, fill='both')

    def display_keypoints(self, keypoints_info, title):
        new_window = Toplevel(self.root)
        new_window.title(title)

        text_widget = Text(new_window, wrap='word')
        keypoints_text = "\n".join([f"Position: ({pt[0]:.2f}, {pt[1]:.2f}), Angle: {angle:.2f}" for pt, angle in keypoints_info])
        text_widget.insert('1.0', f"Keypoints:\n{keypoints_text}")
        text_widget.pack(expand=True, fill='both')


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()

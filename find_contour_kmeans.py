import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np

# Use K-means method to cluster pixel value and do segmentation
class kmeanskernal():
    def __init__(self, window, window_title, image_path="Biden.jpg"):  # image location
        # Set up window interface
        self.window = window
        self.window.title(window_title)

        # Load an image with OpenCV
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.Newcv_img = self.cv_img.copy()

        # Get the image dimensions
        self.height, self.width, no_channels = self.cv_img.shape

        # Create a frame that can fit the original and new image
        self.frame1 = tk.Frame(self.window, width=self.width, height=self.height, bg='blue')
        self.frame1.pack()

        # Create canvases for original(LEFT) and new(RIGHT) image
        self.canvas0 = tk.Canvas(self.frame1, width=self.width, height=self.height * 1.2, bg='yellow')
        self.canvas0.pack(side=tk.LEFT)
        self.canvas1 = tk.Canvas(self.frame1, width=self.width, height=self.height * 1.2, bg='orange')
        self.canvas1.pack(side=tk.LEFT)

        # Use PIL to convert the NumPy ndarray to a "PhotoImage"
        self.photoOG = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))

        # Add a PhotoImage to the Canvas
        self.canvas0.create_image(self.width // 2, self.height // 2, image=self.photoOG)
        self.canvas1.create_image(self.width // 2, self.height // 2, image=self.photo, anchor=tk.CENTER)

        # Add descriptions to the Canvas for each image
        self.canvas0.create_text(self.width // 2, self.height * 1.1, text="Original Photo")
        self.canvas1.create_text(self.width // 2, self.height * 1.1, text="New Photo")

        # Create new frame at the bottom to set up slide bar and button
        self.frame2 = tk.Frame(self.window, width=self.width // 2, height=self.height)
        self.frame2.pack(side=tk.BOTTOM, fill=tk.BOTH)

        # Create slide bar to let users change the number of clusters
        self.scl_cluster = tk.Scale(window, from_=2, to=20, orient=tk.HORIZONTAL, showvalue=1,
                                    command=self.kcluster, length=500, label="Number of cluster")
        self.scl_cluster.pack(side=tk.LEFT, expand=True)

        # cluster parameter
        self.k = self.scl_cluster.get()

        # Creat buttons for kmeans, Canny, Contour functions
        self.button_k = tk.Button(self.frame2, text="Kmeans", width=15, command=self.kcluster)
        self.button_k.grid(row=0, column=0, padx=5)
        self.button_D = tk.Button(self.frame2, text="Canny", width=15, command=self.Canny)
        self.button_D.grid(row=0, column=1, padx=5)

        self.button_C = tk.Button(self.frame2, text="Initial Contour", width=15, command=self.iniContour)
        self.button_C.grid(row=0, column=3, padx=5)
        self.button_C = tk.Button(self.frame2, text="Largest Contour", width=15, command=self.bigContour)
        self.button_C.grid(row=1, column=3, padx=5)

        # Create button for closing window
        self.button_Q = tk.Button(self.frame2, text="Quit", width=30, command=window.destroy)
        self.button_Q.grid(row=2, column=1, pady=25)

        self.window.mainloop()

    # K-Means cluster functions
    def kcluster(self):
        # set up how many clusters
        k = self.scl_cluster.get()
        # Get image we would like to modified
        originalImage = self.cv_img
        #  Reshape image into a Nx3 shape, where N is the w*h product, and 3 is for the 3 colors.
        newImage = np.float32(self.Newcv_img.reshape(-1, 3))

        # kmeans clustering algorithm: return us a list with the centroids and a list with all the pixels
        # cv2.TERM_CRITERIA_EPS = stop the algorithm iteration if specified accuracy, epsilon, is reached.
        # cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
        # max_iter - An integer specifying maximum number of iterations.
        # epsilon - Required accuracy
        max_iter, epsilon = 5, 1
        Criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

        # flags : This flag is used to specify how initial centers are taken.
        # Normally two flags are used for this : cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.
        flags = cv2.KMEANS_RANDOM_CENTERS
        ret, labels, clusters = cv2.kmeans(newImage, k, None, Criteria, 10, flags)

        # transfer the clusters to unit8
        clusters = np.uint8(clusters)
        # Flatten to one dimension array and then reshape to current to original shape for showing
        clusteredImage = clusters[labels.flatten()].reshape(originalImage.shape)

        # Put image to right side and show it
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(clusteredImage))
        self.canvas1.create_image(self.width // 2, self.height // 2, image=self.photo, anchor=tk.CENTER)

    def Canny(self):
        k = self.scl_cluster.get()
        #  Reshape image into a Nx3 shape, where N is the w*h product, and 3 is for the 3 colors.
        newImage = np.float32(self.Newcv_img.reshape(-1, 3))

        # kmeans clustering algorithm: return us a list with the centroids and a list with all the pixels
        # cv2.TERM_CRITERIA_EPS = stop the algorithm iteration if specified accuracy, epsilon, is reached.
        # cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
        # max_iter - An integer specifying maximum number of iterations.
        # epsilon - Required accuracy
        max_iter, epsilon = 5, 1
        Criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

        # flags : This flag is used to specify how initial centers are taken.
        # Normally two flags are used for this : cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.
        flags = cv2.KMEANS_RANDOM_CENTERS
        ret, labels, clusters = cv2.kmeans(newImage, k, None, Criteria, 10, flags)

        # Remove 1 cluster from image(make them perfectly black) and apply canny edge detection
        removedCluster = 1
        originalImage = self.cv_img
        cannyImage = np.copy(originalImage).reshape((-1, 3))
        cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]

        # In cv2.Canny,
        # Second argument: min value. Third argument: max value
        # Third argument is aperture_size.It is the size of Sobel kernel used for find image gradients.
        minV, maxV = 0,200
        cannyImage = cv2.Canny(cannyImage, minV, maxV).reshape(originalImage.shape)

        # Put image to right side and show it
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cannyImage))
        self.canvas1.create_image(self.width // 2, self.height // 2, image=self.photo, anchor=tk.CENTER)

    def iniContour(self):
        k = self.scl_cluster.get()
        reshapedImage = np.float32(self.Newcv_img.reshape(-1, 3))
        Criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flag = cv2.KMEANS_RANDOM_CENTERS
        ret, labels, clusters = cv2.kmeans(reshapedImage, k, None, Criteria, 10, flag)
        removedCluster = 1
        originalImage = self.cv_img
        cannyImage = np.copy(originalImage).reshape((-1, 3))
        cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]
        cannyImage = cv2.Canny(cannyImage, 0, 200).reshape(originalImage.shape)

        initialContoursImage = np.copy(cannyImage)
        imgray = cv2.cvtColor(initialContoursImage, cv2.COLOR_BGR2GRAY)

        # If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value.
        # Use cv2.threshold to get two output
        # The first is the threshold that was used and the second output is the thresholded image.
        minV, maxV = 100, 255
        flag = cv2.THRESH_BINARY
        _, thresh = cv2.threshold(imgray, minV, maxV, flag)

        # Return contours: a Python list of all the contours in the image
        retrievalMode = cv2.RETR_TREE
        flagF = cv2.CHAIN_APPROX_SIMPLE
        contours, hierarchy = cv2.findContours(thresh, retrievalMode, flagF)

        # Draw "all" contours on the image: contours = -1
        contourColor = (0, 0, 255)
        contoursNum = -1
        cv2.drawContours(initialContoursImage, contours, contoursNum, contourColor, flagF)

        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(initialContoursImage))
        self.canvas1.create_image(self.width // 2, self.height // 2, image=self.photo, anchor=tk.CENTER)

    def bigContour(self):
        reshapedImage = np.float32(self.Newcv_img.reshape(-1, 3))
        Criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        ret, labels, clusters = cv2.kmeans(reshapedImage, self.k, None, Criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        removedCluster = 1
        originalImage = self.cv_img
        cannyImage = np.copy(originalImage).reshape((-1, 3))
        cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]
        cannyImage = cv2.Canny(cannyImage, 0, 200).reshape(originalImage.shape)

        initialContoursImage = np.copy(cannyImage)
        imgray = cv2.cvtColor(initialContoursImage, cv2.COLOR_BGR2GRAY)

        flag = cv2.THRESH_BINARY
        minV, maxV = 100, 255
        _, thresh = cv2.threshold(imgray, minV, maxV, flag)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = self.contour(contours)

        biggestContourImage = np.copy(originalImage)
        contourColor = (255, 100, 100)
        thickness = 2
        cv2.drawContours(biggestContourImage, [cnt], -1, contourColor, thickness)

        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(biggestContourImage))
        self.canvas1.create_image(self.width // 2, self.height // 2, image=self.photo, anchor=tk.CENTER)

    # Find contour in the image in a specific condition(or different cluster value)
    def contour(self, contours):
        cnt = contours[0]
        largest_area = 0
        index = 0
        for contour in contours:
            if index > 0:
                area = cv2.contourArea(contour)
                if area > largest_area:
                    largest_area = area
                    cnt = contours[index]
            index = index + 1
        return cnt



kmeanskernal(tk.Tk(), "Image Segmentation")

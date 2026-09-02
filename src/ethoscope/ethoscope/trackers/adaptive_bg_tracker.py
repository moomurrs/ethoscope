import logging
from collections import deque
from math import e as E
from math import exp, log, log10, pi, sqrt

import cv2
import numpy as np

from ethoscope.core.data_point import DataPoint
from ethoscope.core.variables import (
    HeightVariable,
    PhiVariable,
    WidthVariable,
    XPosVariable,
    XYDistance,
    YPosVariable,
)
from ethoscope.trackers.trackers import BaseTracker, NoPositionError

CV_VERSION = cv2.getVersionMajor()
_SQRT_2_PI = sqrt(2.0 * pi)
# log10(e) for stable log-space Gaussian (exp -> log10 without underflow)
_LOG10_E = log10(E)


class ObjectModel:
    """
    A class to model, update and predict foreground object (i.e. tracked animal).
    """

    def __init__(self, history_length=1000, ready_threshold=1000):
        # fixme this should be time, not number of points!
        self._features_header = [
            "fg_model_area",
            "fg_model_height",
            # "fg_model_aspect_ratio",
            "fg_model_mean_grey",
        ]

        self._history_length = history_length
        self._ready_threshold = min(int(ready_threshold), int(history_length))
        self._n_samples = 0
        self._ring_buff = np.zeros(
            (self._history_length, len(self._features_header)),
            dtype=np.float32,
            order="F",
        )
        self._std_buff = np.zeros(
            (self._history_length, len(self._features_header)),
            dtype=np.float32,
            order="F",
        )
        self._ring_buff_idx = 0
        self._is_ready = False
        self._roi_img_buff = None
        self._mask_img_buff = None
        self._img_buff_shape = np.array([0, 0])

        self._last_updated_time = 0
        # If the model is not updated for this duration, it is reset. Patches #39
        self._max_unupdated_duration = 1 * 60 * 1000.0  # ms

    @property
    def is_ready(self):
        return self._n_samples >= self._ready_threshold

    @property
    def features_header(self):
        return self._features_header

    def update(self, img, contour, time):
        features = self.compute_features(img, contour)
        self._last_updated_time = time
        self._ring_buff[self._ring_buff_idx] = features

        self._ring_buff_idx += 1
        self._n_samples = min(self._n_samples + 1, self._history_length)

        if self._ring_buff_idx == self._history_length:
            self._ring_buff_idx = 0

        # keep legacy flag in sync for debugging / external checks
        self._is_ready = self._n_samples >= self._ready_threshold

        return features

    def distance(self, features, time):
        if time - self._last_updated_time > self._max_unupdated_duration:
            logging.warning("FG model not updated for too long. Resetting.")
            self.__init__(self._history_length, self._ready_threshold)
            return 0

        if self._n_samples == 0:
            return 0

        # use actual valid samples, not full buffer, until buffer wraps
        last_row = self._n_samples

        means = np.mean(self._ring_buff[:last_row], 0)

        # True std (not MAD). Previous code used mean(abs(x-mean)) which
        # underestimates sigma by ~20% (sigma ~= 1.253*MAD for Gaussian).
        # Use population std (ddof=0); floor below handles n=1 case.
        stds = np.std(self._ring_buff[:last_row], axis=0)
        # Per-feature std floor to avoid over-tight model in uniform
        # bright ROIs rejecting slow moves (regression after per-ROI split).
        # Values chosen as ~0.5* typical std for each feature:
        # area log10 ~0.1, height ~1.0, mean_grey ~5.0. Prevents
        # likelihood explosion when ROI variance is naturally low.
        _std_floor = np.array([0.2, 1.0, 4.0], dtype=np.float32)
        # align floor length to feature count for forward-compat
        if len(_std_floor) != len(stds):
            _std_floor = np.resize(_std_floor, stds.shape)
        stds = np.maximum(stds, _std_floor)
        # Guard NaN/Inf (previous dead `stds==0` after floor is now removed)
        if np.any(np.isnan(stds)) or np.any(np.isinf(stds)):
            return float("inf")

        # Stable log-space Gaussian without exp->log10 roundtrip.
        # Previously: a=1/(std*sqrt2pi), b=exp(-z^2/2), log10(a*b)
        # Underflow: b==0 -> likelihood==0 incorrectly returned 0 (accept).
        # Now: log10(p) = -log10(std*sqrt2pi) - 0.5*z^2*log10(e)
        log10_std_term = np.log10(stds * _SQRT_2_PI)
        z = (features - means) / stds
        log10_likelihoods = -log10_std_term - 0.5 * (z * z) * _LOG10_E

        if np.any(np.isneginf(log10_likelihoods)) or np.any(
            np.isnan(log10_likelihoods)
        ):
            return float("inf")
        # print features, means
        logls = np.mean(log10_likelihoods)
        return -logls

    def compute_features(self, img, contour):
        x, y, w, h = cv2.boundingRect(contour)

        # Ensure bounding rectangle stays within image bounds
        img_height, img_width = img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)

        # Validate that we have a valid region after boundary clipping
        if w <= 0 or h <= 0:
            # Return default features when no valid region exists
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if self._roi_img_buff is None or self._roi_img_buff.shape[:2] < img.shape[0:2]:
            # dynamically reallocate buffer if needed
            self._img_buff_shape[1] = max(self._img_buff_shape[1], w)
            self._img_buff_shape[0] = max(self._img_buff_shape[0], h)

            self._roi_img_buff = np.zeros(self._img_buff_shape, np.uint8)
            self._mask_img_buff = np.zeros_like(self._roi_img_buff)

        sub_mask = self._mask_img_buff[0:h, 0:w]

        sub_grey = img[y : y + h, x : x + w]
        sub_mask.fill(0)

        cv2.drawContours(sub_mask, [contour], -1, 255, -1, offset=(-x, -y))

        # Defensive check: ensure arrays have compatible shapes before cv2.mean()
        if sub_grey.shape[:2] != sub_mask.shape[:2]:
            # Fallback: use minimum dimensions to ensure compatibility
            min_h = min(sub_grey.shape[0], sub_mask.shape[0])
            min_w = min(sub_grey.shape[1], sub_mask.shape[1])
            sub_grey = sub_grey[:min_h, :min_w]
            sub_mask = sub_mask[:min_h, :min_w]

        try:
            mean_col = cv2.mean(sub_grey, sub_mask)[0]
        except cv2.error:
            # Graceful fallback when cv2.mean fails
            mean_col = 0.0

        (_, _), (width, height), angle = cv2.minAreaRect(contour)
        width, height = max(width, height), min(width, height)
        # (height + 1) / (width + 1)
        # todo speed should use time
        #
        # if len(self.positions) > 2:
        #
        #     pm, pmm = self._positions[-1],self._positions[-2]
        #     xm, xmm = pm["x"], pmm["x"]
        #     ym, ymm = pm["y"], pmm["y"]
        #
        #     instantaneous_speed = abs(xm + 1j*ym - xmm + 1j*ymm)
        # else:
        #     instantaneous_speed = 0
        # if np.isnan(instantaneous_speed):
        #     instantaneous_speed = 0

        features = np.array(
            [
                log10(cv2.contourArea(contour) + 1.0),
                height + 1,
                # sqrt(ar),
                # instantaneous_speed +1.0,
                mean_col + 1,
                # 1.0
            ]
        )

        return features


# Precomputed natural log of 2, used to convert a half-life into a decay rate.
_LOG2 = log(2)


class BackgroundModel:
    """
    A class to model background. It uses a dynamic running average and support arbitrary and heterogeneous frame rates
    """

    def __init__(
        self, max_half_life=500.0 * 1000, min_half_life=5.0 * 1000, increment=1.2
    ):
        # the maximal half life of a pixel from background, in seconds
        self._max_half_life = float(max_half_life)
        # the minimal one
        self._min_half_life = float(min_half_life)

        # starts with the fastest learning rate
        self._current_half_life = self._min_half_life

        # fixme theoretically this should depend on time, not frame index
        self._increment = increment
        # the mean background
        self._bg_mean = None
        # self._bg_sd = None

        self._buff_alpha_matrix = None
        self._buff_invert_alpha_mat = None
        # the time stamp of the frame las used to update
        self.last_t = 0

    @property
    def bg_img(self):
        return self._bg_mean

    def increase_learning_rate(self):
        self._current_half_life /= self._increment

    def decrease_learning_rate(self):
        self._current_half_life *= self._increment

    def update(self, img_t, t, fg_mask=None):
        dt = float(t - self.last_t)
        if dt < 0:
            # raise EthoscopeException("Negative time interval between two consecutive frames")
            raise NoPositionError(
                "Negative time interval between two consecutive frames"
            )

        # clip the half life to possible value:
        self._current_half_life = np.clip(
            self._current_half_life, self._min_half_life, self._max_half_life
        )

        # ensure preallocated buffers exist. otherwise, initialise them
        if self._bg_mean is None:
            self._bg_mean = img_t.astype(np.float32)
            # self._bg_sd = np.zeros_like(img_t)
            # self._bg_sd.fill(128)

        if self._buff_alpha_matrix is None:
            self._buff_alpha_matrix = np.ones_like(img_t, dtype=np.float32)

        # the learning rate, alpha, is an exponential function of half life
        # it correspond to how much the present frame should account for the background

        lam = _LOG2 / self._current_half_life
        # how much the current frame should be accounted for
        alpha = 1 - exp(-lam * dt)

        # set-p a matrix of learning rate. it is 0 where foreground map is true
        self._buff_alpha_matrix.fill(alpha)
        if fg_mask is not None:
            cv2.dilate(fg_mask, None, fg_mask)
            cv2.subtract(
                self._buff_alpha_matrix,
                self._buff_alpha_matrix,
                self._buff_alpha_matrix,
                mask=fg_mask,
            )

        if self._buff_invert_alpha_mat is None:
            self._buff_invert_alpha_mat = 1 - self._buff_alpha_matrix
        else:
            np.subtract(1, self._buff_alpha_matrix, self._buff_invert_alpha_mat)

        np.multiply(self._buff_alpha_matrix, img_t, self._buff_alpha_matrix)
        np.multiply(
            self._buff_invert_alpha_mat, self._bg_mean, self._buff_invert_alpha_mat
        )
        np.add(self._buff_alpha_matrix, self._buff_invert_alpha_mat, self._bg_mean)

        self.last_t = t


class AdaptiveBGModel(BaseTracker):
    _description = {
        "overview": "The default tracker for fruit flies. One animal per ROI.",
        "arguments": [],
    }

    def __init__(self, roi, data=None):
        """
        Initializes an adaptive background model for tracking a single animal within a specified region of interest (ROI).
        This model leverages background subtraction techniques enhanced with adaptive learning rates and preprocessing
        strategies to accurately identify and track the subject within the ROI across frames.

        Parameters:
        - roi: tuple or any
            Specifies the region of interest within the frame where tracking is to be focused. The exact type and format
            can vary depending on the implementation details and how the ROI information is utilized within the tracking
            system. It is generally expected to define the spatial boundaries for tracking.
        - data: dict or None, optional
            An optional dictionary of additional data or parameters that may be required for initializing the tracking model.
            This could include calibration data, model parameters, or other configuration settings relevant to the tracking process.

            Supported parameters:
            - 'object_expected_size': float (default: 0.05) - Expected object size as proportion of ROI main axis.
                                     For HD videos with small flies, use smaller values like 0.01-0.02
            - 'max_area_factor': float (default: 5) - Maximum area multiplier for object detection.
                                Larger values are more permissive for size variations

        The method sets up internal buffers and default settings necessary for the operation of the tracking model, including
        object size expectations, smoothing mechanisms for mode detection, and initialization of both background and foreground models.
        """
        self._previous_shape = None

        # Configure object size expectations from data parameter (retro-compatible)
        if data is not None and isinstance(data, dict):
            # Allow configuration of object size expectations for different video resolutions
            self._object_expected_size = data.get(
                "object_expected_size", 0.05
            )  # Default: 5% of ROI
            max_area_factor = data.get(
                "max_area_factor", 5
            )  # Default: 5x the expected size
            self._max_area = (max_area_factor * self._object_expected_size) ** 2

            # Foreground detection sensitivity (lower = more sensitive to slow moves in bright ROIs)
            # 15 was too permissive (wide ovals on bright ROIs), 20 is original.
            self._fg_threshold = int(data.get("fg_threshold", 20))
            self._max_m_log_lik = float(data.get("max_m_log_lik", 5.6))

            fg_history = int(data.get("fg_history_length", 1000))
            fg_ready = int(data.get("fg_ready_threshold", 50))
            # Clamp ready to history
            fg_ready = min(fg_ready, fg_history)

            # Special mode: disable size filtering for detection analysis
            if data.get("disable_size_filtering", False):
                self._object_expected_size = 0.001  # Very small - won't affect blur
                self._max_area = 1.0  # Allow detection of any size (100% of ROI)
        else:
            # Backward compatibility: use original hardcoded values
            self._object_expected_size = 0.05  # proportion of the roi main axis
            self._max_area = (5 * self._object_expected_size) ** 2
            self._fg_threshold = 20
            self._max_m_log_lik = 5.6
            fg_history = 1000
            fg_ready = 50

        self._smooth_mode = deque()
        self._smooth_mode_tstamp = deque()
        self._smooth_mode_window_dt = 30 * 1000  # miliseconds

        # Pre-calculate and store the blur radius
        self.blur_rad = None

        self._bg_model = BackgroundModel()
        self.fg_model = ObjectModel(history_length=fg_history, ready_threshold=fg_ready)
        self._buff_grey = None
        self._buff_object = None
        self._buff_object_old = None
        self._buff_grey_blurred = None
        self._buff_fg = None
        self._buff_convolved_mask = None
        self._buff_fg_backup = None
        self._buff_fg_diff = None
        self._old_sum_fg = 0
        self._buff_img = None  # cached pre-allocation, passed to others

        self._roi = roi

        super().__init__(roi, data)

    def _calculate_blur_radius(self, img_shape):
        """
        Calculate the blur radius based on the object expected size and the maximum dimension of the image.

        Args:
            img_shape (tuple): The shape of the input image.

        The function calculates the blur radius using the formula: blur_radius = int(object_expected_size * max(img_shape) / 2.0).
        It then ensures that the blur radius is an odd number by incrementing it by 1 if it's even.

        Example:
        _calculate_blur_radius((640, 480))  # Calculates the blur radius for an image with dimensions 640x480.
        """

        self.blur_rad = int(self._object_expected_size * np.max(img_shape) / 2.0)

        if self.blur_rad % 2 == 0:
            self.blur_rad += 1

    def _pre_process_input_minimal(self, img, mask, t, darker_fg=True):
        """
        Preprocesses an input image for object tracking, with optional foreground darkening.

        This function first checks if the input image is in color and converts it to grayscale if necessary.
        It then applies a Gaussian blur to smooth the image, potentially inverts the image colors
        to highlight darker foreground elements, normalizes the image brightness, and finally applies
        a mask to isolate the region of interest. The result is a preprocessed image optimized for subsequent
        tracking operations.

        Parameters:
        - img: numpy.ndarray
            The input image, either in grayscale or BGR color.
        - mask: numpy.ndarray or None
            A binary mask defining the region of interest. If None, a mask covering the entire image is used.
        - t: any
            A timestamp or identifier for the input image. Currently not used in processing,
            but included for compatibility with future extensions.
        - darker_fg: bool, optional (default=True)
            If True, inverts the image colors to make the foreground elements darker than the background.

        Returns:
        - buff_img: numpy.ndarray
            The preprocessed image, ready for object tracking.

        Raises:
        - NoPositionError: If an error occurs due to division by zero in scaling calculations.
        """

        # Check if the image has more than one channel (indicative of a color image)
        # and convert it to grayscale if necessary
        if len(img.shape) > 2 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not self.blur_rad:
            self._calculate_blur_radius(img.shape)
        if mask is None:
            mask = np.ones(img.shape, dtype=np.uint8) * 255

        # Check if pre-allocation exists
        if (
            self._buff_img is None
            or self._buff_img.shape != img.shape
            or self._buff_img.dtype != img.dtype
        ):
            # Create the pre-allocation
            self._buff_img = np.empty(img.shape, dtype=img.dtype)

        # Maintain local binding
        buff_img = self._buff_img
        # In-place: write img into the pre-allocated buffer.
        np.copyto(buff_img, img)

        cv2.GaussianBlur(buff_img, (self.blur_rad, self.blur_rad), 1.2, buff_img)

        if darker_fg:
            cv2.subtract(255, buff_img, buff_img)

        # Defensive check: ensure arrays have compatible shapes before cv2.mean()
        if buff_img.shape[:2] != mask.shape[:2]:
            # Fallback: use minimum dimensions to ensure compatibility
            min_h = min(buff_img.shape[0], mask.shape[0])
            min_w = min(buff_img.shape[1], mask.shape[1])
            buff_img = buff_img[:min_h, :min_w]
            mask = mask[:min_h, :min_w]

        mean = cv2.mean(buff_img, mask)

        try:
            scale = 128.0 / mean[0]
        except ZeroDivisionError as e:
            raise NoPositionError from e

        cv2.multiply(buff_img, scale, dst=buff_img)
        cv2.bitwise_and(buff_img, mask, buff_img)

        return buff_img

    def _find_position(self, img, mask, t):
        """
        Middleman between the tracker and the actual tracking routine
        It cuts the portion defined by mask (i.e. the ROI), converts it to grey and passes it on to the actual tracking routine
        to look for the flies to track. The result of the tracking routine is a list of points describing the objects found in that ROI
        """

        pre_processed_image = self._pre_process_input_minimal(img, mask, t)

        try:
            return self._track(img, pre_processed_image, mask, t)

        except NoPositionError as e:
            self._bg_model.update(pre_processed_image, t)
            raise NoPositionError from e

    def _track(self, img, grey, mask, t):
        """
        Tracks objects in a given frame by detecting changes from a background model, identifying contours,
        fitting ellipses to these contours, and updating tracking models based on the analysis.
        Conditions such as the proportion of foreground pixels and contour analysis guide the tracking process,
        including decisions to update learning rates and when to raise exceptions for tracking failures.

        Parameters:
        - img: numpy.ndarray
            The current frame in its original color space.
        - grey: numpy.ndarray
            The current frame converted to grayscale.
        - mask: numpy.ndarray
            An optional mask to focus tracking on a specific region of interest in the frame.
        - t: int or float
            The current timestamp or frame index.

        Returns:
        - list of DataPoint
            A list containing a single DataPoint object representing the tracked object's position,
            dimensions, orientation, and the logarithmic distance moved since the last frame.

        Raises:
        - NoPositionError
            If the background model is not set, the proportion of foreground pixels is too high or zero,
            no valid contours are found, or the detected change exceeds the maximum log-likelihood threshold.
        """
        if self._bg_model.bg_img is None:
            self._buff_fg = np.empty_like(grey)
            self._old_pos = 0.0 + 0.0j

            self._buff_object = np.empty_like(grey)
            self._buff_fg_backup = np.empty_like(grey)

            raise NoPositionError

        # Background subtraction to isolate foreground objects.
        cv2.subtract(grey, self._bg_model.bg_img.astype(np.uint8), dst=self._buff_fg)
        cv2.threshold(
            self._buff_fg, self._fg_threshold, 255, cv2.THRESH_TOZERO, dst=self._buff_fg
        )

        # Backup the foreground buffer for subsequent analysis.
        np.copyto(self._buff_fg_backup, self._buff_fg)

        # Calculate the proportion of foreground pixels.
        prop_fg_pix = np.count_nonzero(self._buff_fg) / (grey.size)

        if prop_fg_pix > self._max_area or prop_fg_pix == 0:
            self._bg_model.increase_learning_rate()
            raise NoPositionError

        contours, _ = cv2.findContours(
            self._buff_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [
            cv2.approxPolyDP(c, 1.2, True) for c in contours if cv2.contourArea(c) >= 6
        ]

        # Process contours
        hull, distance, is_ambiguous = self._process_contours(img, contours, t)

        if distance > self._max_m_log_lik:
            self._bg_model.increase_learning_rate()
            raise NoPositionError

        self._previous_shape = np.copy(hull)

        # Ellipse fitting and adjustments
        (x, y), (w, h), angle = self._fit_and_adjust_ellipse(hull, grey)

        h_im = min(grey.shape)
        w_im = max(grey.shape)
        max_h = 2 * h_im
        if w > max_h or h > max_h:
            raise NoPositionError

        # Update tracking models based on analysis
        self._update_models(
            img, grey, mask, hull, t, distance, prop_fg_pix, is_ambiguous
        )

        # normalised position
        # this does not really need to be normalised because the image size does not change between frames
        pos = (x + 1.0j * y) / w_im
        xy_dist = round(log10(1.0 / float(w_im) + abs(pos - self._old_pos)) * 1000)

        # non normalised
        # pos = x + 1.0j * y  # Keep position without normalization
        # xy_dist = round(log10(abs(pos - self._old_pos) + 1) * 1000)  # Add 1 to avoid log(0)

        self._old_pos = pos

        # This can be use during offline tracking for debug purposes.
        # cv2.imshow(f"ROI_{self._roi.idx}", grey ); cv2.waitKey(1)

        return [
            DataPoint(
                [
                    XPosVariable(int(round(x))),
                    YPosVariable(int(round(y))),
                    WidthVariable(int(round(w))),
                    HeightVariable(int(round(h))),
                    PhiVariable(int(round(angle))),
                    XYDistance(int(xy_dist)),
                ]
            )
        ]

    def _update_models(
        self, img, grey, mask, hull, t, distance, prop_fg_pix, is_ambiguous
    ):
        """
        Updates the background and foreground models based on the analysis of the current frame.
        The background model's learning rate is adjusted according to the ambiguity of the detection,
        and the foreground model is updated with the current detection details. Optionally,
        a mask can be applied to the foreground before updating the models to focus on a specific
        area of interest.

        Parameters:
        - img: numpy.ndarray
            The current frame in its original color space, used for updating the foreground model.
        - grey: numpy.ndarray
            The grayscale version of the current frame, used for updating the background model.
        - mask: numpy.ndarray or None
            An optional binary mask that defines the region of interest for the foreground update.
            If provided, it is applied to the foreground before the model updates.
        - hull: numpy.ndarray
            The contour points of the detected object, used for updating the foreground model.
        - t: int or float
            The current timestamp or frame index, used for temporal reference in model updates.
        - distance: float
            The measured distance or change metric between the current and previous detections.
            This parameter is currently not used directly in this function but included for potential
            extensions or conditional logic.
        - prop_fg_pix: float
            The proportion of foreground pixels in the detection. This parameter is not directly
            used in this function but included for potential extensions or conditional logic.
        - is_ambiguous: bool
            A flag indicating whether the current detection situation is ambiguous, affecting
            how the background model's learning rate is adjusted.

        Returns:
        None
        """

        if mask is not None:
            cv2.bitwise_and(self._buff_fg, mask, self._buff_fg)

        if is_ambiguous:
            self._bg_model.increase_learning_rate()
            self._bg_model.update(grey, t)
        else:
            self._bg_model.decrease_learning_rate()
            self._bg_model.update(grey, t, self._buff_fg)

        self.fg_model.update(img, hull, t)

    def _process_contours(self, img, contours, t):
        """
        Processes detected contours to identify the primary object for tracking, assesses the
        ambiguity of the detection, and calculates the distance moved by the tracked object based
        on foreground model predictions. This method updates the learning rate of the background model
        based on detection outcomes and raises exceptions if no valid contours are found or the foreground
        model is not ready for ambiguous situations.

        Parameters:
        - img: numpy.ndarray
            The current frame in its original color space, used for feature computation by the foreground model.
        - contours: list of numpy.ndarray
            A list of contour arrays, where each contour is represented by an array of points.
        - t: int or float
            The current timestamp or frame index, used for temporal reference in distance calculations.

        Returns:
        - tuple: (hull, distance, is_ambiguous)
            hull: numpy.ndarray
                The contour of the identified primary object for tracking.
            distance: float
                The computed distance metric indicating the movement of the object, based on the
                foreground model's analysis of the current and previous frames.
            is_ambiguous: bool
                A flag indicating whether the current frame's detections are ambiguous, based on
                the number of significant contours identified.

        Raises:
        - NoPositionError
            If no valid contours are found, if the foreground model is not ready in ambiguous situations,
            or if the identified primary contour does not meet minimum criteria for tracking.
        """

        if not contours:
            self._bg_model.increase_learning_rate()
            raise NoPositionError

        if len(contours) > 1:
            hulls = [h for h in contours if h.shape[0] >= 3]

            if len(hulls) < 1:
                raise NoPositionError

            if not self.fg_model.is_ready:
                # Warm-up: not enough FG history for reliable likelihood.
                # Avoid freezing oval on fast moves: pick largest contour
                # as fallback and treat as unambiguous so BG does not
                # freeze. Will switch to likelihood-based selection after
                # ready_threshold (100 samples per ROI) is reached.
                hull = max(hulls, key=cv2.contourArea)
                distance = 0
                is_ambiguous = False
            else:
                is_ambiguous = len(hulls) > 1

                cluster_features = [
                    self.fg_model.compute_features(img, h) for h in hulls
                ]
                all_distances = [
                    self.fg_model.distance(cf, t) for cf in cluster_features
                ]
                good_clust = np.argmin(all_distances)

                hull = hulls[good_clust]
                distance = all_distances[good_clust]

        else:
            is_ambiguous = False

            hull = contours[0]
            if hull.shape[0] < 3:
                self._bg_model.increase_learning_rate()
                raise NoPositionError

            features = self.fg_model.compute_features(img, hull)
            distance = self.fg_model.distance(features, t)

        return hull, distance, is_ambiguous

    def _fit_and_adjust_ellipse(self, hull, grey):
        """
        Fits an ellipse to the given contour (hull) and adjusts its dimensions and orientation.
        Validates that the ellipse does not exceed the image dimensions. Draws the adjusted
        ellipse on a foreground buffer and calculates its center of mass.

        Parameters:
        - hull: numpy.ndarray
            The contour points of the detected object.
        - grey: numpy.ndarray
            The grayscale image on which the object was detected.

        Returns:
        - tuple: ((x, y), (w, h), angle)
            The center, dimensions, and orientation angle of the fitted and adjusted ellipse.
            - (x, y): The center of mass of the area within the drawn ellipse.
            - (w, h): The width and height of the ellipse.
            - angle: The orientation angle of the ellipse.

        Raises:
        - NoPositionError:
            If the fitted ellipse's dimensions exceed the dimensions of the input image.
        """

        (x, y), (w, h), angle = cv2.minAreaRect(hull)

        if w < h:
            angle -= 90
            w, h = h, w
        angle = angle % 180

        img_height, img_width = grey.shape[:2]
        if w > img_width or h > img_height:
            raise NoPositionError

        cv2.ellipse(
            self._buff_fg, ((x, y), (int(w * 1.5), int(h * 1.5)), angle), 255, -1
        )

        # todo center mass just on the ellipse area
        cv2.bitwise_and(self._buff_fg_backup, self._buff_fg, self._buff_fg_backup)

        # Calculate all moments of the image array
        M = cv2.moments(self._buff_fg_backup)

        # m00 is the total area. We must check if it's zero to avoid a ZeroDivisionError
        # which will crash your script if a frame has no flies (a completely black mask).
        if M["m00"] != 0:
            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
        else:
            # failed to get center of mass
            logging.warning("Failed to get center of mass.")
            raise NoPositionError

        return (x, y), (w, h), angle

import numpy as np
from filterpy.kalman import KalmanFilter


class Kalman:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    def __init__(self, bbox_3d):
        """
        Initialises a tracker using initial bounding box.
        [x, y, z, dx, dy, dz, heading]
        """
        # define constant velocity model
        #  [x, y, z, dx, dy, dz, heading, vx, vy, vz]
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement matrix,
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P[7:, 7:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[7:, 7:] *= 0.01  # process noise matrix
        self.kf.x[:7] = bbox_3d.reshape((7, 1))

    def update(self, bbox_3d):
        """
        Updates the state vector with observed bbox.
        """
        # ------------------
        # orientation correction
        if self.kf.x[6] >= np.pi:
            self.kf.x[6] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[6] < -np.pi:
            self.kf.x[6] += np.pi * 2

        new_theta = bbox_3d[6]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox_3d[6] = new_theta

        predicted_theta = self.kf.x[6]
        # if the angle of two theta is not acute angle
        if np.pi / 2.0 < abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
            self.kf.x[6] += np.pi
            if self.kf.x[6] > np.pi:
                self.kf.x[6] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[6] < -np.pi:
                self.kf.x[6] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[6]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[6] += np.pi * 2
            else:
                self.kf.x[6] -= np.pi * 2
        # ------------------

        self.kf.update(bbox_3d)

        if self.kf.x[6] >= np.pi:
            self.kf.x[6] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[6] < -np.pi:
            self.kf.x[6] += np.pi * 2

    def predict(self, t=1) -> np.array:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        for _ in range(t):
            self.kf.predict()
        if self.kf.x[6] >= np.pi:
            self.kf.x[6] -= np.pi * 2
        if self.kf.x[6] < -np.pi:
            self.kf.x[6] += np.pi * 2
        return self.kf.x[:7]

    def get_box(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7]

    def get_predicted_box(self):
        return np.dot(self.kf.F, self.kf.x)[:7]


class KalmanPSR:
    def __init__(self, psr):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # (px, py, pz, sx, sy, sz, rx, ry, rz, vx, vy, vz)
        self.kf = KalmanFilter(dim_x=12, dim_z=9)
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement matrix,
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P[9:, 9:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[9:, 9:] *= 0.01  # process noise matrix
        self.kf.x[:9] = psr.reshape((9, 1))

    @staticmethod
    def _rotation_correction(rot):
        """
        Lets the rotation angles in range [-pi, pi)
        :param rot:
        :return:
        """
        for i in range(3):
            r = rot[i]
            if r >= np.pi:
                r -= np.pi * 2
            elif r < -np.pi:
                r += np.pi * 2
            rot[i] = r
        return rot

    def _rotation_acute_correction(self, psr):
        for i in (6, 7, 8):
            # if the angle of two theta is not acute angle
            if np.pi / 2.0 < abs(psr[i] - self.kf.x[i]) < np.pi * 3 / 2.0:
                self.kf.x[i] += np.pi
                if self.kf.x[i] > np.pi:
                    self.kf.x[i] -= np.pi * 2  # make the theta still in the range
                if self.kf.x[i] < -np.pi:
                    self.kf.x[i] += np.pi * 2

            # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
            if abs(psr[i] - self.kf.x[i]) >= np.pi * 3 / 2.0:
                if psr[i] > 0:
                    self.kf.x[i] += np.pi * 2
                else:
                    self.kf.x[i] -= np.pi * 2

    def update(self, psr):
        """
        Updates the state vector with observed bbox.
        """
        psr[6:9] = self._rotation_correction(psr[6:9])
        self._rotation_acute_correction(psr)
        self.kf.update(psr)
        self.kf.x[6:9] = self._rotation_correction(self.kf.x[6:9])

    def predict(self, t) -> np.array:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        for _ in range(t):
            self.kf.predict()
        self.kf.x[6:9] = self._rotation_correction(self.kf.x[6:9])
        return self.kf.x[:9]

    def get_box(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:9]

    def get_predicted_box(self):
        return np.dot(self.kf.F, self.kf.x)[:9]

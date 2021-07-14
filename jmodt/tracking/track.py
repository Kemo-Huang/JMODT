from jmodt.tracking.kalman import Kalman


class Track:
    new_id = 1

    def __init__(self, bbox, score, info, feature):
        self.id = Track.new_id
        Track.new_id += 1
        self.motion_model = Kalman(bbox)
        self.score = score
        self.info = info
        self.feature = feature
        self.misses = 0
        self.hits = 0

    def predict(self, t=1):
        self.misses += t
        state = (self.motion_model.predict(t).flatten(), self.score, self.feature)
        return state

    def _update_state(self):
        self.misses = 0
        self.hits += 1

    def _update_motion(self, bbox_3d):
        self.motion_model.update(bbox_3d)

    def _update_feature(self, feature):
        self.feature = feature

    def _update_score(self, score):
        self.score = score

    def _update_info(self, info):
        self.info = info

    def update_with_feature(self, box, feature, score, info):
        self._update_state()
        self._update_motion(box)
        self._update_info(info)
        self._update_feature(feature)
        self._update_score(score)

    def update_feature_score(self, feature, score):
        self._update_feature(feature)
        self._update_score(score)

    def get_data(self):
        return self.id, self.info, self.score

    def get_predicted_data(self):
        return self.id, self.motion_model.get_predicted_box(), self.info, self.score

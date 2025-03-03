import cv2
import numpy as np

class RoomCounter:
    def __init__(self, area_list):
        self.area_list = area_list
        self.people_entering = [{} for _ in area_list]
        self.people_exiting = [{} for _ in area_list]
        self.entering = [set() for _ in area_list]
        self.exiting = [set() for _ in area_list]

    def update_count(self, tracker_data):
        for i, (area_in, area_out) in enumerate(self.area_list):
            for bbox in tracker_data:
                x3, y3, x4, y4, id = bbox
                
                if cv2.pointPolygonTest(np.array(area_out, np.int32), (x4, y4), False) >= 0:
                    self.people_entering[i][id] = (x4, y4)
                if id in self.people_entering[i]:
                    if cv2.pointPolygonTest(np.array(area_in, np.int32), (x4, y4), False) >= 0:
                        self.entering[i].add(id)

                if cv2.pointPolygonTest(np.array(area_in, np.int32), (x4, y4), False) >= 0:
                    self.people_exiting[i][id] = (x4, y4)
                if id in self.people_exiting[i]:
                    if cv2.pointPolygonTest(np.array(area_out, np.int32), (x4, y4), False) >= 0:
                        self.exiting[i].add(id)

        return [(len(self.entering[i]), len(self.exiting[i])) for i in range(len(self.area_list))]

import os.path as osp
import sys

sys.path.insert(0, osp.dirname(__file__) + '/..')
import roslibpy
from robot.ros import remote


class VisionClient:
    def __init__(self):
        self.image_service = roslibpy.core.Service(remote.ros, '/web_server/vision_server',
                                                   'web_server/VisionService')
        self.pcd_service = roslibpy.core.Service(remote.ros, '/web_server/pcd_server',
                                                 'web_server/PointCloudService')

    def save_pcd(self, quality='hd'):
        req = {'quality': quality}
        res = self.pcd_service.call(req)
        print(res['response'])

    def save_rgb(self, quality='hd'):
        req = {'quality': quality, 'type': 'depth'}
        res = self.image_service.call(req)
        print(res['response'])

    def save_depth(self, quality='hd'):
        req = {'quality': quality, 'type': 'color'}
        res = self.image_service.call(req)
        print(res['response'])

    def save_both_image(self, quality='hd'):
        req = {'quality': quality, 'type': 'image'}
        res = self.image_service.call(req)
        print(res['response'])

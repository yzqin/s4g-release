from mujoco.base.mjcf_xml import MujocoXML
from mujoco.base.arena_base import ArenaBase
import os
from configs import config
from mujoco.base.mjcf_utils import array_to_string

dirname = os.path.dirname(__file__)


class GripperBase(MujocoXML):
    def __init__(self, xml_file):
        MujocoXML.__init__(self, xml_file, normalize_names=False)
        self.gripper = list(self.worldbody)[0]


class ToyGripper(GripperBase):
    def __init__(self, xml_file=os.path.join(dirname, '../assets/toy_gripper.xml'), match_config=True):
        GripperBase.__init__(self, xml_file)
        if match_config:
            self.match_config_size()

    def match_config_size(self):
        body_list = list(self.gripper)
        left = body_list[2]
        right = body_list[3]
        bottom = body_list[4]

        left.set('pos',
                 array_to_string([config.FINGER_LENGTH / 2, config.HALF_BOTTOM_SPACE + config.FINGER_WIDTH / 2, 0]))
        left_geom = left.find('geom')
        left_geom.set('size',
                      array_to_string([config.FINGER_LENGTH / 2, config.FINGER_WIDTH / 2, config.HALF_HAND_THICKNESS]))
        left_joint = left.find('joint')
        left_joint.set('range', array_to_string([0, config.HALF_BOTTOM_SPACE]))

        right.set('pos',
                  array_to_string([config.FINGER_LENGTH / 2, -config.HALF_BOTTOM_SPACE - config.FINGER_WIDTH / 2, 0]))
        right_geom = left.find('geom')
        right_geom.set('size',
                       array_to_string([config.FINGER_LENGTH / 2, config.FINGER_WIDTH / 2, config.HALF_HAND_THICKNESS]))
        right_joint = right.find('joint')
        right_joint.set('range', array_to_string([0, config.HALF_BOTTOM_SPACE]))

        bottom.set('pos', array_to_string([-config.BOTTOM_LENGTH / 2, 0, 0]))
        bottom_geom = bottom.find('geom')
        bottom_geom.set('size', array_to_string(
            [config.BOTTOM_LENGTH / 2, config.HALF_BOTTOM_WIDTH, config.HALF_HAND_THICKNESS]))


if __name__ == '__main__':
    toy = GripperBase(os.path.join(dirname, '../assets/toy_gripper.xml'))

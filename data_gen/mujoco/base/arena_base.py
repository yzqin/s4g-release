from .mjcf_xml import MujocoXML


class ArenaBase(MujocoXML):
    def __init__(self, xml_file):
        MujocoXML.__init__(
            self,
            xml_file,
            normalize_names=False)

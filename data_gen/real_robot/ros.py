import roslibpy


class MovoRemote:
    def __init__(self):
        self.ros = roslibpy.Ros(host='10.66.171.1', port=9090)
        self.ros.run()
        self.ros.on_ready(lambda: print('Is ROS connected?', self.ros.is_connected))

    def terminate(self):
        self.ros.terminate()


remote = MovoRemote()

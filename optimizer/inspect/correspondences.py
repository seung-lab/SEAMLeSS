from sockjs.tornado import SockJSConnection, SockJSRouter
from tornado import web, ioloop, httpserver
from collections import OrderedDict

import threading
import sys
import json
import numpy as np
import os

class Controller:
    """Simple class to set & get point-pairs from neuroglancer
    """
    state = {}
    generation = 0
    clients = set()

    def __init__(self, router):
        self.router = router

    def send_state(self):
        self.generation += 1
        self.router.broadcast(self.clients.copy(), '{{"g": {}, "t":"setState", "k":"s", "s":{}}}'.format(self.generation, json.dumps(self.state)))

    def get(self):
        if 'correspondences' in self.state['layers']:
            if 'points' in self.state['layers']['correspondences']:
                return map(tuple, self.state['layers']['correspondences']['points'])
        else:
            self.set()
        return []

    def set(self, points=[]):
        self.state['layers']['correspondences'] = {'type':'synapse', 'points':points}
        self.send_state()

    def set_z(self, z):
        self.state['navigation']['pose']['position']['voxelCoordinates'][2] = z
        self.send_state()

    def get_position(self):
        return self.state['navigation']['pose']['position']['voxelCoordinates']

    # websockets connections
    class Connection(SockJSConnection):
        n_messages = 0

        def on_open(self, info):
            """
            info is an object which contains caller IP address, query string
            parameters and cookies associated with this request"""
            # When new client comes in, will add it to the clients list
            Controller.clients.add(self)

        def on_message(self, json_state):
            """
            This will call initialize_current_state or on_current_state_change depening on if it is
            the first message recieved.
            """
            # print(json_state + ': ' + str(len(Controller.state)) + '\n')
            if not self.n_messages:
                self.n_messages += 1
                print('state initialized')
                # print(json_state)

            message = json.JSONDecoder(object_pairs_hook=OrderedDict).decode(json_state)
            if "t" in message:
                if message["t"] == "setState" and message["k"] == "s": # "s" stands for Shared State. There is also "p" for private and "c" for config
                    Controller.state = message["s"]
                elif message["t"] == "getState" and message["k"] == "s" and len(Controller.state) > 0:
                    self.send('{{"g": {}, "t":"setState", "k":"s", "s":{}}}'.format(Controller.generation, json.dumps(Controller.state)))
                elif message["t"] == "action":
                    if message["actions"][-1]["action"] == "initState":
                        Controller.state = message["actions"][-1]["state"]
                    self.send('{{"t": "ackAction", "id": {0}}}'.format(message["id"]))


        def on_close(self):
            # If client disconnects, remove him from the clients list
            Controller.clients.remove(self)


# Tornado & Tk need to run on separate thresads
class TornadoThread(threading.Thread):
    def __init__(self, port):
        # super(TornadoThread, self).__init__()
        # self._stop = threading.Event()
        threading.Thread.__init__(self)
        self.daemon = True
        self.port = port
        self.router = SockJSRouter(Controller.Connection)
        socketApp = web.Application(self.router.urls)
        self.http_server = httpserver.HTTPServer(socketApp, ssl_options={
            'certfile': os.path.join(sys.path[0], 'certificate.crt'),
            'keyfile': os.path.join(sys.path[0], 'privateKey.key')
        })
        self.http_server.bind(port) #port
        self.http_server.start(1)

    def run(self):
        print('IOLoop starting on port {0}'.format(self.port))
        ioloop.IOLoop.instance().start()

def controller(port=9999):
    thread = TornadoThread(port)
    thread.start()
    return Controller(thread.router)

if __name__ == '__main__':
    c = controller()


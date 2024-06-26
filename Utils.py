import json
from types import SimpleNamespace
import os
import shutil
from datetime import datetime


class Utils:
    def __init__(self):
        # self.res_folder = self.make_res_folder()
        with open('./Parameters.json', 'r') as json_file:
            self._params = json.load(json_file,
                                     object_hook=lambda d: SimpleNamespace(**d))

    @property
    def params(self):
        return self._params

    # def make_res_folder(self, root_dir='./'):
    #
    #     now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     folder = 'tr{0}'.format(now)
    #     dirname = os.path.join(root_dir, folder)
    #
    #     if folder is not None and os.path.exists(folder) and not os.path.exists(dirname):
    #         os.mkdir(dirname)
    #     elif not os.path.exists(dirname):
    #         os.makedirs(dirname)
    #
    #     self.res_folder = dirname
    #     shutil.copy('./Parameters.json', self.res_folder)
    #     return dirname

    # def get_log_dir(self):
    #     self._log_dir = os.path.join(self.res_folder, 'log')
    #     if not os.path.exists(os.path.join(self.res_folder, 'log')):
    #         os.mkdir(os.path.join(self.res_folder, 'log'))
    #     return self._log_dir
    #
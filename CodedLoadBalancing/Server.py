class Server:
    """Define the server class"""

#    files_list = []
#    load = 0
#    id = [] # server identifier


    def __init__(self, identity):
        self.id = identity # server identifier
        self.load = 0 # number of balls (request) in the server
        self.files_list = [] # list of available files (in fact file chunks) in the server


    def set_files_list(self, list):
        self.files_list = list


    # The file is a file or a chunk of a file in the coded case
    def append_files_list(self, file):
        self.files_list.append(file)


    def get_files_list(self):
        return self.files_list


    def add_load(self, ld):
        self.load = self.load + ld


    def get_load(self):
        return self.load


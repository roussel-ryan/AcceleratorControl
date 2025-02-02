import time
import logging

class Sleep:
    def __init__(self, sleep_time):
        '''
        sleep_time : float
            Sleep time in seconds

        '''
        self.logger = logging.getLogger(__name__)
        self.s = sleep_time

    def __call__(self, controller):
        self.logger.info(f'waiting {self.s} seconds before observations_list')
        time.sleep(self.s)

class KeyPress:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __call__(self, controller):
        self.logger.info('waiting for any key press to do observations_list')
        input('Press any key to continue...')

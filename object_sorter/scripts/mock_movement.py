# temporary fake movement file
import time

start_time = None
move_time = 5
sort_time = 5

def move(obj_info):
    if not start_time:
        start_time = time.time()
    if time.time() - start_time >= move_time:
        start_time = None
        return "finished_moving"
    
def sort(obj_info):
    if not start_time:
        start_time = time.time()
    if time.time() - start_time >= sort_time:
        start_time = None
        return "finished_sorting"
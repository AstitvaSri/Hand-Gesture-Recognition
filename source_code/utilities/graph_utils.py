import numpy as np
from collections import deque

def add_neighbours(a, queue, x, y, finished):
    #check 8 unfinished neighbours of the current pixel x, y
    if 0 <= x-1 < a.shape[0] and 0 <= y-1 < a.shape[1] and finished[x-1, y-1]==0:
        queue.append([x-1, y-1])
    if 0 <= x <= a.shape[0] and 0 <= y-1 < a.shape[1] and finished[x, y-1]==0:
        queue.append([x, y-1])
    if 0 <= x+1 < a.shape[0] and 0 <= y-1 < a.shape[1] and finished[x+1, y-1]==0:
        queue.append([x+1, y-1])
    if 0 <= x-1 < a.shape[0] and 0 <= y < a.shape[1] and finished[x-1, y]==0:
        queue.append([x-1, y])
    if 0 <= x+1 < a.shape[0] and 0 <= y < a.shape[1] and finished[x+1, y]==0:
        queue.append([x+1, y])
    if 0 <= x-1 < a.shape[0] and 0 <= y+1 < a.shape[1] and finished[x-1, y+1]==0:
        queue.append([x-1, y+1])
    if 0 <= x < a.shape[0] and 0 <= y+1 < a.shape[1] and finished[x, y+1]==0:
        queue.append([x, y+1])
    if 0 <= x+1 < a.shape[0] and 0 <= y+1 < a.shape[1] and finished[x+1, y+1]==0:
        queue.append([x+1, y+1])

def bfs(a, x, y, target_value):
    a = np.array(a)
    finished = np.zeros(a.shape, dtype=np.uint8)
    bfs_queue = deque([])
    bfs_queue.append([x, y])
    result = [-1, -1]
    while len(bfs_queue) != 0:
        qx, qy = bfs_queue.popleft()
        if qx < 0 or qx >=a.shape[0]:
            continue
        if qy < 0 or qy >= a.shape[1]:
            continue 
        if finished[qx, qy] == 1:
            continue 
        finished[qx, qy] = 1
        if a[qx, qy] <= target_value:
            result = [qx, qy]
            break
        add_neighbours(a, bfs_queue, qx, qy, finished)
    return result

def bfs_multiprocessor(a, x, y, target_value, return_dict):
    result = bfs(a, x, y, target_value)
    return_dict["%d:%d" % (x,y)] = result
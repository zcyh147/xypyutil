import time

wait_sec = 1
n_iters = 3
counter = 0
while counter < n_iters:
    print(f'iter: {counter}/{n_iters}, sleep {wait_sec}')
    time.sleep(wait_sec)
    counter += 1

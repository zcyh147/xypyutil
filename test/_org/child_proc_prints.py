# test_output.py
import platform
import subprocess
import sys
import time


def run_to_end():
    print("Starting...")

    for i in range(2):
        print(f"stdout: Count {i+1}")
        time.sleep(1)

    print("Ending...")


def throw_calledprocesserror():
    """
    - called by watch_cmd as a grandchild process
    - although it throws a subprocess.CalledProcessError,
    - its parent (watch_cmd's child) does not care and simply returns 1 and include the grandchild exception in its own stderr
    """
    py = 'python' if platform.system() == 'Windows' else 'python3'
    subprocess.run([py, '-c', 'raise RuntimeError("failed")'], check=True)


def throw_filenotfound():
    subprocess.run(['missing'])


if __name__ == "__main__":
    imp_map = {
        'default': run_to_end,
        'suberr': throw_calledprocesserror,
        'exc': throw_filenotfound,
    }
    imp_map[sys.argv[1]]()

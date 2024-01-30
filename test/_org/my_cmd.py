import subprocess
import sys


def throw_calledprocesserror():
    subprocess.run(['poetry', 'run', 'python', '-c', 'raise RuntimeError("failed")'], check=True)


def throw_filenotfound():
    subprocess.run(['missing'])


if __name__ == '__main__':
    if sys.argv[1] == 'suberr':
        throw_calledprocesserror()
    elif sys.argv[1] == '100':
        print('OK: util.run_cmd() has converted number to string internally')
    else:
        throw_filenotfound()

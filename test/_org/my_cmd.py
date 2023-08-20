import subprocess


def fail_in_child_cmd():
    subprocess.run(['missing'])


if __name__ == '__main__':
    fail_in_child_cmd()

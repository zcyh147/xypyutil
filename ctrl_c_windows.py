import ctypes
import platform
import sys


def main():
	if platform.system() != 'Windows':
		return 1

	kernel = ctypes.windll.kernel32
	pid = int(sys.argv[1])
	kernel.FreeConsole()
	kernel.AttachConsole(pid)
	kernel.SetConsoleCtrlHandler(None, 1)
	kernel.GenerateConsoleCtrlEvent(0, 0)
	sys.exit(0)


if __name__ == '__main__':
	main()


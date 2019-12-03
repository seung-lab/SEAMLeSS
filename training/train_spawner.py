from subprocess import Popen
import sys


while True:
  print("Restarting...")
  p = Popen(["python", *sys.argv[1:]])
  p.wait()


# Parse the tape head logs
import fileinput
import matplotlib.pyplot as plt

read = []
write = []

def parse_line(line):
  start_idx = line.index("[") + 1
  nums = line[start_idx:len(line)-2].split(" ")
  return [float(n) for n in nums]

for line in fileinput.input():
  nums = parse_line(line)
  try:
    idx = line.index("read")
    read.append(nums)
  except:
    idx = line.index("write")
    write.append(nums)

plt.matshow(read)
plt.savefig("images/copy_task/read_head.png")
plt.matshow(write)
plt.savefig("images/copy_task/write_head.png")

import re
f = open("out.log","r+")
nums = [line for line in f if re.match(r"\d{10}",line)]

i = 0
while i+1 < len(nums):
  minutes = (int(nums[i+1].strip()) - int(nums[i].strip()))//60
  seconds = (int(nums[i+1].strip()) - int(nums[i].strip()))%60
  string = str(minutes) + ":" + str(seconds) + "\n"
  f.write(string)
  i = i+1

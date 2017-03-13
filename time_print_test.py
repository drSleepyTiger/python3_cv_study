import time
print (time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

print (time.strftime("%y%m%d_%H%M%S",time.localtime()))
print (time.strftime("%d%b%Y_%H%M%S",time.localtime()))

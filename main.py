import time
def dprint(str):
    for c in str:
        print(c, end = '', flush = True)
        time.sleep(0.08)

dprint('Please upload your data into the folder and write the name of the file here :')


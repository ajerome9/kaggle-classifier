from datetime import datetime

def msg(m):
    print('{}: {}'.format(datetime.now().strftime("%d-%H:%M:%S"), m))
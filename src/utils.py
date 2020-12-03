import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def mk_dir(path):

    num = 0
    while 1:

        dir_name = 'result/' + path + str(num)
        dir_name = os.path.join(PROJECT_ROOT, 'result', path + str(num))

        if os.path.isdir(dir_name):
            num += 1
        else:
            os.mkdir(dir_name)
            break

    return dir_name

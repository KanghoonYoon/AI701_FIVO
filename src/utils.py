import os



def mk_dir(path):

    num = 0
    while 1:

        dir_name = 'result/' + path + str(num)
        if os.path.isdir():
            num += 1
        else:
            os.mkdir(dir_name)
            break

    return dir_name

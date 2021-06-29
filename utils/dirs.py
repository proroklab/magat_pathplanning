import os
import logging


def create_dirs(dirs):
    '''
    Create directories in the systm
    Args:
        dirs: a list of directories to create if these directories are not found

    Returns:

    '''
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)

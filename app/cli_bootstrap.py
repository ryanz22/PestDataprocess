import logging
import sys


def bootstrap_cli(logger_name: str = "dataprocess", level: int = logging.ERROR):
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 11):
        sys.exit("this program needs python 3.11 and above to run")

    l_fmt = "[%(name)s %(levelname)s] %(asctime)s - %(message)s"
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(l_fmt))
    logger = logging.getLogger(logger_name)
    logger.addHandler(ch)
    logger.setLevel(level)

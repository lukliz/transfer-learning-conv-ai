from pathlib import Path
import coloredlogs
import datetime
import sys
import logging

def setup_logging(name, level=logging.INFO):
    ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
    log_path = Path(f'../logs/irc_bot_{ts}.log')
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=level, 
        format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    coloredlogs.install(level=level)

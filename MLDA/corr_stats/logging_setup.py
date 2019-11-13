"""Setup for logging"""
logging.config.fileConfig("/home/jesper/Work/macledan/Logging/logging.conf")
# create logger
logger = logging.getLogger()
# For Testing
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')

import os
import logging
import logging.config

def setup_logging(output_folder, console_level="DEBUG", file_level="DEBUG"):
    """Configures logging to write to the pipeline's output directory with specified levels."""
    log_file_path = os.path.join(output_folder, "pipeline.log")

    # Convert string levels to numeric values.
    numeric_console_level = getattr(logging, console_level.upper(), logging.DEBUG)
    numeric_file_level = getattr(logging, file_level.upper(), logging.DEBUG)
    # Set the root logger level to the lower (more verbose) of the two.
    numeric_root_level = min(numeric_console_level, numeric_file_level)
    root_level = logging.getLevelName(numeric_root_level)

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': console_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'level': file_level,
                'class': 'logging.FileHandler',
                'filename': log_file_path,
                'mode': 'a', # Append to the log file
                'formatter': 'standard',
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': root_level,
        }
    }

    logging.config.dictConfig(logging_config)
    logging.info(f"Logging initialized. Logs will be saved to {log_file_path}")

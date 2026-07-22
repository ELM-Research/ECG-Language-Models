from elm.config.load import get_config
from elm.utils.parallelism_protection import init_dist, cleanup

if __name__ == "__main__":
    config = get_config()
    try:
        init_dist(config)
    finally:
        cleanup()
from elm.config.load import get_config
from elm.utils.parallelism_protection import init_dist, cleanup

if __name__ == "__main__":
    print(get_config())
    try:
        init_dist()
    finally:
        cleanup()
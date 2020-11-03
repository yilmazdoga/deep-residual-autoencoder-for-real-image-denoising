import os
import shutil


def clean_previous_logs(opt, log_dir):
    if os.path.exists(log_dir):
        try:
            if opt.verbose:
                print("Cleaning previous logs")
            shutil.rmtree(log_dir)
        except OSError as e:
            print("Error: %s : %s" % (log_dir, e.strerror))

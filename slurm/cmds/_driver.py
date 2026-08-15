# Drivers are materialized into .pipeline/scripts/, so they cannot import a sibling
# by relative path -- each one puts this directory on sys.path explicitly.
import os
import subprocess


# Children inherit the redirected descriptors, so this runs once at startup.
def tee_streams(out_path, err_path):
    for path, fd in ((out_path, 1), (err_path, 2)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # tee appends, so a re-run would otherwise concatenate onto the previous one's log
        open(path, "w").close()
        r, w = os.pipe()
        subprocess.Popen(["tee", "-a", path], stdin=r, stdout=os.dup(fd))
        os.close(r)
        os.dup2(w, fd)
        os.close(w)


# Children load numba/pyarrow extensions built against their own venv's libarrow.
def use_pyarrow_libs(python):
    libdir = subprocess.run(
        [python, "-c", "import pyarrow; print(pyarrow.get_library_dirs()[0])"],
        capture_output=True, text=True, check=True).stdout.strip()
    os.environ["LD_LIBRARY_PATH"] = libdir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

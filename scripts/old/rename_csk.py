# output/kcore/cen/{percentile}/max_kcore/{node}/kcore_k{n}.txt
# makes there be at most 10 nodes in each percentile

import glob
import os


def main():
    for percentile in range(3, 100, 4):
        files = glob.glob(f"output/kcore/cen/{percentile}/max_kcore/*/kcore*.txt")
        for i, file in enumerate(files):
            tokens = file.split("/")
            tokens[-1] = f"kcore_k{int(i / 10)}.txt"
            newfile = "/".join(tokens)

            print(f"{file} -> {newfile}")
            os.rename(file, newfile)


if __name__ == "__main__":
    main()

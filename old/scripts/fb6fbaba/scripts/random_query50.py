import random
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} edgefile output")
        exit(1)

    data = np.loadtxt(sys.argv[1])
    data = np.sort(np.unique(data))
    data = data.tolist()

    with open(sys.argv[2], "w") as fout:
        random.seed(1337)
        lines = []
        for q in random.sample(data, 50):
            lines.append(f"{int(q)} 10\n")
        fout.writelines(lines)


if __name__ == "__main__":
    main()

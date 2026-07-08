import random
import sys


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} indexfile output")
        exit(1)

    with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
        numlines = len(fin.readlines())

        random.seed(1337)
        lines = []
        for q in random.sample(list(range(numlines)), 50):
            lines.append(f"{q} 10\n")
        fout.writelines(lines)


if __name__ == "__main__":
    main()

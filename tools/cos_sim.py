import numpy as np
import argparse
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", type=str)
    parser.add_argument("--f2", type=str)
    args = parser.parse_args()

    file1 = args.f1
    file2 = args.f2

    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    a = np.array([float(line.replace("\n", "")) for line in lines1 if line.replace("\n", "").strip()])
    b = np.array([float(line.replace("\n", "")) for line in lines2 if line.replace("\n", "").strip()])

    print(cos_sim(a, b))

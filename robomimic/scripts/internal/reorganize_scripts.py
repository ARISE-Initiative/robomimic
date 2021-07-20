"""
Simple script to help reorganize a sequence of runs into separate groups to copy and paste
into a bash script.
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of bash script - assumes the script is in the same directory as this file
    parser.add_argument(
        "--script",
        type=str,
    )

    parser.add_argument(
        "--n_machines",
        type=int,
    )

    args = parser.parse_args()

    python_lines_by_machine = [[] for _ in range(args.n_machines)]

    with open(args.script, "r") as f:
        # collect python commands from bash script
        lines = f.readlines()
        python_lines = [l.strip() for l in lines if l.strip().startswith("python")]

    for i in range(len(python_lines)):
        python_lines_by_machine[i % args.n_machines].append(python_lines[i])

    print("")
    for i in range(args.n_machines):
        print("\nmachine_{}\n".format(i))
        for j in range(len(python_lines_by_machine[i])):
            print(python_lines_by_machine[i][j])

import argparse
import ast
import os
from opcode import name_op


def main():
    parser = argparse.ArgumentParser(description="Read lists from multiple files in a specified directory.")
    parser.add_argument("number", type=int, help="The number of files to read.")
    parser.add_argument("basename", type=str, help="The base string for filenames.")
    parser.add_argument("directory", type=str, help="The path to the directory containing the files.")
    
    args = parser.parse_args()
    lists = []
    base_dir = os.path.abspath(os.path.join(os.getcwd(), args.directory))

    res = ''

    if args.basename == 'f':
        name_list = ['blocks3', 'blocks4', 'delivery', 'driverlog', 'ferry', 'grid', 'gripper', 'hanoi', 'logistics',
                 'miconic', 'npuzzle', 'cpuzzle', 'sokoban', 'sokopull']
    elif args.basename == 'd':
        name_list = ['dropped_blocks3', 'dropped_ferry', 'dropped_miconic', 'dropped_cpuzzle']
    else:
        name_list = None

    for name in name_list:
        lists = []
        for i in range(args.number):
            filename = f"{name}_{i}.txt"
            filepath = os.path.join(base_dir, filename)

            if not os.path.isfile(filepath):
                print(f"File not found: {filepath}")
                continue

            with open(filepath, "r") as file:
                content = file.read()
                try:
                    parsed_list = ast.literal_eval(content)
                    if isinstance(parsed_list, list):
                        lists.append(parsed_list)
                    else:
                        print(f"Warning: Content in {filepath} is not a list.")
                except (SyntaxError, ValueError) as e:
                    print(f"Error reading {filepath}: {e}")

        res_list = [0] * len(lists[0])
        for l in list(lists):
            l[5] = float(l[5].split()[0])
            l[8] = float(l[8].split()[0])
            for pos, val in enumerate(l):
                res_list[pos] += val

        overall_res_list = [float(x/len(lists)) for x in res_list]

        overall_res_list = [int(x) if x.is_integer() else round(x, 2) for x in overall_res_list]

        res += "{} & {} & {} & {} & {} & {} & {} & {} & {} s & {} % & {} & {} & {}s \\\\ \n".format(
            name,
            overall_res_list[7],
            overall_res_list[6],
            overall_res_list[0],
            overall_res_list[1],
            overall_res_list[2],
            overall_res_list[3],
            overall_res_list[4],
            overall_res_list[5],
            (10 * len(lists)),
            overall_res_list[10],
            overall_res_list[9],
            overall_res_list[8]
            )
    print(res)

if __name__ == "__main__":
    main()

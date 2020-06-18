import sys
import os
import xml_parser

def main():
    filepath = sys.argv[1]      # job..out file
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()
    f = open(sys.argv[2], "w")      # output file
    fp = open(filepath, "r")
    cnt = 0
    is_considering = False
    path = ""
    time = ""
    input_val = ""
    for line in fp:
        if "The path" in line and cnt == 0:
            line = line.strip()
            is_considering = True
            path = line.split(": ")[1]
            cnt += 1
        elif is_considering and cnt == 1:
            cnt += 1
        elif is_considering and cnt == 2:
            line = line.strip()
            time = line
            cnt += 1
        elif is_considering and cnt == 3:
            inp = line.strip()
            cnt = 0
            is_considering = False
            arr = xml_parser.xml_parser(sys.argv[3],inp)    # input xml file
            if len(arr) != int(sys.argv[4]):        # num. parameteres in xml
                continue
            else:
                f.write(str(arr)+","+time+","+path+"\n")
                path = ""
                time = ""
                input_val = ""
    cnt += 1
    f.close()

if __name__ == '__main__':
    main()

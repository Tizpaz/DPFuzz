import sys
import os
import xml_parser
import LogisticRegression
import time
import trace

SIZE_INDEX = [0,1,12]
min_max = [100000,50,100]
filename_out = "../outputs/logistic_regression_outputs_instrumentations_0.txt"
filename_res = "../outputs/logistic_regression_results_instrumentations_0.txt"
libpath = os.path.normpath(os.path.dirname(os.__file__))
tracer = trace.Trace(
    ignoredirs=[sys.prefix, sys.exec_prefix,libpath,"/home/issta/.local/lib/python2.7/site-packages/sklearn/externals/joblib/externals/loky/backend/"],
    ignoremods=["xml_parser","semaphore_tracker"], trace=1, count=0)

def main():
    # This part to read from job file,
    filepath = sys.argv[1]      # job..out file
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()
    fp = open(filepath, "r")
    f = open(filename_out, "w")      # output file
    cnt = 0
    is_considering = False
    interesting_inpts = set()
    time_val = ""
    for line in fp:
        if "The path" in line and cnt == 0:
            is_considering = True
            cnt += 1
        elif is_considering and cnt == 1:
            cnt += 1
        elif is_considering and cnt == 2:
            line = line.strip()
            time_val = line
            cnt += 1
        elif is_considering and cnt == 3:
            inp = line.strip()
            cnt = 0
            is_considering = False
            if float(time_val) > 100000.0:
                interesting_inpts.add(inp)

    fp.close()
    label = -1
    for inp in interesting_inpts:
        label += 1
        size_val = inp.split(" ")[SIZE_INDEX[0]]
        inp1 = inp.replace(size_val + " ",str(min_max[0]) + " ",1)
        size_val = inp1.split(" ")[SIZE_INDEX[1]]
        new_inp = inp1.replace(" " + size_val + " "," " + str(min_max[1]) + " ",1)
        size_val = new_inp.split(" ")[SIZE_INDEX[2]]
        new_inp_1 = new_inp.replace(" " + size_val + " "," " + str(min_max[2]) + " ",1)
        stdout_ = sys.stdout
        sys.stdout = f
        tracer.runfunc(LogisticRegression.logistic_regression,new_inp_1)
        f.write("***\n")
        sys.stdout = stdout_
        time.sleep(1)
    f.close()
    f2 = open(filename_res, "w")      # output file
    fp1 = open(filename_out, "r")
    line_count = {}
    module_func = {}
    for line in fp1:
        if "***" in line:
            for key,val in line_count.items():
                f2.write(key + "," + str(val)+"\n")
            f2.write("***\n")
            line_count = {}
        if ".py(" in line:
            module = line.split(".py(")[0]
            codes = line.split("     ")[1:]
            for lc in codes:
                code = lc.rstrip()
                code = code.lstrip()
                if code != "":
                    code = code[:20]
                    code = code.replace(","," ")
                    code = code.replace("\""," ")
                    break
            # func = ""
            # if module in module_func.keys():
            #     func = module_func[module]
            line_no = line.split(".py(")[1].split("):")[0]
            key = module + "-" + line_no + "-" + code
            if key not in line_count.keys():
                line_count[key] = 1
            else:
                line_count[key] += 1
        # if "funcname:" in line:
        #     module_name = line.split(",")[0].split(": ")[-1].rstrip()
        #     func_name = line.split(",")[1].split(": ")[-1].rstrip()
        #     module_func[module_name] = func_name
    f2.close()


if __name__ == '__main__':
    main()

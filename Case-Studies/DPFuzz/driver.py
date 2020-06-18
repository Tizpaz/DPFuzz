import sys
import math
import random
import os
import time
import subprocess32 as subprocess
import numpy as np
from numpy.polynomial import polynomial as P
from sklearn.cluster import KMeans
import xml_parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", help='The name of library')
parser.add_argument("--size", help='The maximum size of inputs')
parser.add_argument("--clusters", help='The number of clusters')
parser.add_argument("--max_iter", help='The maximum number of iterations')
parser.add_argument("--num_param", help='The number of parameters')
parser.add_argument("--size_index", help='The size index(cies)')
args = parser.parse_args()

MAX_SIZE = int(args.size)       # for parameterize value, use the max value from xml file
NUM_CLUSTERS = int(args.clusters)
NUM_ITER =  int(args.max_iter)
NUM_PARAMETERS = int(args.num_param)
SIZE_INDEX = [int(x) for x in args.size_index.split(",")]

MAX_POP_SIZE = 5000
MAX_POP_SIZE_TOT = 20000
Actual_Time = True
INPUT_SIZE = False   # paramterized inputs or not, True for absolute value and False for a parameter value
STEPS_TO_KILL = 200
TRIASL_EACH_ITER = 100
CONSTANT_FACTOR = 2.0
PERCENTAGE_TO_KEEP = 80
DO_CLUSTERING = False
STEPS_TO_DO_CLUSTERING = 1000
DEGREE_TO_FIT = 1
ALLOWED_REMOVE_CLUSTER_PATH = False
MIN_NUM_PATH_PER_CLUST = 4
MIN_NUM_PATH = MIN_NUM_PATH_PER_CLUST * NUM_CLUSTERS

if(args.name == "LogisticRegression"):
    import LogisticRegression
    input_program = LogisticRegression.logistic_regression
    input_program_tree = 'logistic_regression_Params.xml'
elif(args.name == "make_classification"):
    import make_classification
    input_program = make_classification.generate_data_classification
    input_program_tree = 'make_classification_Params.xml'
    STEPS_TO_KILL = 500
    TRIASL_EACH_ITER = 200
elif(args.name == "gen_batches"):
    import gen_batches
    input_program = gen_batches.generate_batches
    input_program_tree = 'gen_batches_Params.xml'
    STEPS_TO_KILL = 100
    TRIASL_EACH_ITER = 200
    CONSTANT_FACTOR = 10.0
elif(args.name == "GaussianProc"):
    import GaussianProc
    input_program = GaussianProc.GaussianProcess
    input_program_tree = 'Gaussian_Proc_Params.xml'
    STEPS_TO_KILL = 500
    TRIASL_EACH_ITER = 200
elif(args.name == "minibatch_kmeans"):
    import minibatch_kmeans
    input_program = minibatch_kmeans.minibatch_kmeans
    input_program_tree = 'minibatch_kmeans_Params.xml'
    MAX_POP_SIZE_TOT = 100000
    STEPS_TO_KILL = 1000
    TRIASL_EACH_ITER = 500
elif(args.name == "TreeRegressor"):
    import TreeRegressor
    input_program = TreeRegressor.TreeRegress
    input_program_tree = 'TreeRegressor_Params.xml'
    MAX_POP_SIZE_TOT = 10000
elif(args.name == "Discriminant_Analysis"):
    import Discriminant_Analysis
    input_program = Discriminant_Analysis.disc_analysis
    input_program_tree = 'Discriminant_Analysis_Params.xml'
elif(args.name == "Decision_Tree_Classifier"):
    import Decision_Tree_Classifier
    input_program = Decision_Tree_Classifier.DecisionTree
    input_program_tree = 'Decision_Tree_Classifier_Params.xml'
    STEPS_TO_KILL = 500
    TRIASL_EACH_ITER = 200

class Coverage(object):
    # Trace function
    def traceit(self, frame, event, arg):
        if self.original_trace_function is not None:
            self.original_trace_function(frame, event, arg)

        if event == "line":
            function_name = frame.f_code.co_name
            lineno = frame.f_lineno
            self._trace.append((function_name, lineno))

        return self.traceit

    def __init__(self):
        self._trace = []

    # Start of `with` block
    def __enter__(self):
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.traceit)
        return self

    # End of `with` block
    def __exit__(self, exc_type, exc_value, tb):
        sys.settrace(self.original_trace_function)

    def trace(self):
        """The list of executed lines, as (function_name, line_number) pairs"""
        return self._trace

    def coverage(self):
        """The set of executed lines, as (function_name, line_number) pairs"""
        path = set()
        for str, line in self.trace():
            path.add(line)
        path_sign = 0
        for line in path:
            path_sign ^= line
        return path_sign, len(self.trace())

from functools import wraps
import errno
import signal

class TimeoutError(Exception):
    pass

def timeout(seconds=10):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError("time_error")

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


class Runner(object):
    # Test outcomes
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"

    def __init__(self):
        """Initialize"""
        pass

    def run(self, inp):
        """Run the runner with the given input"""
        return (inp, Runner.UNRESOLVED)

class PrintRunner(Runner):
    def run(self, inp):
        """Print the given input"""
        print(inp)
        return (inp, Runner.UNRESOLVED)

class ProgramRunner(Runner):
    def __init__(self, program):
        """Initialize.  `program` is a program spec as passed to `subprocess.run()`"""
        self.program = program
        self.time_cost = 0
    def run_process(self, inp=""):
        """Run the program with `inp` as input.  Return result of `subprocess.run()`."""
        res = subprocess.run(self.program,
                              input=inp,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
        return res

    def run(self, inp=""):
        """Run the program with `inp` as input.  Return test outcome based on result of `subprocess.run()`."""
        result = self.run_process(inp)

        if result.returncode == 0:
            outcome = self.PASS
        elif result.returncode < 0:
            outcome = self.FAIL
        else:
            outcome = self.UNRESOLVED


class Fuzzer(object):
    def __init__(self):
        self.time_cost = 0
        pass

    def fuzz(self):
        """Return fuzz input"""
        return ""

    # @timeout(5)
    def run(self, runner=Runner()):
        """Run `runner` with fuzz input"""
        if Actual_Time:
            start = time.time() * 1000000
            res = runner.run(self.fuzz())
            end = time.time() * 1000000
            self.time_cost = end - start
            return res
        else:
            res = runner.run(self.fuzz())
            return res

    def runs(self, runner=PrintRunner(), trials=10):
        """Run `runner` with fuzz input, `trials` times"""
        # Note: the list comprehension below does not invoke self.run() for subclasses
        # return [self.run(runner) for i in range(trials)]
        outcomes = []
        for i in range(trials):
            outcomes.append(self.run(runner))
        return outcomes

class MutationFuzzer(Fuzzer):
    def __init__(self, seed, min_mutations=2, max_mutations=10):
        self.seed = seed
        self.min_mutations = min_mutations
        self.max_mutations = max_mutations
        self.reset()

    def reset(self):
        self.population = self.seed
        self.seed_index = 0

    def insert_random_character(self, s):
        """Returns s with a random character inserted"""
        pos = random.randint(0, len(s))
        random_character = chr(random.randint(33, 127))
        # print("Inserting", repr(random_character), "at", pos)
        return s[:pos] + random_character + s[pos:]

    def delete_random_character(self, s):
        """Returns s with a random character deleted"""
        if s == "":
            return s
        pos = random.randint(0, len(s) - 1)
        # print("Deleting", repr(s[pos]), "at", pos)
        return s[:pos] + s[pos + 1:]

    def flip_random_character(self, s):
        """Returns s with a random bit flipped in a random position"""
        if s == "":
            return s

        pos = random.randint(0, len(s) - 1)
        c = s[pos]
        if ord(c) <= 39:
            new_c = chr(random.randint(48, 57))
        else:
            bit = 1 << random.randint(0, 6)
            new_c = chr(ord(c) ^ bit)
        # print("Flipping", bit, "in", repr(c) + ", giving", repr(new_c))
        return s[:pos] + new_c + s[pos + 1:]

    def rnd_subset_reorder(self, s):
        """randomly change the order of a subset of the input bytes"""
        if s == "":
            return s
        pos = random.randint(0, len(s) - 1)
        range = random.randint(pos, len(s) - 1)
        substr = s[pos:range]
        substr_rev = substr[::-1]
        return s[:pos] + substr_rev + s[range + 1:]

    def flip_digit_character(self, s):
        """Returns s with a random bit flipped in a random position"""
        if s == "":
            return s
        poss_chars_indx = []
        indx = 0
        for char in s:
            if ord(char) >= 48 and ord(char) <= 57:
                poss_chars_indx.append(indx)
            indx = indx + 1
        pos = random.choice(poss_chars_indx)
        c = s[pos]
        bit = 1 << random.randint(0, 6)
        new_c = chr(ord(c) ^ bit)
        # print("Flipping", bit, "in", repr(c) + ", giving", repr(new_c))
        return s[:pos] + new_c + s[pos + 1:]

    def SwapByte(self,data):
        fuzzed = ''
        if len(data) < 2:
            return data

        rnd1 = random.randint(0, len(data) - 1)
        if rnd1 >= 1:
            rnd2 = random.randint(0, rnd1 - 1)
        elif rnd1 + 1 <= len(data) - 1:
            rnd2 = random.randint(rnd1 + 1, len(data) - 1)

        min_rnd = min(rnd1, rnd2)
        max_rnd = max(rnd1, rnd2)

        byte1 = data[min_rnd]
        byte2 = data[max_rnd]

        fuzzed = data[:min_rnd]
        fuzzed += byte2
        fuzzed += data[min_rnd + 1:max_rnd]
        fuzzed += byte1
        fuzzed += data[max_rnd + 1:]

        return fuzzed

    def SwapWord(self,data):
        fuzzed = ''
        if len(data) < 4:
            return data

        rnd1 = random.randint(0, len(data) - 2)

        if rnd1 >= 2:
            rnd2 = random.randint(0, rnd1 - 2)
        elif rnd1 + 2 <= len(data) - 2:
            rnd2 = random.randint(rnd1 + 2, len(data) - 2)
        else:
            return data

        min_rnd = min(rnd1, rnd2)
        max_rnd = max(rnd1, rnd2)

        word1 = data[min_rnd:min_rnd + 2]

        word2 = data[max_rnd:max_rnd + 2]

        fuzzed = data[:min_rnd]
        fuzzed += word1
        fuzzed += data[min_rnd + 2:max_rnd]
        fuzzed += word2
        fuzzed += data[max_rnd + 2:]

        return fuzzed

    def  ByteNullifier(self, data):
        fuzzed = ''
        if len(data) == 0:
            return data
        index = random.randint(0, len(data) - 1)

        fuzzed = '%s\x00%s' % (data[:index], data[index + 1:])
        return fuzzed

    def IncreaseByOneMutator(self, data, howmany=1):
        if len(data) == 0:
            return data
#        howmany = random.choice(range(1,4))
        if len(data) < howmany:
            howmany = random.randint(1, len(data))

        fuzzed = data

        for _ in xrange(howmany):
            index = random.randint(0, len(data) - 1)
            if ord(data[index]) != 0xFF:
                fuzzed = '%s%c%s' % (
                        data[:index],
                        ord(data[index]) + 1,
                        data[index + 1:]
                    )
            else:
                fuzzed = '%s\x00%s' % (
                        data[:index],
                        data[index + 1:]
                        )

            data = fuzzed

        return fuzzed

    def DecreaseByOneMutator(self, data, howmany=1):
        if len(data) == 0:
            return data
#        howmany = random.choice(range(1,4))
        if len(data) < howmany:
            howmany = random.randint(0, len(data) - 1)

        fuzzed = data
        for _ in xrange(howmany):
            index = random.randint(0, len(data) - 1)
            if ord(data[index]) != 0:
                fuzzed = '%s%c%s' % (
                        data[:index],
                        ord(data[index]) - 1,
                        data[index + 1:]
                    )
            else:
                fuzzed = '%s\xFF%s' % (
                    data[:index],
                    data[index + 1:]
                )
            data = fuzzed
        return fuzzed

    def ProgressiveIncreaseMutator(self, data, howmany=8):
        if len(data) < 2:
            return data
        howmany = random.choice(range(2,len(data)+1))
        index = random.randint(0, len(data) - howmany)
        buf = ''
        fuzzed = ''

        for curr in xrange(index, index + howmany):
            addend = 1
            if addend + ord(data[curr]) > 0xFF:
                addend -= 0xFF
            buf += chr(ord(data[curr]) + addend)

        fuzzed = '%s%s%s' % (data[:index], buf, data[index + howmany:])
        return fuzzed

    def ProgressiveDecreaseMutator(self, data, howmany=8):
        if len(data) < 2:
            return data
        howmany = random.choice(range(2,len(data)+1))
        index = random.randint(0, len(data) - howmany)
        buf = ''
        fuzzed = ''

        for curr in xrange(index, index + howmany):
            dec = 1
            if ord(data[curr]) >= dec:
                buf += chr(ord(data[curr]) - dec)
            else:
                buf += chr(dec - ord(data[curr]))

        fuzzed = '%s%s%s' % (data[:index], buf, data[index + howmany:])
        return fuzzed

    def SetHighBitFromByte(self, data):
        fuzzed = ''

        if len(data) > 0:
            index = random.randint(0, len(data) - 1)
            byte = ord(data[index])
            byte |= 0x80
            fuzzed = data[:index]
            fuzzed += chr(byte)
            fuzzed += data[index + 1:]

        return fuzzed

    def DuplicateByte(self, data, howmany=1):
        fuzzed = ''
        if len(data) < 1:
            return data

        for _ in xrange(howmany):
            index = random.randint(0, len(data) - 1)
            byte = data[index]
            fuzzed = data[:index]
            fuzzed += byte
            fuzzed += data[index:]

        return fuzzed

    def crossover_2(self, s):
        if len(self.population.keys()) == 0:
            return s;
        key = random.choice(self.population.keys())
        s1 = random.choice(self.population[key])
        if len(s1) <= 1 or len(s) <= 1:
            return s;
        # print("original string is: " + s)
        pos = random.randint(0, len(s) - 1)
        pos1 = random.randint(0, len(s1) - 1)
        father, mother = s[:pos], s1[pos1+1:]
        index1 = random.randint(1, len(s) - 1)
        index2 = random.randint(1, len(s1) - 1)
        if index1 > index2: index1, index2 = index2, index1
        child1 = father[:index1] + mother[index1:index2] + father[index2:]
        child2 = mother[:index1] + father[index1:index2] + mother[index2:]
        # print("crossover string is: " + child1 + " " + child2)
        return child1 + child2

    def mutate(self, s):
        """Return s with a random mutation applied"""
        mutators = [
            self.delete_random_character,
            self.insert_random_character,
            self.flip_random_character,
           self.rnd_subset_reorder,
           self.SwapByte,
#            self.SwapWord,
            self.IncreaseByOneMutator,
            self.DecreaseByOneMutator,
#            self.ByteNullifier,
#            self.ProgressiveIncreaseMutator,
#            self.ProgressiveDecreaseMutator,
            self.SetHighBitFromByte
#            self.DuplicateByte
            ]
        mutator = random.choice(mutators)
        index_s = mutators.index(mutator)
        # specific for input with array separated with space
        s = " ".join(mutator(s).split())
        return s, index_s

    def action_RL(self, s, a):
        """Return s with an action from RL"""
        mutators = [
            self.delete_random_character,
            self.insert_random_character,
            self.flip_random_character,
            self.rnd_subset_reorder,
            self.crossover_2,
            self.SwapByte,
            self.SwapWord,
            self.IncreaseByOneMutator,
            self.DecreaseByOneMutator,
#            self.ByteNullifier,
            self.ProgressiveIncreaseMutator,
            self.ProgressiveDecreaseMutator,
            self.SetHighBitFromByte
#            self.DuplicateByte
        ]
        if a == 4:
            s, _ = self.mutate(s)
        mutator = mutators[a]
        # specific for input with array separated with space
        s = " ".join(mutator(s).split())
        return s

    def create_candidate(self, mutate = True, cur_act = 0, e = 0.01, m_p = 0.2):
        a_index = -1
        if mutate:
            key = random.choice(self.population.keys())
            sum_costs = np.sum(self.coverages_seen[key])
            prob = [float(x/sum_costs) for x in self.coverages_seen[key]]
            if np.sum(prob) > 0.999:
                candidate = np.random.choice(self.population[key],p=prob)
            else:
                candidate = np.random.choice(self.population[key])
            trials = random.randint(self.min_mutations, self.max_mutations)
            for i in range(trials):
                candidate, a_index = self.mutate(candidate)
            if random.random() < m_p:       # with this probability, do crossover
                candidate = self.crossover_2(candidate)
            # key = random.choice(self.population.keys())
            # candidate = random.choice(self.population[key])
            # trials = random.randint(self.min_mutations, self.max_mutations)
            # for i in range(trials):
            #     candidate, a_index = self.mutate(candidate)
            # if random.random() < m_p:       # with this probability, do crossover
            #     candidate = self.crossover_2(candidate)
        else:
            candidate = self.inp
            if random.random() < e:       # with small probability do mutations!
                trials = random.randint(self.min_mutations, self.max_mutations)
                for i in range(trials):
                    candidate, a_index = self.mutate(candidate)
                if random.random() < m_p:       # with this probability, do crossover
                    candidate = self.crossover_2(candidate)
            else:
                candidate = self.action_RL(candidate, cur_act)
                a_index = cur_act
        return candidate, a_index

    def fuzz(self, mutate = True, cur_act = 0, e_prob = 0.01):
        if self.seed_index < len(self.seed):
            # Still seeding
            self.inp = self.seed[self.seed_index]
            self.seed_index += 1
        else:
            # Mutating
            self.inp, self.action = self.create_candidate(mutate, cur_act, e_prob)
        # print("current input is: " + self.inp)
        return self.inp

class FunctionRunner(Runner):
    def __init__(self, function):
        """Initialize.  `function` is a function to be executed"""
        self.function = function


    def run_function(self, inp):
        return self.function(inp)

    def run(self, inp):
        try:
            result = self.run_function(inp)
            outcome = self.PASS
        except Exception:
            result = None
            outcome = self.FAIL

        return result, outcome

class FunctionCoverageRunner(FunctionRunner):
    def run_function(self, inp):
        with Coverage() as cov:
            try:
                result = super(FunctionCoverageRunner,self).run_function(inp)
            except Exception as exc:
                self._coverage = cov.coverage()
                raise exc

        self._coverage = cov.coverage()
        return result

    def coverage(self):
        return self._coverage

class MutationCoverageFuzzer(MutationFuzzer):
    def reset(self):
        super(MutationCoverageFuzzer,self).reset()
        self.coverages_seen = {}
        self.population = {}
        self.coverages_seen_tot = {}
        self.population_tot = {}
        self.worst_costs = {}
        self.last_update = {}
        self.removed_path = []
        self.len_inputs = {}
        self.path_cluster = {}
        self.cluster_paths = {}
        self.updated_since_clustered = {}
        self.eval_functions = {}  
        self.new_coverage = 0
        self.action = -1
        self.inp = ""
        self.num_inp = 0

    def run(self, runner):
        """Run function(inp) while tracking coverage.
           If we reach new coverage,
           add inp to population and its coverage to population_coverage
        """
        try:
            result, outcome = super(MutationCoverageFuzzer,self).run(runner)
        except TimeoutError as error:
            print("Caght an error!")
            return ""
        key_path = runner.coverage()[0]
        if Actual_Time:
            val_cost = self.time_cost
        else:
            val_cost = runner.coverage()[1]
        if key_path in self.removed_path:
            return ""
        self.new_coverage = val_cost
        self.num_inp += 1

        # Do Fitting and Clustering
        if INPUT_SIZE and DO_CLUSTERING and self.num_inp % STEPS_TO_DO_CLUSTERING == 0:
            for key_path in self.coverages_seen.keys():
                if key_path in self.updated_since_clustered.keys() and self.updated_since_clustered[key_path] == False:
                    continue;
                X = [len(x) for x in self.population[key_path]]
                y = [x for x in self.coverages_seen[key_path]]
                if len(set(X)) >= DEGREE_TO_FIT+1:
                    vals, stats_res = P.polyfit(X,y,DEGREE_TO_FIT,full=True)
                    self.model_fit[key_path] = (vals,stats_res[0])

            path_orders = []

            for key_path in self.model_fit.keys():
                if self.updated_since_clustered[key_path] == False:
                    path_orders.append(key_path)
                else:
                    self.updated_since_clustered[key_path]=False
                    path_orders.append(key_path)
                    self.eval_functions[key_path] = [self.model_fit[key_path][0][1] * i_size + self.model_fit[key_path][0][0] for i_size in range(1,MAX_SIZE)]
            eval_functions_array = np.array([self.eval_functions[key] for key in self.eval_functions.keys()])
            if len(self.eval_functions.keys()) >= NUM_CLUSTERS:
                kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=1).fit(eval_functions_array)
                self.cluster_paths = {}
                for i, clust in enumerate(kmeans.labels_):
                    self.path_cluster[path_orders[i]] = clust
                    if clust in self.cluster_paths.keys():
                        self.cluster_paths[clust].append(path_orders[i])
                    else:
                        self.cluster_paths[clust] = [path_orders[i]]

        if DO_CLUSTERING and self.num_inp % STEPS_TO_KILL == 0:
            max_val = -1
            max_path = -1
            max_clust = -1
            for cluster in self.cluster_paths.keys():
                for x in self.cluster_paths[cluster]:
                    if self.worst_costs[x] > max_val:
                        max_val = self.worst_costs[x]
                        max_path = x
                        max_clust = cluster
            for key_path in self.coverages_seen.keys():
                if key_path not in self.model_fit.keys():
                    if self.worst_costs[key_path] < max_val:
                        self.population.pop(key_path, None)
                        self.coverages_seen.pop(key_path, None)
                        self.worst_costs.pop(key_path, None)
                        self.last_update.pop(key_path, None)
                        self.removed_path.append(key_path)

        # size limitation based on the number of arguments
        inp_num_args = self.inp.split(" ")
        is_interesting = True
        if not key_path in self.coverages_seen.keys():
            self.coverages_seen[key_path] = [val_cost]
            self.population[key_path] = [self.inp]
            self.worst_costs[key_path] = val_cost
            self.last_update[key_path] = self.num_inp
            if DO_CLUSTERING:
                self.updated_since_clustered[key_path] = True
        # this is based on the length of input string
        elif outcome == Runner.PASS and INPUT_SIZE and self.new_coverage > np.percentile(self.coverages_seen[key_path],PERCENTAGE_TO_KEEP) and len(self.inp) <= MAX_SIZE and len(self.inp) <= np.median(map(len,self.population[key_path])) + CONSTANT_FACTOR * np.sqrt(np.median(map(len,self.population[key_path]))):
            self.population[key_path].append(self.inp)
            self.coverages_seen[key_path].append(self.new_coverage)
        # this is based on the arguments in the input
        elif outcome == Runner.PASS and not INPUT_SIZE and self.new_coverage > np.percentile(self.coverages_seen[key_path],PERCENTAGE_TO_KEEP) and len(inp_num_args) <= NUM_PARAMETERS:
            arr = xml_parser.xml_parser(input_program_tree,self.inp)
            # choose the size parame
            if(len(arr) == NUM_PARAMETERS):
                len_inp_pop = 1
                for index in SIZE_INDEX:
                    len_inp_pop *= arr[index]
            else:
                len_inp_pop = 0
                is_interesting = False
            if is_interesting and key_path not in self.len_inputs.keys():
                self.population[key_path].append(self.inp)
                self.coverages_seen[key_path].append(self.new_coverage)
                self.len_inputs[key_path] = len_inp_pop
            elif is_interesting and len_inp_pop <= self.len_inputs[key_path] + CONSTANT_FACTOR * np.sqrt(self.len_inputs[key_path]):
                self.population[key_path].append(self.inp)
                self.coverages_seen[key_path].append(self.new_coverage)
                if len_inp_pop > self.len_inputs[key_path]:
                    self.len_inputs[key_path] = len_inp_pop
            else:
                is_interesting = False
        # not interesting
        else:
            is_interesting = False

        if is_interesting and self.worst_costs[key_path] < val_cost:
            self.worst_costs[key_path] = val_cost
            self.last_update[key_path] = self.num_inp
            if DO_CLUSTERING:
                self.updated_since_clustered[key_path] = True 
        elif self.num_inp - self.last_update[key_path] > STEPS_TO_KILL and len(self.population) > MIN_NUM_PATH and self.worst_costs[key_path] <= np.percentile(self.worst_costs.values(),50):
            min = self.worst_costs[key_path]
            min_path = key_path
            for path_other in self.worst_costs.keys():
                if self.worst_costs[path_other] < min:
                    min = self.worst_costs[path_other]
                    min_path = path_other
            if key_path == min_path:
                self.population.pop(key_path, None)
                self.coverages_seen.pop(key_path, None)
                self.worst_costs.pop(key_path, None)
                self.last_update.pop(key_path, None)
                self.removed_path.append(key_path)
                return ""
            else:
                self.population.pop(min_path, None)
                self.coverages_seen.pop(min_path, None)
                self.worst_costs.pop(min_path, None)
                self.last_update.pop(min_path, None)
                self.removed_path.append(min_path)

        elif DO_CLUSTERING and key_path in self.path_cluster.keys() and ALLOWED_REMOVE_CLUSTER_PATH and self.num_inp - self.last_update[key_path] > STEPS_TO_KILL and len(self.cluster_paths[self.path_cluster[key_path]]) > MIN_NUM_PATH_PER_CLUST:
            clust = self.path_cluster[key_path]
            for x in self.cluster_paths[clust]:
                if self.worst_costs[x] > self.worst_costs[key_path]:
                    self.cluster_paths[clust].remove(key_path)
                    self.path_cluster.pop(key_path,None)
                    self.model_fit.pop(key_path,None)
                    self.updated_since_clustered.pop(key_path,None)
                    self.eval_functions.pop(key_path,None)
                    self.population.pop(key_path, None)
                    self.coverages_seen.pop(key_path, None)
                    self.worst_costs.pop(key_path, None)
                    self.last_update.pop(key_path, None)
                    self.removed_path.append(key_path)
                    break

        if is_interesting == False:
            return ""
        if len(self.population[key_path]) > MAX_POP_SIZE:
            if INPUT_SIZE:
                indicies = sorted(range(len(self.coverages_seen[key_path])), key=lambda i: self.coverages_seen[key_path][i]/(len(self.population[key_path][i])))[-MAX_POP_SIZE/8:]
                self.coverages_seen_new = []
                self.population_new = []
                for index in indicies:
                    self.population_new.append(self.population[key_path][index])
                    self.coverages_seen_new.append(self.coverages_seen[key_path][index])
                self.population[key_path] =  self.population_new
                self.coverages_seen[key_path] =  self.coverages_seen_new
            else:
                indicies = []
                for k in range(MAX_POP_SIZE/20):
                    indicies.append(random.randint(0,7*MAX_POP_SIZE/8))
                self.coverages_seen_new = []
                self.population_new = []
                for index in indicies:
                    self.population_new.append(self.population[key_path][index])
                    self.coverages_seen_new.append(self.coverages_seen[key_path][index])
                self.population_new = self.population_new + self.population[key_path][-MAX_POP_SIZE/8:]
                self.coverages_seen_new = self.coverages_seen_new + self.coverages_seen[key_path][-MAX_POP_SIZE/8:]
                self.population[key_path] =  self.population_new
                self.coverages_seen[key_path] =  self.coverages_seen_new
        return result

def population_coverage(population, function):
    cumulative_coverage = []
    prev_cov = 0
    all_coverage = []

    for s in population:
        with Coverage() as cov:
            try:
                function(s)
            except:
                pass
        all_coverage.append(cov.coverage())
        cumulative_coverage.append(cov.coverage() - prev_cov)
        prev_cov = cov.coverage()

    return all_coverage, cumulative_coverage

def run_driver(seed_input):

    inp_program_instr = FunctionCoverageRunner(input_program)
    mutation_fuzzer = MutationCoverageFuzzer(seed=seed_inputs, min_mutations = 1, max_mutations = 5)
    for i in range(NUM_ITER):
        mutation_fuzzer.runs(inp_program_instr, trials=TRIASL_EACH_ITER)
        for key in mutation_fuzzer.coverages_seen.keys():
            print("The path key is: " + str(key))
            max_int = mutation_fuzzer.coverages_seen[key].index(max(mutation_fuzzer.coverages_seen[key]))
            print("step: " + str(i+1))
            print(max(mutation_fuzzer.coverages_seen[key]))
            print(mutation_fuzzer.population[key][max_int])

    # print("Best input and coverage overall!")
    # max_int = mutation_fuzzer.coverages_seen.index(max(mutation_fuzzer.coverages_seen))
    # print(max(mutation_fuzzer.coverages_seen))
    # print(mutation_fuzzer.population[max_int])
    # store the results!
    # based on the length of inputs!
    if INPUT_SIZE:
        f1 = open("complexity_driver_" + str(input_program).split(" ")[1] +".csv", "w")
        for key in mutation_fuzzer.coverages_seen.keys():
            included_lines = []
            index = 0
            for inp_pop in mutation_fuzzer.population[key]:
                if len(inp_pop) not in included_lines:
                    avail_ind = [i for i in range(0,len(mutation_fuzzer.population[key])) if len(mutation_fuzzer.population[key][i]) == len(inp_pop)]
                    c1 = np.max([mutation_fuzzer.coverages_seen[key][i] for i in avail_ind])
                    f1.write(str(len(inp_pop)) + "," + str(c1) + "," + str(key) +"\n")
                    included_lines.append(len(inp_pop))
                index += 1
        f1.close()
    # based on some parts of inputs
    else:
        f1 = open("complexity_driver_" + str(input_program).split(" ")[1] +".csv", "w")
        for key in mutation_fuzzer.coverages_seen.keys():
            included_lines = {}
            ind_cnt = 0
            for inp_pop in mutation_fuzzer.population[key]:
                arr = xml_parser.xml_parser(input_program_tree,inp_pop)
                # choose the size parameter
                if(len(arr) == NUM_PARAMETERS):
                    len_inp_pop = 1
                    for index in SIZE_INDEX:
                        len_inp_pop *= arr[index]
                else:
                    ind_cnt += 1
                    continue
                if len_inp_pop not in included_lines:
                    c1 = mutation_fuzzer.coverages_seen[key][ind_cnt]
                    f1.write(str(arr) + "," + str(len_inp_pop) + "," + str(c1) + "," + str(key) +"\n")
                    included_lines[len_inp_pop] = c1
                else:
                    c1 = mutation_fuzzer.coverages_seen[key][ind_cnt]
                    max_val = included_lines[len_inp_pop]
                    if max_val < c1:
                        f1.write(str(arr) + "," + str(len_inp_pop) + "," + str(c1) + "," + str(key) +"\n")
                        included_lines[len_inp_pop] = c1
                ind_cnt += 1
        f1.close()

if __name__ == "__main__":
    #seed_inputs = ["1 3","12 9 10 5","17 15 20"]
    #seed_inputs = ["0.11 3 4 10","-1.16 5 3 50","2.12 7 4 101"]
    # seed_inputs = ["50 1000 50 1","150 1000 60 0","250 1000 70 0"]
    #seed_inputs = ["400 50 10 0","430 50 10 0","100 50 10 0","150 50 10 0","180 50 10 0","210 50 10 0","240 50 10 0","275 50 10 0","300 50 10 0","330 50 10 0","350 50 10 0","450 50 10 0","500 50 10 0","520 50 10 0","530 50 10 0","550 50 10 0","580 50 10 0","610 50 10 0","650 50 10 0","700 50 10 0","719 50 10 0","745 50 10 0","800 50 10 0","820 50 10 0","850 50 10 0","870 50 10 0","900 50 10 0","920 50 10 0","950 50 10 0","970 50 10 0","990 50 10 0","1000 50 10 0"]
    # seed_inputs = ["1000 200 10.0 10 1 1 0","2000 200 1.0 10 1 1 1","5000 200 0.1 10 1 1 0"]
    # seed_inputs = ["10000 10 2 5 1 2 1 1 0.0001 1.0 0 1.0 100 0"]
    # seed_inputs = ["1000 10 2 2 0 1 0 1 0 0 0.01 0.0"]
    # seed_inputs = ["50 2 0 2 0 0 10 0.1 0 3 0.01 0 2.0 1.0 1"]
    seed_inputs = ["1000 5 5 2 50 0 1 0 0 0 0"]

    run_driver(seed_inputs)

#!/home/hecher/miniconda3/bin/python3
"""
Main module providing the application logic.
"""

import matplotlib.pyplot as plt
import sys
# from textwrap import dedent
from collections import OrderedDict
import clingo
#import clingoext
from pprint import pprint
import networkx as nx
#import lib.htd_validate
#from groundprogram import ClingoRule
import os
import inspect
import logging
import subprocess
import math
from itertools import product

# set library path

# library paths 
src_path = os.path.abspath(os.path.realpath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.realpath(os.path.join(src_path, '../..')))

src_path = os.path.realpath(os.path.join(src_path, '../../lib'))

libs = ['htd_validate', 'clingoparser', 'nesthdb', 'htd']

if src_path not in sys.path:
    for lib in libs:
        sys.path.insert(0, os.path.join(src_path, lib))

#print(sys.path)

logger = logging.getLogger("asp2sat")
logging.basicConfig(format='[%(levelname)s] %(name)s: %(message)s', level="INFO")

from htd_validate.utils import hypergraph, graph

import clingoext
from clingoext import ClingoRule
#from htd_validate.decompositions import *
from dpdb import reader
from dpdb import treedecomp
from dpdb.problems.sat_util import *
from dpdb.writer import StreamWriter

#import wfParse

class AppConfig(object):
    """
    Class for application specific options.
    """

    def __init__(self):
        self.eclingo_verbose = 0


class Application(object):
    """
    Application class that can be used with `clingo.clingo_main` to solve CSP
    problems.
    """

    def __init__(self):
        self.program_name = "clingoext"
        self.version = "0.0.1"
        self.config = AppConfig()
        # store the weights of literals here
        self._weights = {}
        # store the clauses here
        self._clauses = []
        # store the projected variables
        self._projected = {}
        # remember one variable for x <_t x' regardless of t
        self._lessThan = {}
        self._done = {}

    def _read(self, path):
        if path == "-":
            return sys.stdin.read()
        with open(path) as file_:
            return file_.read()

    def primalGraph(self):
        return self._graph

    def var2idx(self, var):
        sym = clingo.parse_term(var)
        if sym in self.control.symbolic_atoms:
            lit = self.control.symbolic_atoms[sym].literal
            return self._atomToVertex[lit]
        return 0

    def new_var(self, name):
        self._max += 1
        self._nameMap[self._max] = name
        return self._max

    def remove_tautologies(self):
        tmp = []
        for o in self.control.ground_program.objects:
            if isinstance(o, ClingoRule) and set(o.head).intersection(set(o.body)) == set():
                tmp.append(o)
        self.control.ground_program.objects = tmp

    def _generatePrimalGraph(self):
        self.remove_tautologies()
        self._graph = hypergraph.Hypergraph()
        self._program = []
        self._atomToVertex = {} # htd wants succinct numbering of vertices / no holes
        self._vertexToAtom = {} # inverse mapping of _atomToVertex 
        self._max = 0
        self._nameMap = {}
        unary = []
        for o in self.control.ground_program.objects:
            if isinstance(o, ClingoRule):
                o.atoms = set(o.head)
                o.atoms.update(tuple(map(abs, o.body)))
                self._program.append(o)
                if len(o.atoms) > 1:
                    for a in o.atoms.difference(self._atomToVertex):	# add mapping for atom not yet mapped
                        self._atomToVertex[a] = self.new_var(str(a))
                        self._vertexToAtom[self._max] = a
                    self._graph.add_hyperedge(tuple(map(lambda x: self._atomToVertex[x], o.atoms)))
                else:
                    unary.append(o)
        for o in unary:
            for a in o.atoms.difference(self._atomToVertex):	# add mapping for atom not yet mapped
                self._atomToVertex[a] = self.new_var(str(a))
                self._vertexToAtom[self._max] = a

        self._projected = set(range(1, self._max + 1))
        #for sym in self.control.symbolic_atoms:
        #    print(self._atomToVertex[sym.literal], sym.symbol)
        #    print(sym.literal, sym.symbol)


    def _computeComponents(self):
        self.dep = nx.DiGraph()
        for r in self.control.ground_program.objects:
            #print(r)
            if isinstance(r, ClingoRule):
                for a in r.head:
                    for b in r.body:
                        if b > 0:
                            self.dep.add_edge(a, b)
        #print(self.dep)
        comp = nx.algorithms.strongly_connected_components(self.dep)
        self._components = list(comp)
        self._condensation = nx.algorithms.condensation(self.dep, self._components)
        #print(self._components)


    def _decomposeGraph(self):
        # Run htd
        p = subprocess.Popen([os.path.join(src_path, "htd/bin/htd_main"), "--seed", "12342134", "--input", "hgr"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        #logger.info("Parsing input file")
        #input = problem.prepare_input(file)
        #if "gr_file" in kwargs and kwargs["gr_file"]:
        #    logger.info("Writing graph file")
        #    with FileWriter(kwargs["gr_file"]) as fw:
        #        fw.write_gr(*input)
        logger.info("Running htd")
        #with open('graph.txt', mode='wb') as file_out:
        #    self._graph.write_graph(file_out, dimacs=False, non_dimacs="tw")
        self._graph.write_graph(p.stdin, dimacs=False, non_dimacs="tw")
        p.stdin.close()
        #print(p.stdout.readlines())
        tdr = reader.TdReader.from_stream(p.stdout)
        p.wait()
        logger.info("TD computed")
        self._td = treedecomp.TreeDecomp(tdr.num_bags, tdr.tree_width, tdr.num_orig_vertices, tdr.root, tdr.bags, tdr.adjacency_list, None)
        logger.info(f"Tree decomposition #bags: {self._td.num_bags} tree_width: {self._td.tree_width} #vertices: {self._td.num_orig_vertices} #leafs: {len(self._td.leafs)} #edges: {len(self._td.edges)}")
        #logger.info(self._td.nodes)


    # write a single clause
    # connective == 0 -> and, == 1 -> or, == 2 -> impl, == 3 -> iff, == 4 -> *, == 5 -> +
    def clause_writer(self, p, c1 = 0, c2 = 0, connective = 0):
        if c1 == 0:
            c1 = self.new_var(f"{p}'sc[0]")
        if c2 == 0:
            c2 = self.new_var(f"{p}'sc[1]")
        if connective == 0:
            self._clauses.append([-p, c1])
            self._clauses.append([-p, c2])
            self._clauses.append([p, -c1, -c2])
        if connective == 1:
            self._clauses.append([p, -c1])
            self._clauses.append([p, -c2])
            self._clauses.append([-p, c1, c2])
        if connective == 2:
            self._clauses.append([p, c1])
            self._clauses.append([p, -c2])
            self._clauses.append([-p, -c1, c2])
        if connective == 3:
            c = self.clause_writer(p, c1 = self.new_var(f"{c1}->{c2}"), c2 = self.new_var(f"{c2}->{c1}"))
            self.clause_writer(c[0], c1 = c1, c2 = c2, connective = 2)
            self.clause_writer(c[1], c1 = c2, c2 = c1, connective = 2)
        if connective == 4:
            self._clauses.append([-p, c1])
            self._clauses.append([-p, c2])
            self._clauses.append([p, -c1])
            self._clauses.append([p, -c2])
        if connective == 5:
            self._clauses.append([p, -c1])
            self._clauses.append([p, -c2])
            self._clauses.append([-p, c1, c2])
            self._clauses.append([-p, -c1, -c2])
        return (c1, c2)

    # a subroutine to generate x < x'
    def generateLessThan(self, x, xp, local = False, node = None):
        # setup and check if this has already been handled
        if local:
            if node == None:
                logger.error("node cannot be None!")
                exit(1)
            if not (x,xp) in self._lessThan:
                self._lessThan[(x,xp)] = self.new_var(f"{x}<{xp}")
                self._done[(x,xp)] = set()
            if node in self._done[(x,xp)]:
                return self._lessThan[(x,xp)]
            self._done[(x,xp)].add(node)
        else:
            if not (x,xp) in self._lessThan:
                self._lessThan[(x,xp)] = self.new_var(f"{x}<{xp}")
            else:
                return self._lessThan[(x,xp)]

        # check if x and xp are in differens components
        xs_comp = self._condensation.graph["mapping"][x]
        xps_comp = self._condensation.graph["mapping"][xp]
        if xs_comp != xps_comp:
            # determine which is in the higher component
            if nx.algorithms.shortest_paths.generic.has_path(self._condensation, xs_comp, xps_comp):
                self._clauses.append([-self._lessThan[(x,xp)]])
            elif nx.algorithms.shortest_paths.generic.has_path(self._condensation, xps_comp, xs_comp):
                self._clauses.append([self._lessThan[(x,xp)]])
            else: # there is no connection between these at all. should not occur.
                logger.error("No connection between nodes that need to be connected!")
                exit(1)
            return self._lessThan[(x,xp)]

        # x and xp are in the same component 
        # obtain the bits and their number
        if local:
            count = len(self.bits[node][x])
            x_bits = self.bits[node][x]
            xp_bits = self.bits[node][xp]
        else:
            count = len(self.bits[x])
            x_bits = self.bits[x]
            xp_bits = self.bits[xp]

        # remember all the disjuncts here
        include = []
        for i in range(count):
            include.append(self.new_var(f"disj_{i}"))
            includeAnd = [xp_bits[i], -x_bits[i]]
            for j in range(i + 1, count):
                impVar = self.new_var(f"{x}<_{node}{xp}V{i}w3W{j}0")
                includeAnd.append(impVar)
                self.clause_writer(impVar, c1 = x_bits[j], c2 = xp_bits[j], connective = 2) # c[0] <-> b_x^j -> b_x'^j
            self._clauses.append([include[-1]] + [-x for x in includeAnd])
            for v in includeAnd:
                self._clauses.append([-include[-1], v])

        self._clauses.append([-self._lessThan[(x,xp)]] + include)                                                 # myId <-> new_var_1 || ... || new_var_n
        for v in include:
            self._clauses.append([self._lessThan[(x,xp)], -v])
        return self._lessThan[(x,xp)]
                         
    def generate_bits(self, local = False):
        # remember which atoms we used for the bits 
        self.bits = {}
        if local:
            for t in self._td.nodes:
                t.atoms = set(map(lambda x: self._vertexToAtom[x], t.vertices))
                self.bits[t] = {}
                tmp_at = t.atoms.copy()
                while not len(tmp_at) == 0:
                    cur = tmp_at.pop()
                    if cur in self._condensation.graph["mapping"]:
                        # otherwise cur does not occur positively
                        comp_id = self._condensation.graph["mapping"][cur]
                        comp = self._condensation.nodes[comp_id]["members"]
                        tmp_at = tmp_at.difference(comp)
                        both = t.atoms.intersection(comp)
                        count = math.ceil(math.log(len(both),2))
                        for a in both:
                            self.bits[t][a] = []
                            #self.bits[t][1][a] = list(range(self._max + 1, self._max + count + 1))
                            #self._max += count
                            for i in range(count):
                                self.bits[t][a].append(self.new_var(f"b_{a}_{t}^{i}"))

        else:
            for comp in self._components:
                cur_max = 0
                # in case of preformance issues comment this out and replace it with the line that is commented out
                #for x in comp:
                #    for y in comp:
                #        cur_max = max(cur_max, len(max(nx.all_simple_paths(self.dep, x, y), key = len, default = [1])))
                #count = math.ceil(math.log(cur_max,2))
                count = math.ceil(math.log(len(comp),2))
                for a in comp:
                    self.bits[a] = [0]*count
                    for i in range(count):
                        self.bits[a][i] = self.new_var(f"b_{a}^{i}")



    def _tdguidedReduction(self, local = False):
        # maps a node t to a set of atoms a for which we require p_t^a or p_{<=t}^a variables for t
        # this is the case if there is a rule suitable for proving a in or below t
        proven_at_atoms = {}
        proven_below_atoms = {}
        # maps a node t to a set of rules that need to be considered in t
        # it actually suffices if every rule is considered only once in the entire td..
        rules = {}
        # temporary copy of the program, will be empty after the first pass
        program = list(self._program)
        self.generate_bits(local)
        # first td pass: determine rules and prove_atoms
        for t in self._td.nodes:
            rules[t] = []
            proven_at_atoms[t] = {}
            # compute t.atoms
            t.atoms = set(map(lambda x: self._vertexToAtom[x], t.vertices))
            # take the rules we need and remove them
            rules[t] = [r for r in program if r.atoms.issubset(t.atoms)]
            program = [r for r in program if not r.atoms.issubset(t.atoms)]
            for r in rules[t]:
                for a in r.head:
                    if a not in proven_at_atoms[t]:
                        proven_at_atoms[t][a] = self.new_var(f"p_{t}^{a}")
                    if a not in proven_below_atoms:
                        proven_below_atoms[a] = set()
                    proven_below_atoms[a].add(proven_at_atoms[t][a])

        #take care of the remaining unary rules
        for r in program:
            if not r.choice: 
                self._clauses.append(list(map(lambda x: self._atomToVertex[abs(x)]*(-1 if x < 0 else 1), r.head + [-x for x in r.body])))

        logger.info("program")
        #logger.info(rules)
        # second td pass: use rules and prove_atoms to generate the reduction
        for t in self._td.nodes:
            # generate (1) the clauses for the rules in the current node
            for r in rules[t]:
                if not r.choice: 
                    self._clauses.append(list(map(lambda x: self._atomToVertex[abs(x)]*(-1 if x < 0 else 1), r.head + [-x for x in r.body])))

            if local:
                # generate (2), i.e. the constraints that maintain the inequalities between nodes
                # use the components here
                for tp in t.children:
                    relevant = tp.atoms.intersection(t.atoms)
                    rel_cp = relevant.copy()
                    while not len(rel_cp) == 0:
                        cur = rel_cp.pop()
                        if cur in self._condensation.graph["mapping"]:
                            comp_id = self._condensation.graph["mapping"][cur]
                            comp = self._condensation.nodes[comp_id]["members"]
                            tmp_at = rel_cp.difference(comp)
                            both = relevant.intersection(comp)
                            for x, xp in product(both, both):
                                if x == xp:
                                    continue
                                self.generateLessThan(x, xp, local = local, node = t)
                                self.generateLessThan(x, xp, local = local, node = tp)
            
            # generate (3), i.e. the constraints that ensure that true atoms that are removed are proven
            for tp in t.children: 
                relevant = tp.atoms.difference(t.atoms)
                for a in relevant:
                    self._clauses.append([-self._atomToVertex[a]] + list(proven_below_atoms.get(a, [])))  # x -> p_{t_1}^x || ... || p_{t_n}^x
            
            # generate (6), i.e. the check for whether an atom was proven at the current node
            for x in proven_at_atoms[t]:
                include = []
                for r in rules[t]:
                    if x in r.head:
                        includeAnd = []
                        include.append(self.new_var(f"{x} proven by {r} at {t}"))                                              # new_var_i
                        for a in r.body:
                            if a > 0:
                                includeAnd.append(self._atomToVertex[a])
                                includeAnd.append(self.generateLessThan(a, x, local = local, node = t))
                            if a < 0:
                                includeAnd.append(-self._atomToVertex[-a])
                        for a in r.head:
                            if a != x:
                                includeAnd.append(-self._atomToVertex[a])
                        self._clauses.append([include[-1]] + [-x for x in includeAnd])
                        for v in includeAnd:
                            self._clauses.append([-include[-1], v])
                self._clauses.append([-proven_at_atoms[t][x]] + include)                                             # c[1] <-> new_var_1 || ... || new_var_n
                for v in include:
                    self._clauses.append([proven_at_atoms[t][x], -v])
            
        # generate (4), i.e. the constraints that ensure that true atoms in the root are proven
        for a in self._td.root.atoms:
            self._clauses.append([-self._atomToVertex[a]] + list(proven_below_atoms.get(a, [])))

        #self.print_prog(rules)


    # function for debugging
    def model_to_names(self):
        f = open("model.out")
        f.readline()
        for i in range(668):
            vs = [int(x) for x in f.readline().split() if abs(int(x)) < 25 and int(x) != 0]
            def getName(v):
                for sym in self.control.symbolic_atoms:
                    if sym.literal == v:
                        return str(sym.symbol)
            #with open("out.cnf", "a") as file_out:
            #    file_out.write(" ".join([str(-v) for v in vs]) + " 0\n")
            #for v in vs:
            #    print(("-" if v < 0 else "")+getName(self._vertexToAtom[abs(v)]))
            
            print(":-" + ", ".join([("not " if v > 0 else "") + getName(self._vertexToAtom[abs(v)]) for v in vs]) + ".")

    def write_dimacs(self, stream):
        stream.write(f"p cnf {self._max} {len(self._clauses)}\n".encode())
        stream.write(("c pv " + " ".join([str(v) for v in self._projected]) + " 0\n" ).encode())
        #f = open("named.cnf", "w")
        for c in self._clauses:
            stream.write((" ".join([str(v) for v in c]) + " 0\n" ).encode())
            #f.write(" ".join([self._nameMap[v] if v > 0 else f"-{self._nameMap[abs(v)]}" for v in c]) + "\n")
        #for (a, w) in self._weights.items():
        #    stream.write(f"w {a} {w}\n".encode())

    def print_prog(self, rules):
        def getName(v):
            for sym in self.control.symbolic_atoms:
                if sym.literal == v:
                    return str(sym.symbol)
        def printrule(r):
            res = ""
            res += ";".join([getName(v) for v in r.head])
            res += ":-"
            res += ",".join([("not " if v < 0 else "") + getName(abs(v)) for v in r.body])
            return res
        for t in self._td.nodes:
            print(t)
            for r in rules[t]:
                print(printrule(r))
                
        
    def my_write(self):
        with open('out.cnf', mode='wb') as file_out:
            file_out.write(f"p cnf {self._max} {len(self._clauses)}\n".encode())
            for c in self._clauses:
                file_out.write((" ".join([str(v) for v in c]) + " 0\n" ).encode())
        with open('white.var', mode='wb') as file_out:
            file_out.write(("\n".join([str(v) for v in self._projected])).encode())
        with open('pv.var', mode='wb') as file_out:
            file_out.write(("pv " + " ".join([str(v) for v in self._projected]) + " 0\n" ).encode())

    def stats(self):
        if len(self._components) > 0:
            largest = max(self._components, key=len)
        else:
            largest = []
        logger.info(f"Largest CC has size {len(largest)}")
        # comment out to see plots of the dependency graph
        #alle = set.union(*self._components)
        #irrelevant = alle.difference(largest)
        #restricted = nx.restricted_view(self.dep, irrelevant, [])
        #nx.draw(self.dep)
        #plt.show()
        #nx.draw(restricted)
        #plt.show()

        local_max = 0
        sum_max = 0
        for t in self._td.nodes:
            local_comp = [set(c).intersection(t.atoms) for c in self._components]
            if len(local_comp) > 0:
                here_max = len(max(local_comp, key=len))
            else:
                here_max = 0
            local_max = max(local_max, here_max)
            sum_max += here_max

        logger.info(f"Largest locally largest CC has size {local_max}")
        logger.info(f"Average locally largest CC has size {sum_max/len(self._td.nodes)}")
        self.encoding_stats()


            

    def encoding_stats(self):
        num_vars, edges= cnf2primal(self._max, self._clauses)
        p = subprocess.Popen([os.path.join(src_path, "htd/bin/htd_main"), "--seed", "12342134", "--input", "hgr"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        logger.debug("Running htd")
        StreamWriter(p.stdin).write_gr(num_vars, edges)
        p.stdin.close()
        tdr = reader.TdReader.from_stream(p.stdout)
        p.wait()
        logger.debug("Parsing tree decomposition")
        td = treedecomp.TreeDecomp(tdr.num_bags, tdr.tree_width, tdr.num_orig_vertices, tdr.root, tdr.bags, tdr.adjacency_list, None)
        logger.info(f"Tree decomposition #bags: {td.num_bags} tree_width: {td.tree_width} #vertices: {td.num_orig_vertices} #leafs: {len(td.leafs)} #edges: {len(td.edges)}")

    def main(self, clingo_control, files):
        """
        Entry point of the application registering the propagator and
        implementing the standard ground and solve functionality.
        """
        if not files:
            files = ["-"]

        self.control = clingoext.Control()

        for path in files:
            self.control.add("base", [], self._read(path))

        self.control.ground()

        logger.info("------------------------------------------------------------")
        logger.info("   Grounded Program")
        logger.info("------------------------------------------------------------")
        #pprint(self.control.ground_program.objects)
        logger.info("------------------------------------------------------------")
        #logger.info(self.control.ground_program)
        logger.info("------------------------------------------------------------")
        
        self._computeComponents();

        self._generatePrimalGraph()
        #logger.info(self._graph.edges())


        self._decomposeGraph()
        #self._tdguidedReduction(local = True)
        self._tdguidedReduction(local = True)
        #parser = wfParse.WeightedFormulaParser()
        #sem = wfParse.WeightedFormulaSemantics(self)
        #wf = "#(1)*(pToS(1)*#(0.3) + npToS(1)*#(0.7))*(pToS(2)*#(0.3) + npToS(2)*#(0.7))*(pToS(3)*#(0.3) + npToS(3)*#(0.7))*(fToI(1,2)*#(0.8215579576173441) + nfToI(1,2)*#(0.17844204238265593))*(fToI(2,1)*#(0.2711032358362575) + nfToI(2,1)*#(0.7288967641637425))*(fToI(2,3)*#(0.6241213691538402) + nfToI(2,3)*#(0.3758786308461598))*(fToI(3,1)*#(0.028975606030084644) + nfToI(3,1)*#(0.9710243939699154))*(fToI(3,2)*#(0.41783665133679737) + nfToI(3,2)*#(0.5821633486632026))"
        #parser.parse(wf, semantics = sem)
        #self.encoding_stats()
        #self.simp()
        #self.my_write()
        # OUTPUT
        #with open('out.cnf', mode='wb') as file_out:
        #    self.write_dimacs(file_out)
        #self.model_to_names()
        self.stats()

if __name__ == "__main__":
    sys.exit(int(clingoext.clingo_main(Application(), sys.argv[1:])))

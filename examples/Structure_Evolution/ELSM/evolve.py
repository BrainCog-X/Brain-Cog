import time
import threading
from threading import Thread
import os
import networkx as nx
import numpy as np
from population import *
import nsganet as engine
from pymop.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import logging
from model import *
from spikes import calc_f2
from multiprocessing import Process,Pool
from datetime import datetime
import time


_logger = logging.getLogger('')
config_parser = parser = argparse.ArgumentParser(description='Evolution Config', add_help=False)

parser = argparse.ArgumentParser(description='SNN Evoving')
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--seed', type=int, default=68, metavar='S')
parser.add_argument('--datapath', default='/data/', type=str, metavar='PATH')
parser.add_argument('--output', default='/data/LSM/Eresult/new', type=str, metavar='PATH')
parser.add_argument('--liquid-size', type=int, default=8000)
parser.add_argument('--pop-size', type=int, default=20)
parser.add_argument('--up', type=int, default=32000000)
parser.add_argument('--low', type=int, default=320000)

parser.add_argument('--n_offspring', type=int, default=200)
parser.add_argument('--n_gens', type=int, default=2000)
parser.add_argument('--arand', type=float, default=285)
parser.add_argument('--brand', type=float, default=1.8)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    args = parser.parse_args(remaining)
    return args

def calc_f1(dirs):
    ci=[]
    G=nx.read_gpickle(dirs)
    largest_component = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_component)
    for u in G.nodes:
        ci.append(nx.clustering(G,u))
    a=sum(ci)
    print("start")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    path=nx.average_shortest_path_length(G)
    print("end")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    return a,path

def mul_f1(pop,steps,rootdir):
    result=[]
    for i in range(0,pop,steps):
        p = Pool(steps)
        dirs=[os.path.join(rootdir,str(i)+'.pkl') for i in range(i,i+steps)]
        ret = p.map(calc_f1,dirs)
        result.extend(ret)
        print(ret)
        p.close()
        p.join()
    return result

class Evolve(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, args,n_var=20, n_obj=1, n_constr=0, lb=None, ub=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int64)
        self.xl = lb
        self.xu = ub
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.args=args


    def _evaluate(self, x, out, *args, **kwargs):
        

        objs = np.full((x.shape[0], self.n_obj), np.nan)
        g1 = np.full((x.shape[0]), np.nan)
        g2 = np.full((x.shape[0]), np.nan)
        gen_dir=os.path.join(self.args.output,'generaion'+str(kwargs['algorithm'].n_gen))
        os.makedirs(gen_dir,exist_ok = True)
        # np.save(os.path.join(gen_dir,"x.npy"),x)
        lsms = x.reshape(x.shape[0],self.args.liquid_size,self.args.liquid_size)
        for i in range(x.shape[0]):
            temp_G = nx.Graph(lsms[i])
            nx.write_gpickle(temp_G, os.path.join(gen_dir,str(i)+".pkl"))
        self.ob1=mul_f1(pop=x.shape[0],steps=10,rootdir=gen_dir)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            _logger.info('Network= {}'.format(arch_id))
            genome = x[i, :]

            g1[i]= genome.sum()-self.args.up
            g2[i]= self.args.low-genome.sum()
            lsmm = genome.reshape(self.args.liquid_size,self.args.liquid_size)
            small_coe_a,small_coe_b=self.ob1[i]
            lsmm=torch.tensor(lsmm,device='cuda:%d' % self.args.device).float()
            crit = calc_f2(lsmm,'cuda:%d' % self.args.device)
            objs[i, 1] = abs(crit-1)
            # all objectives assume to be MINIMIZED !!!!!                
            objs[i, 0] = -(small_coe_a/self.args.arand)/(small_coe_b/self.args.brand)
            

            _logger.info('small word= {}'.format(objs[i, 0]))
            _logger.info('criticality= {}'.format(objs[i, 1]))

            self._n_evaluated += 1

        out["F"] = objs
        out["G"] = np.column_stack([g1,g2])
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")
    
    # report generation info to files
    _logger.info("generation = {}".format(gen))
    _logger.info("population error1: best = {}, mean = {}, "
                 "median1 = {}, worst1 = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    _logger.info('Best1 Genome id= {}'.format(np.argmin(pop_obj[:, 0])))

    _logger.info("population error2: best = {}, mean = {}, "
                 "median2 = {}, worst2 = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))
    _logger.info('Best2 Genome id= {}'.format(np.argmin(pop_obj[:, 1])))
    if gen%20==0:
        best_sid=np.argmin(pop_obj[:, 0])
        best_sname='-'.join([
                'gen'+str(gen),
                's'+str(float('%.4f' % pop_obj[best_sid, 0])),
                'c'+str(float('%.4f' % pop_obj[best_sid, 1])),
            ])
        best_cid=np.argmin(pop_obj[:, 1])
        best_cname='-'.join([
                'gen'+str(gen),
                's'+str(float('%.4f' % pop_obj[best_cid, 0])),
                'c'+str(float('%.4f' % pop_obj[best_cid, 1])),
            ])
        
        np.save(os.path.join('/data/save/genome',best_sname+datetime.now().strftime("%Y%m%d-%H%M%S")),pop_var[np.argmin(pop_obj[:, 0])])
        np.save(os.path.join('/data/save/genome',best_cname+datetime.now().strftime("%Y%m%d-%H%M%S")),pop_var[np.argmin(pop_obj[:, 1])])

if __name__ == '__main__':
    args = _parse_args()
    out_base_dir= os.path.join(args.output, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_base_dir,exist_ok = True)
    args.output=out_base_dir
    setup_default_logging(log_path=os.path.join(out_base_dir, 'log.txt'))

    kkk = Evolve(args,n_var=args.liquid_size*args.liquid_size, 
                  n_obj=2, n_constr=2)
    method = engine.nsganet(pop_size=args.pop_size,
                            sampling=RandomSampling(var_type='custom'),
                            mutation=BinaryBitflipMutation(),
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)
    kres=minimize(kkk,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', args.n_gens))



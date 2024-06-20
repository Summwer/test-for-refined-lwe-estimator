#!/usr/bin/env python
# -*- coding: utf-8 -*-
####
#
#   Copyright (C) 2018-2021 Team G6K
#
#   This file is part of G6K. G6K is free software:
#   you can redistribute it and/or modify it under the terms of the
#   GNU General Public License as published by the Free Software Foundation,
#   either version 2 of the License, or (at your option) any later version.
#
#   G6K is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with G6K. If not, see <http://www.gnu.org/licenses/>.
#
####


"""
LWE Challenge Solving Command Line Client
"""

from __future__ import absolute_import
from __future__ import print_function
import copy
import re
import sys
import time

from collections import OrderedDict # noqa
from math import log

from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

from g6k.algorithms.bkz import pump_n_jump_bkz_tour, dim4free_wrapper,default_dim4free_fun
from g6k.algorithms.pump import pump
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer, dummy_tracer
from g6k.utils.util import load_lwe_challenge

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis_randomly, primal_lattice_basis
from six.moves import range
from g6k.siever_params import SieverParams
from pump_estimation import *
import os
import matplotlib.pyplot as plt
from math import ceil, pi ,log2, floor,sqrt,e
from multiprocessing import Pool


def dim4free_wrapper(dim4free_fun, blocksize):
    """
    Deals with correct dim4free choices for edge cases when non default
    function is chosen.

    :param dim4free_fun: the function for choosing the amount of dim4free
    :param blocksize: the BKZ blocksize

    """
    if blocksize < 40:
        return 0
    dim4free = dim4free_fun(blocksize)
    return int(min((blocksize - 40)/2., dim4free))


def default_dim4free_fun(blocksize):
    """
    Return expected number of dimensions for free, from exact-SVP experiments.

    :param blocksize: the BKZ blocksize

    """
    return int(11.5 + 0.075*blocksize)


def theo_dim4free_fun1(blocksize):
    """
    Theoretical Dimension-for-free function 1 without e in [Duc18]
    """

    return max(0, ceil(blocksize*log(4/3.)/log(blocksize/2./pi)) )


def theo_dim4free_fun2(blocksize):
    """
    Theoretical Dimension-for-free function 2 with e in [Duc18]
    """

    return max(0, ceil(blocksize*log(4/3.)/log(blocksize/2./pi/e)) )



def accs_2023_d4f(slope):
    f = floor(log(sqrt(4/3.))/(-1*slope/4.))
    return f




def get_beta_from_sieve_dim(sieve_dim, d, choose_dims4f_fun):
    f = 0
    for beta  in range(sieve_dim, d):
        if(choose_dims4f_fun == 1 ):
            f = dim4free_wrapper(theo_dim4free_fun1,beta)
        elif(choose_dims4f_fun == 2):
            f = dim4free_wrapper(theo_dim4free_fun2,beta)
        if(beta - f >= sieve_dim):
            return beta
    return d


def get_beta_(beta, jump, d):
    f = dim4free_wrapper(default_dim4free_fun, beta)
    if(jump <= 2):
        return beta
    elif(jump >=3 and jump <=4):
        return get_beta_from_sieve_dim( beta-f,d,2)
    elif(jump>=5):
        get_beta_from_sieve_dim( beta-f,d,1)
    return beta



def preprocess(n,alpha, blocksizes, fpylll_crossover = 50, verbose = False):
    """
    Run the primal attack against Darmstadt LWE instance (n, alpha).

    :param n: the dimension of the LWE-challenge secret
    :param params: parameters for LWE:

        - lwe/alpha: the noise rate of the LWE-challenge

        - lwe/m: the number of samples to use for the primal attack

        - lwe/goal_margin: accept anything that is
          goal_margin * estimate(length of embedded vector)
          as an lwe solution

        - lwe/svp_bkz_time_factor: if > 0, run a larger pump when
          svp_bkz_time_factor * time(BKZ tours so far) is expected
          to be enough time to find a solution

        - bkz/blocksizes: given as low:high:inc perform BKZ reduction
          with blocksizes in range(low, high, inc) (after some light)
          prereduction

        - bkz/tours: the number of tours to do for each blocksize

        - bkz/jump: the number of blocks to jump in a BKZ tour after
          each pump

        - bkz/extra_dim4free: lift to indices extra_dim4free earlier in
          the lattice than the currently sieved block

        - bkz/fpylll_crossover: use enumeration based BKZ from fpylll
          below this blocksize

        - bkz/dim4free_fun: in blocksize x, try f(x) dimensions for free,
          give as 'lambda x: f(x)', e.g. 'lambda x: 11.5 + 0.075*x'

        - pump/down_sieve: sieve after each insert in the pump-down
          phase of the pump

        - dummy_tracer: use a dummy tracer which captures less information

        - verbose: print information throughout the lwe challenge attempt

    """

    A, c, q = load_lwe_challenge(n=n, alpha=alpha)
    # print("-------------------------")
    # print("Primal attack, LWE challenge n=%d, alpha=%.4f" % (n, alpha))

    try:
        min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q,
                                        samples=A.nrows, decouple=True)
        (b, s, m) = min_cost_param
    except TypeError:
        raise TypeError("No winning parameters.")
    # print("Chose %d samples. Predict solution at bkz-%d + svp-%d" % (m, b, s))
    # print()

    target_norm = 1.5 * (alpha*q)**2 * m + 1

    B = primal_lattice_basis_randomly(A, c, q, m=m)
    
    # B = primal_lattice_basis(A, c, q, m=m)
    
    
    params = SieverParams(threads = 4)
    pump_params = {"down_sieve": True}

    g6k = Siever(B, params)
    # print("GSO precision: ", g6k.M.float_type)

    tracer = dummy_tracer
    d = g6k.full_n
    g6k.lll(0, g6k.full_n)
    slope = basis_quality(g6k.M)["/"]
    # print("Intial Slope = %.5f\n" % slope)
    cum_prob = 0.
    T0 = time.time()
    sigma = q*alpha
    for blocksize in blocksizes:
        # BKZ tours
        if blocksize < fpylll_crossover:
            if verbose:
                print("Starting a fpylll BKZ-%d tour. " % (blocksize), end=' ')
                sys.stdout.flush()
            bkz = BKZReduction(g6k.M)
            par = fplll_bkz.Param(blocksize,
                                      strategies=fplll_bkz.DEFAULT_STRATEGY,
                                      max_loops=1)
            bkz(par)

        else:
            if verbose:
                print("Starting a pnjBKZ-%d tour. " % (blocksize))

            pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=1,
                                     verbose=verbose,
                                     extra_dim4free=12,
                                     dim4free_fun="default_dim4free_fun",
                                     goal_r0=target_norm,
                                     pump_params=pump_params)
        

        beta_ = get_beta_(blocksize, 1, d)
        proba = 1. * chdtr(beta_, g6k.M.get_r(d-beta_, d-beta_)/ (sigma**2))
        
        cum_prob += (1-cum_prob) * proba
        if verbose:
            slope = basis_quality(g6k.M)["/"]
            fmt = "slope: %.5f, walltime: %.3f sec"
            print(fmt % (slope, time.time() - T0))

        g6k.lll(0, g6k.full_n)

        if g6k.M.get_r(0, 0) <= target_norm: #and  abs(g6k.M.B[0][-1]) == 1:
            print("Too much preprocess, resample the lwe instance.")
            return None, basis_quality(g6k.M)["/"]
        
    return g6k,target_norm, tracer, d, q, cum_prob, basis_quality(g6k.M)["/"]

def last_pump(g6k, target_norm, tracer, d, dsvp,verbose=False):
    llb = d - dsvp
    
    sieve_dim = pump(g6k, tracer, llb, d-llb, 0, verbose=verbose, down_sieve= True, goal_r0=target_norm * (d - llb)/(1.*d))
    
    f = d - llb - sieve_dim 
    g6k.lll(0, g6k.full_n)
    if g6k.M.get_r(0, 0) <= target_norm: #and  abs(g6k.M.B[0][-1]) == 1:
            # print("Finished! TT=%.2f sec" % (time.time() - T0))
            # print(g6k.M.B[0])
        return True, f

    return False, 0




def draw_figure_for_lwe_d4f_prob(n,alpha,dsvps, dsvp_probs, d4f_probs,actual_dsvps,fss, actual_probs,tours,actual_slopes):
    actual_dsvp = actual_dsvps[-1]
    if actual_dsvp in dsvps:
        ind = dsvps.index(actual_dsvp)
        theo_dsvp_prob = dsvp_probs[ind]
    elif actual_dsvp > dsvps[-1]:
        theo_dsvp_prob = 1.
    else:
        theo_dsvp_prob = 0.
        
        
    #The prob of f should multiply the sucess probability of each dsvp.
    fs = fss[-1]
    
    
    accs_fs = {}
    for i in range(len(actual_slopes)):
        f = accs_2023_d4f(actual_slopes[i])
        if(f not in accs_fs):
            accs_fs[f] = 1
        else:
            accs_fs[f] +=1
            
    if fs != {}:
        sieve_dims = [actual_dsvp-f for f in fs]
        actual_f_probs = [sum([fs[ff] for ff in fs if ff >= f] )/tours for f in fs]
        fig, ax = plt.subplots(figsize=(9, 7),dpi=1000)
        ax.plot(dsvps,[p*theo_dsvp_prob for p in d4f_probs],label="theo prob", color = "red", linestyle="--", zorder = 4)
            # ax.plot(prob_dsvps,theo_probs ,label="theo prob", color = "green", linestyle="--", zorder = 4)
        ax.scatter(sieve_dims[1:],actual_f_probs[1:],label="actual prob", color = "purple",marker="x", zorder = 4)
        
         
        ax.bar([actual_dsvp-theo_dim4free_fun1(actual_dsvp)], [1], label = "dsvp - theo_d4f1(dsvp)", color = "blue")
        
        ax.bar([actual_dsvp-theo_dim4free_fun2(actual_dsvp)], [1], label = "dsvp - theo_d4f2(dsvp)", color = "orange")
        
        
        accs_sieve_dims = [actual_dsvp-f for f in accs_fs]
        accs_f_probs = [sum([accs_fs[ff] for ff in accs_fs if ff >= f] )/tours for f in accs_fs]
        ax.bar(accs_sieve_dims,accs_f_probs,label="accs_d4f", color = "green", zorder = 4)
        

        ax.legend(fontsize=19)
        plt.xlabel(r"sieve_dim = $d_{svp}$ - d4f",fontsize=19)
        plt.xticks(fontsize=19)
        plt.ylabel(r"$P(B\leq {\rm sieve\_dim})$",fontsize=19)
        plt.yticks(fontsize=19)
        plt.savefig("test_dsvps_d4f_prob/[{n:03d}-{alpha:03d}]last-pump-dsvp-d4f-prob-%d.png".format(n=n, alpha=int(alpha*1000)) %actual_dsvp)
        plt.close()


def one_dsvp_tour(parallel_param):
    (n,alpha,blocksizes,dsvp, tours)  = parallel_param 
    
    fs = {dsvp: 0}
    succ_times = 0
    i = 0
    slopes = []
    while(i < tours):
        print("dsvp = %d, %d/%d" %(dsvp,i+1,tours))
        data = [None,None]
        while(data[0] is None and i < tours):
            data = preprocess(n,alpha, blocksizes)
            if(data[0] is None): #it means it succeed in BKZ process
                succ_times += 1
                i += 1
                fs[dsvp] +=1
                slopes.append(data[1])
        if(i == tours):
            break
        g6k,target_norm, tracer, d, q, cum_prob,slope = data
        slopes.append(slope)
        succ, f = last_pump(g6k,target_norm, dummy_tracer, d, dsvp)
        
        if succ:
            succ_times +=1
            if f not in fs:
                fs[f] = 1
            else:
                fs[f] +=1
        i += 1
    return dsvp,succ_times, fs,slopes

def test_dsvp(n,alpha, blocksizes, dsvp, tours = 100): #tours: all test times
    fn = open("test_dsvps_d4f_prob/{n:03d}_{alpha:03d}.log".format(n=n, alpha=int(alpha*1000)), "w")
    fn.write("-------------------------\n")
    fn.write("dsvp probability test on LWE challenge n=%d, alpha=%.4f\n" % (n, alpha))
    fn.write("BKZ blocksizes preprocess: "+str(blocksizes)+"\n")
    print("-------------------------")
    print("dsvp probability test on LWE challenge n=%d, alpha=%.4f" % (n, alpha))
    print("BKZ blocksizes preprocess: ", str(blocksizes))
    
    
    bkz_suc_times = 0
    data = None
    while(data is None and bkz_suc_times < tours):
        data = preprocess(n,alpha, blocksizes)
        if(data is None):
            bkz_suc_times+=1
    fn.write("Success times in BKZ:" + str(bkz_suc_times))
    print("Success times in BKZ:" + str(bkz_suc_times))
    g6k,target_norm, tracer, d, q,cum_prob,slope = data
    
    rr = [g6k.M.get_r(i, i) for i in range(d)]
    # sigma = alpha * q
    # log2_rr = [log2(rr[i])/2  - log(sigma,2) for i in range(d)]
    # print(log2_rr)
    print()
    fn.write("prob estimate: \n")
    _, prob_dsvps, theo_probs = prob_pump_estimation(rr,q, alpha)
    fn.write("prob estimate: \n")
    fn.write("(Actual rr) prob_dsvps:")
    fn.write(str(prob_dsvps)+"\n")
    print(prob_dsvps)
    fn.write("\n")
    fn.write("(Actual rr)Theoretical theo_probs:")
    fn.write(str(theo_probs)+"\n")
    print(theo_probs)
    fn.write("\n")
    
    # _, prob_dsvps, theo_probs = cum_prob_and_prob_pump_estimation(rr,q, alpha,cum_prob = cum_prob)
    # fn.write("cum_prob + prob estimate: \n")
    # fn.write("(Actual rr) cum_prob+prob_dsvps:")
    # fn.write(str(prob_dsvps)+"\n")
    # print(prob_dsvps)
    # fn.write("\n")
    # fn.write("(Actual rr)Theoretical theo_cum_prob_and_probs:")
    # fn.write(str(theo_probs)+"\n")
    # print(theo_probs)
    # fn.write("\n")
    
    
    dsvps,dsvp_probs,d4f_probs = pro_sieve_estimation_20230609(rr,q, alpha, cum_prob = cum_prob)

    fn.write("cum_prob + prob estimate: \n")
    fn.write("(Actual rr) dsvps:")
    fn.write(str(dsvps)+"\n")
    print(dsvps)
    fn.write("\n")
    fn.write("(Actual rr) dsvp_probs:")
    fn.write(str(dsvp_probs)+"\n")
    print(dsvp_probs)
    fn.write("\n")
    fn.write("(Actual rr)Theoretical d4f_probs:")
    fn.write(str(d4f_probs)+"\n")
    print(d4f_probs)
    fn.write("\n")
    
    
    
    # actual_dsvps = deepcopy(prob_dsvps)
    # actual_dsvps.reverse()
    # actual_dsvps = [actual_dsvps[i] for i in range(len(actual_dsvps)) if (i%2==0)]
    # actual_dsvps.reverse()
    actual_probs = []
    prob_dsvps = list(range(30,dsvp+1))
    actual_dsvps = deepcopy(prob_dsvps)
    # actual_dsvps.reverse()
    # actual_dsvps = [actual_dsvps[i] for i in range(len(actual_dsvps)) if (i%4==0)]
    # actual_dsvps.reverse()
    
    if(bkz_suc_times==tours):
        actual_probs = [bkz_suc_times/tours for _ in range(len(actual_dsvps))]
        fn.write("actual_dsvps:")
        fn.write(str(actual_dsvps))
        fn.write("\n")
        fn.write("actual_probs:")
        fn.write(str(actual_probs))
        fn.close()
        return
    
    
    with Pool(len(actual_dsvps)) as p:
        result = p.map(one_dsvp_tour,[(n,alpha,blocksizes,actual_dsvps[-1], tours)])
    # for j in range(len(actual_dsvps)):
        # dsvp = actual_dsvps[j]
        # p = one_dsvp_tour(n,alpha,blocksizes,dsvp, target_norm)
        # print("dsvp =" %(dsvp))
        # fn.write("dsvp = %d, succ_prob = %3.2f\n" %(dsvp,succ_times/tours) )
        # fn.write("fs:")
        # fn.write(str(fs)+"\n")
        # actual_probs.append(succ_times/tours)
    actual_probs = [_[1]/tours for _ in result]
    actual_dsvps = [_[0] for _ in result]
    actual_slopess = [_[-1] for _ in result]
    fss = [_[2] for _ in result]
    fn.write("actual_dsvps:")
    fn.write(str(actual_dsvps))
    fn.write("\n")
    fn.write("actual_probs:")
    fn.write(str(actual_probs))
    fn.write("\n")
    fn.write("fss = ")
    fn.write(str(fss))
    fn.write("\n")
    fn.write("slopess = ")
    fn.write(str(actual_slopess))
    fn.close()
    
    draw_figure_for_lwe_d4f_prob(n,alpha,dsvps,dsvp_probs, d4f_probs,actual_dsvps,fss, actual_probs,tours,actual_slopess[-1])
        
try:
    os.mkdir("test_dsvps_d4f_prob")
except FileExistsError:
    pass

n , alpha, blocksizes, dsvp = 40, 0.005, [10], 75
test_dsvp(n , alpha, blocksizes, dsvp)

n , alpha, blocksizes, dsvp = 55, 0.005, list(range(10,48)), 86
test_dsvp(n , alpha, blocksizes, dsvp)

n , alpha, blocksizes, dsvp = 40, 0.015, list(range(10,50)), 102
test_dsvp(n , alpha, blocksizes, dsvp)

n , alpha, blocksizes, dsvp = 45, 0.010, list(range(10,50)), 94
test_dsvp(n , alpha, blocksizes, dsvp)

n , alpha, blocksizes, dsvp = 60, 0.005, list(range(10,50)), 104
test_dsvp(n , alpha, blocksizes, dsvp)

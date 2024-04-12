# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from scipy.special import chdtr 
from copy import deepcopy
from fpylll.util import gaussian_heuristic

#pump estimation cumulated previous failure probability
# def pump_estimation(fn, rr,q, alpha, cum_prob = 0., succ_prob = 0.999):
#     """
#     Return the (kappa, beta, f) for pump.

#     :param rr: vector of squared Gram-Schmidt norms
#     :param q: q of LWE
#     :param alpha: alpha of LWE
#     :param cum_prob: the cumulated success probability currently.

#     """
#     alpha = float(alpha)
#     sigma = alpha * q
#     d=len(rr)
#     beta0 = 30
#     dsvp_prime = None
#     cum_prob2 = deepcopy(cum_prob)
#     prob_dsvps = []
#     cum_dsvps = []
#     theo_cum_probs = []
#     theo_probs = []
#     for beta in range(beta0, d):
#         GH = gaussian_heuristic(rr[d-beta:])
#         length=(GH/(sigma**2))
#         psvp1 = chdtr(beta, length)
#         psvp2 = chdtr(beta, 4/3.*length)
        
#         prob_dsvps.append(beta)
#         theo_probs.append(psvp1)
#         if(dsvp_prime is None):
#             if(cum_prob2 > succ_prob):
#                 dsvp_prime = beta
#             else:
#                 cum_prob2 = cum_prob2 + (1 - cum_prob2)* psvp2
#         if(cum_prob <= succ_prob):
#             cum_prob = cum_prob + (1 - cum_prob) * psvp1
#             cum_dsvps.append(beta)
#             theo_cum_probs.append(cum_prob)
#         if(psvp1 > succ_prob):
#             break
#         # if(psvp1 > succ_prob):
#         #     break
#     # print(prob_dsvps)
#     # fn.write("cum_prob_dsvps:")
#     # fn.write(str(cum_dsvps))
#     # fn.write("\n")
#     fn.write("(Actual rr)prob_dsvps:")
#     fn.write(str(prob_dsvps))
#     fn.write("\n")
#     # fn.write("Theoretical theo_cum_probs:")
#     # fn.write(str(theo_cum_probs))
#     # fn.write("\n")
#     fn.write("(Actual rr)Theoretical theo_probs:")
#     fn.write(str(theo_probs))
#     fn.write("\n")
#     # print(theo_probs)
#     # llbs = [d-beta for beta in prob_dsvps]
#     # llb = d - beta
#     # f = beta - dsvp_prime 
#     # f = 0
#     return prob_dsvps,cum_dsvps,theo_cum_probs,theo_probs #, theo_cum_probs




# def pump_estimation(fn, rr,q, alpha, cum_prob = 0.):
#     """
#     Return the (kappa, beta, f) for pump.

#     :param rr: vector of squared Gram-Schmidt norms
#     :param q: q of LWE
#     :param alpha: alpha of LWE
#     :param cum_prob: the cumulated success probability currently.

#     """
#     alpha = float(alpha)
#     sigma = alpha * q
#     d=len(rr)
#     beta0 = 30
#     # dsvp_prime = None
#     # cum_prob2 = deepcopy(cum_prob)
#     prob_dsvps = []
#     # cum_dsvps = []
#     # theo_cum_probs = []
#     theo_probs = []
#     for beta in range(beta0, d):
#         GH = gaussian_heuristic(rr[d-beta:])
#         length=(GH/(sigma**2))
#         psvp1 = chdtr(beta, length)
#         # psvp2 = chdtr(beta, 4/3.*length)
        
#         prob_dsvps.append(beta)
#         theo_probs.append(cum_prob + (1-cum_prob)*psvp1)
#         # if(dsvp_prime is None):
#         #     if(cum_prob2 > succ_prob):
#         #         dsvp_prime = beta
#         #     else:
#         #         cum_prob2 = cum_prob2 + (1 - cum_prob2)* psvp2
#         # if(cum_prob <= succ_prob):
#         #     cum_prob = cum_prob + (1 - cum_prob) * psvp1
#         #     cum_dsvps.append(beta)
#         #     theo_cum_probs.append(cum_prob)
#         if(cum_prob + (1-cum_prob)*psvp1 >= 0.999):
#             break
#         # if(psvp1 > succ_prob):
#         #     break
#     # print(prob_dsvps)
#     # fn.write("cum_prob_dsvps:")
#     # fn.write(str(cum_dsvps))
#     # fn.write("\n")
#     # print(cum_prob)
#     fn.write("(Actual rr)prob_dsvps:")
#     fn.write(str(prob_dsvps))
#     print(prob_dsvps)
#     fn.write("\n")
#     fn.write("(Actual rr)Theoretical theo_probs:")
#     fn.write(str(theo_probs))
#     print(theo_probs)
#     fn.write("\n")
#     # print(theo_probs)
#     # llbs = [d-beta for beta in prob_dsvps]
#     # llb = d - beta
#     # f = beta - dsvp_prime 
#     # f = 0
#     # return prob_dsvps,cum_dsvps,theo_cum_probs,theo_probs #, theo_cum_probs



#pump estimation cumulated previous failure probability
def prob_pump_estimation(rr,q, alpha,  succ_prob = 0.999):
    """
    Return the (kappa, beta, f) for pump.

    :param rr: vector of squared Gram-Schmidt norms
    :param q: q of LWE
    :param alpha: alpha of LWE
    :param cum_prob: the cumulated success probability currently.

    """
    alpha = float(alpha)
    sigma = alpha * q
    d=len(rr)
    beta0 = 30
    prob_dsvps = []
    # cum_dsvps = []
    # theo_cum_probs = []
    theo_probs = []
    for beta in range(beta0, d):
        GH = gaussian_heuristic(rr[d-beta:])
        length=(GH/(sigma**2))
        psvp1 = chdtr(beta, length)
        # psvp2 = chdtr(beta, 4/3.*length)s

        # theo_probs.append(psvp1)
        # if(dsvp_prime is None):
        # if(cum_prob2 > succ_prob):
        #     dsvp_prime = beta
        # else:
        # cum_prob2 = cum_prob + (1 - cum_prob)* psvp1
        
        prob_dsvps.append(beta)
        # theo_probs.append(cum_prob2)
        theo_probs.append(psvp1)
        # if(cum_prob <= succ_prob):
            # cum_prob = cum_prob + (1 - cum_prob) * psvp1
            # cum_dsvps.append(beta)
            # theo_cum_probs.append(cum_prob)
        if(psvp1 > succ_prob):
            break
        # if(cum_prob2 > succ_prob):
        #     break

    return beta, prob_dsvps, theo_probs #, theo_cum_probs


#pump estimation cumulated previous failure probability
def cum_prob_and_prob_pump_estimation(rr,q, alpha, cum_prob = 0., succ_prob = 0.999):
    """
    Return the (kappa, beta, f) for pump.

    :param rr: vector of squared Gram-Schmidt norms
    :param q: q of LWE
    :param alpha: alpha of LWE
    :param cum_prob: the cumulated success probability currently.

    """
    alpha = float(alpha)
    sigma = alpha * q
    d=len(rr)
    beta0 = 30
    cum_prob2 = 0.
    prob_dsvps = []
    theo_probs = []
    for beta in range(beta0, d):
        GH = gaussian_heuristic(rr[d-beta:])
        length=(GH/(sigma**2))
        psvp1 = chdtr(beta, length)

        cum_prob2 = cum_prob + (1 - cum_prob)* psvp1
        
        prob_dsvps.append(beta)
        theo_probs.append(cum_prob2)
        
        if(cum_prob2 > succ_prob):
            break

    return beta, prob_dsvps, theo_probs #, theo_cum_probs



#pump estimation cum previous failure probability
def pro_sieve_estimation_20230609(rr,q, alpha, cum_prob = 0., succ_prob = 0.999):
# def pump_estimation2(log_rr,q, alpha, succ_prob = 0.99, ebeta = 50, goal_margin=1.5):
    """
    Return min pump time cost estimate according to progressive sieve following [Duc18]

    :param rr: vector of squared Gram-Schmidt norms
    :param beta: current bkz blocksize
    :param target_norm: target_norm = sigma^2*d following the LWE distribution

    """
    alpha = float(alpha)
    sigma = alpha * q
    d=len(rr)
    pre_psvp2 = 0.
    cum_pump_time  = 0.
    beta = 30
    flag1 = False
    flag2 = False
    
    dsvps = []
    dsvp_probs = []
    d4f_probs = []
    
    prob1 = 0.
    prob2 = 0.
    
    while(beta <= d):
        GH = gaussian_heuristic(rr[d-beta:])
        length=(GH/(sigma**2))
        psvp1 = chdtr(beta, length)
        psvp2 = chdtr(beta, 4/3.*length)
        # cum_pump_time += get_pump_time(beta,d) *(psvp2-pre_psvp2)
        
        prob1 = cum_prob + (1 - cum_prob)* psvp1
        prob2 = cum_prob + (1 - cum_prob)* psvp2
        
        if(prob1 > succ_prob and not flag1):
            dsvp1 = beta
            flag1 = True
        if(prob2 > succ_prob and not flag2):
            dsvp2 = beta
            flag2 = True
        if(flag1 and flag2):
            break
        # pre_psvp2 = psvp2
        
        dsvps.append(beta)
        dsvp_probs.append(prob1)
        d4f_probs.append(prob2)
        
        beta +=1

    # llb = d - dsvp1
    # f = dsvp1 - dsvp2
    # llb = max(0, llb)

    # return llb,f#,GH
    return dsvps,dsvp_probs,d4f_probs
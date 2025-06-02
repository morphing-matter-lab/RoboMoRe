# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:46:42 2015

@author: Mitch
"""

import numpy as np
import numpy.linalg as alg
# import torch.linalg as alg
import numpy as np
import numbers
from numpy.linalg import solve, inv
# from torch.linalg import solve, inv
import torch
import time
# These translate hyperparameters to vector form for minimization

def pack_hyps(xb, hyp_ARD, hyp_coeff, hyp_noise):
    (m,dim) = xb.shape
    xb_row = np.reshape(xb,(1,m*dim))
    return np.concatenate((xb_row,hyp_ARD,hyp_coeff,hyp_noise), axis=1)
    
def unpack_hyps(vec, m, dim):
    vec = np.reshape(vec, (1,(m+1)*dim + 2))
    xb = np.reshape(vec[0][:m*dim],(m,dim))
    hyp_ARD = np.reshape(vec[0][m*dim:-2],(1,dim))
    hyp_coeff = vec[0][-2]
    hyp_noise = vec[0][-1]
    
    return (xb, hyp_ARD, hyp_coeff, hyp_noise)
    
#################################################
    
# Computes kernel on input points
    
def RBF_kernel(x1, x2, hyp_ARD, hyp_coeff, is_self=False):    
    (n1, dim) = x1.shape
    n2 = x2.shape[0]
    
    b = np.exp(hyp_ARD)
    coeff = np.exp(hyp_coeff)
    
    # use ARD to scale
    b_sqrt = np.sqrt(b)
    x1 = x1 * b_sqrt
    x2 = x2 * b_sqrt
    
    try:
        x1_sum_sq = np.reshape(np.sum(x1 * x1, axis=1), (n1,1))
        x2_sum_sq = np.reshape(np.sum(x2 * x2, axis=1), (1,n2))
    except:
        raise
    K = -2 * np.dot(x1, x2.transpose())
    K = K + x1_sum_sq + x2_sum_sq
    K = coeff * np.exp(-.5 * K)
    
    if(is_self):
        jitter = 1e-6
        K = K + jitter * np.eye(n1)
    
    return K
    
#########################################
    
def r2(y, ytrue):
    n = y.shape[0]
    y = np.reshape(y,(n,1))
    ytrue = np.reshape(ytrue,(n,1))
    return 1 - np.sum((y - ytrue)**2) / np.sum((ytrue - ytrue.mean())**2)
    
#############################################
    
# Inverts based on cholesky factorization inverse
    
def chol_invert(factor):
    inv_fact = alg.inv(factor)
    return np.dot(inv_fact.transpose(), inv_fact)
    
#############################################
    
# finds index of closest point in list to input point
    
def closestPoint(x, X):
    x = np.array(x)
    points = np.array(X)
    deltas = points - x
    dist2 = np.sum(deltas * deltas, axis=1)
    return np.argmin(dist2)
    
#############################################
    
# computes pairwise distance between vector elements
    
def pair_dist(x1, x2):
    n1 = max(x1.shape)
    n2 = max(x2.shape)
    return np.reshape(np.subtract.outer(x1,x2),(n1,n2))
    
##############################################
    
# Computes negative log likelihood and derivatives for minimization
    
def SPGP_likelihood(params, y, x, m, compute_deriv=True):
    (N, dim) = x.shape
    y = np.reshape(np.array(y),(N,1))
    (xb, hyp_ARD, hyp_coeff, hyp_noise) = unpack_hyps(np.array([params]), m, dim)

    jitter = 1e-6
    b = np.exp(hyp_ARD)[0]
    b_sqrt = np.sqrt(b)
    coeff = np.exp(hyp_coeff)
    sigma = np.exp(hyp_noise)
    
    # compute Q matrix, covariance between pseudo-inputs
    Q = RBF_kernel(xb, xb, hyp_ARD, hyp_coeff, is_self=True)
    
    # compute K matrix, cov between pseudo-inputs and data
    K = RBF_kernel(xb, x, hyp_ARD, hyp_coeff)
    
    L = alg.cholesky(Q)
    
    V = alg.solve(L, K)
    ep = 1 + np.reshape(coeff - np.sum(V * V, axis=0), (1,N)) / sigma
    ep_sqrt = np.sqrt(ep)
    K = K / ep_sqrt
    V = V / ep_sqrt
    y = y / ep_sqrt.transpose()
    
    Lm = alg.cholesky(sigma * np.eye(m) + np.dot(V, V.transpose()))
        
    invLmV = alg.solve(Lm, V)
    bet = np.dot(invLmV, y)
    
    # compute negative log likelihood
    partial = np.log(Lm.diagonal()).sum()
    partial += (N - m) * hyp_noise / 2
    partial += float(np.dot(y.transpose(), y) - np.dot(bet.transpose(), bet)) / (2 * sigma)
    partial += np.log(ep).sum() / 2
    partial += N * np.log(2 * np.pi) / 2
    
    lik = partial
    
    if(not compute_deriv):
        return lik
    
    # now compute derivates for minimization
    
    # ugly precomputations
    Lt = np.dot(L, Lm)
    B1 = alg.solve(Lt.transpose(), invLmV)
    b1 = alg.solve(Lt.transpose(), bet)
    invLV = alg.solve(L.transpose(), V)
    invQ = chol_invert(L)
    invA = chol_invert(Lt)
    mu = np.dot(alg.solve(Lm.transpose(),bet).transpose(), V).transpose()
    sumVsq = np.reshape(np.sum(V * V, axis=0), (N,1))
    bigsum = 0.5 + y * np.dot(bet.transpose(), invLmV).transpose() / sigma
    bigsum -= np.reshape(np.sum(invLmV * invLmV, axis=0),(N,1)) / 2
    bigsum -= (y*y + mu*mu) / (2 * sigma)
    TT = np.dot(invLV, invLV.transpose() * bigsum)
    
    xb_deriv = np.zeros((m,dim))
    ARD_deriv = np.zeros((1,dim))
    for i in range(dim):
        dnnQ = pair_dist(xb[:,i], xb[:,i]) * Q
        dNnK = pair_dist(-xb[:,i],-x[:,i]) * K
        
        epdot = -2 * dNnK * invLV / sigma
        epPmod = -np.reshape(np.sum(epdot, axis=0), (N,1))
        
        dxb = -b1 * (np.dot(dNnK, (y - mu)) / sigma + np.dot(dnnQ, b1))
        dxb += np.reshape(np.sum((invQ - sigma * invA) * dnnQ,axis=1),(m,1))
        dxb += np.dot(epdot, bigsum)
        dxb -= 2 * np.reshape(np.sum(dnnQ * TT, axis=1),(m,1)) / sigma
        
        dARD = ((y - mu).transpose() * np.dot(b1.transpose(),dNnK)) / sigma
        dARD = dARD + (epPmod * bigsum).transpose()
        dARD = np.dot(dARD, np.reshape(x[:,i],(N,1)))
        
        dNnK = dNnK * B1
        dxb = dxb + np.reshape(np.sum(dNnK,axis=1),(m,1))
        dARD = dARD - np.dot(np.reshape(np.sum(dNnK,axis=0),(1,N)), np.reshape(x[:,i],(N,1)))
        
        dxb = b_sqrt[i] * dxb
        
        dARD = dARD / b_sqrt[i]
        dARD = dARD + np.dot(dxb.transpose(),np.reshape(xb[:,i],(m,1))) / b[i]
        dARD = b_sqrt[i] * dARD / 2
        
        xb_deriv[:,i] = dxb[:,0]
        ARD_deriv[0][i] = dARD
        
    ep = ep.transpose()
    epc = (coeff / ep - sumVsq - jitter * np.reshape(np.sum(invLV * invLV,axis=0),(N,1))) / sigma
    
    coeff_deriv = (m + jitter*np.trace(invQ - sigma * invA))
    coeff_deriv -= sigma * np.sum(invA * Q.transpose()) / 2
    coeff_deriv -= np.dot(mu.transpose(),y - mu) / sigma
    coeff_deriv += np.dot(np.dot(b1.transpose(), Q - jitter * np.eye(m)),b1) / 2
    coeff_deriv += np.dot(epc.transpose(), bigsum)
    
    noise_deriv = np.reshape(np.sum(bigsum / ep),(1,1))
    
    deriv = pack_hyps(xb_deriv, ARD_deriv, coeff_deriv, noise_deriv)
       
    return lik, deriv[0] / alg.norm(deriv)



class OGP(object):
    def __init__(self, dim, noise, covar, maxBV=200, 
                 prmean=None, prmeanp=None, proj=True, weighted=False, thresh=1e-6, 
                 dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.nin = dim
        self.maxBV = maxBV
        self.numBV = 0
        self.proj = proj
        self.weighted = weighted
        self.covar=covar
        self.dtype = dtype
        self.device = device
        # if(covar in ['RBF_ARD']):
        #     self.covar = covar
        #     self.covar_params = hyperparams[:2]
        # else:
        #     print('Unknown covariance function')
        #     raise
            
        self.noise_var = noise
        
        self.prmean = prmean
        self.prmeanp = prmeanp
        
        # initialize model state
        self.BV = np.zeros(shape=(0,self.nin))
        self.alpha = np.zeros(shape=(0,1))
        self.C = np.zeros(shape=(0,0))
        
        self.KB = np.zeros(shape=(0,0))
        self.KBinv = np.zeros(shape=(0,0))
        
        self.thresh = thresh
        
    def fit(self, X, Y, m=0):
        # just train on all the data in X. m is a dummy parameter
        for i in range(X.shape[0]):
            print(i, end='\r')
            self.update(np.array(X[i,:],ndmin=2),Y[i])
        print('')
        
    def update(self, x_new, y_new):
        # compute covariance with BVs
        k_x = self.computeCov(self.BV, x_new)
        k = self.computeCov(x_new, x_new)
        # print(k_x, k)
        # raise NotImplementedError
        # compute mean and variance
        
        cM = np.dot(np.transpose(k_x),self.alpha)
        cV = k + np.dot(np.transpose(k_x),np.dot(self.C,k_x))
        # not needed if nout==1: cV = (cV + np.transpose(cV)) / 2
        
        cV = max(cV, 1e-12)
        
        pM = self.priorMean(x_new)
        # print(pM)
        (logLik, K1, K2) = logLikelihood(self.noise_var, y_new, cM+pM, cV)
        
        # compute gamma, a geometric measure of novelty
        # t_start = time.time()
        hatE = solve(self.KB, k_x)
        # print('use_used', time.time()-t_start)
        # hatE  = torch.linalg.solve(
        #     torch.from_numpy(self.KB).to(dtype=self.dtype, device=self.device), 
        #      torch.from_numpy(k_x).to(dtype=self.dtype, device=self.device)
        #     ).detach().cpu().numpy()
        gamma = k - np.dot(np.transpose(k_x),hatE)
        
        if(gamma < self.thresh*k):
            # not very novel, just tweak parameters
            self._sparseParamUpdate(k_x, K1, K2, gamma, hatE)
        else:
            # expand model
            self._fullParamUpdate(x_new, k_x, k, K1, K2, gamma, hatE)
        
        # reduce model according to maxBV constraint
        while(self.BV.shape[0] > self.maxBV):
            minBVind = self.scoreBVs()
            self.deleteBV(minBVind)
            
    def predict(self, x_in):
        # reads in a (n x dim) vector and returns the (n x 1) vector 
        #   of predictions along with predictive variance for each
    
        k_x = self.computeCov(x_in, self.BV)
        k = self.computeCov(x_in, x_in)
        pred = np.dot(k_x, self.alpha)
        var = k + np.dot(k_x,np.dot(self.C,k_x.transpose()))
        
        pmean = self.priorMean(x_in)
        return pmean + pred, var
            
    def _sparseParamUpdate(self, k_x, K1, K2, gamma, hatE):
        # computes a sparse update to the model without expanding parameters
    
        eta = 1
        if(self.proj):
            eta += K2 * gamma
        CplusQk = np.dot(self.C, k_x) + hatE
        self.alpha = self.alpha + (K1 / eta) * CplusQk
        eta = K2 / eta
        self.C = self.C + eta * np.dot(CplusQk,CplusQk.transpose())
        self.C = stabilizeMatrix(self.C)
        
    def _fullParamUpdate(self, x_new, k_x, k, K1, K2, gamma, hatE):
        # expands parameters to incorporate new input
        
        # add new input to basis vectors
        oldnumBV = self.BV.shape[0]
        numBV = oldnumBV + 1
        self.BV = np.concatenate((self.BV,x_new), axis=0)
        # print(hatE)
        # raise NotImplementedError
        hatE = extendVector(hatE, val=-1)
                # update KBinv
        
        self.KBinv = extendMatrix(self.KBinv)
        self.KBinv = self.KBinv + (1 / gamma) * np.dot(hatE,hatE.transpose())
        
        # update Gram matrix
        self.KB = extendMatrix(self.KB)
        if(numBV > 1):
            self.KB[0:oldnumBV,[oldnumBV]] = k_x
            self.KB[[oldnumBV],0:oldnumBV] = k_x.transpose()
        self.KB[oldnumBV,oldnumBV] = k
        
        Ck = extendVector(np.dot(self.C, k_x), val=1)
        
        self.alpha = extendVector(self.alpha)
        self.C = extendMatrix(self.C)
        
        self.alpha = self.alpha + K1 * Ck
        self.C = self.C + K2 * np.dot(Ck, Ck.transpose())
        
        # stabilize matrices for conditioning/reducing floating point errors?
        self.C = stabilizeMatrix(self.C)
        self.KB = stabilizeMatrix(self.KB)
        self.KBinv = stabilizeMatrix(self.KBinv)
        
    def scoreBVs(self):
        # measures the importance of each BV for model accuracy  
        # currently quite slow for the weighted GP if numBV is much more than 50
        
        numBV = self.BV.shape[0]
        a = self.alpha
        if(not self.weighted):
            scores = ((a * a).reshape((numBV)) / 
                (self.C.diagonal() + self.KBinv.diagonal()))
        else:
            scores = np.zeros(shape=(numBV,1))
            
            # This is slow, in particular the numBV calls to computeWeightedDiv
            for removed in range(numBV):
                (hatalpha, hatC) = self.getUpdatedParams(removed)
                
                scores[removed] = self.computeWeightedDiv(hatalpha, hatC, removed)
                            
        return scores.argmin()
        
    def priorMean(self, x):
        if(callable(self.prmean)):
            if(self.prmeanp is not None):
                return self.prmean(x, self.prmeanp)
            else:
                return self.prmean(x)
        elif(isinstance(self.prmean,numbers.Number)):
            return self.prmean
        else:
            # if no prior mean function is supplied, assume zero
            return 0
    
    def deleteBV(self, removeInd):
        # removes a BV from the model and modifies parameters to 
        #   attempt to minimize the removal's impact        
        
        numBV = self.BV.shape[0]
        keepInd = [i for i in range(numBV) if i != removeInd]
        
        # update alpha and C
        (self.alpha, self.C) = self.getUpdatedParams(removeInd)
            
        # stabilize C
        self.C = stabilizeMatrix(self.C)
        
        # update KB and KBinv
        q_star = self.KBinv[removeInd,removeInd]
        red_q = self.KBinv[keepInd][:,[removeInd]]
        self.KBinv = (self.KBinv[keepInd][:,keepInd] - 
            (1 / q_star) * np.dot(red_q, red_q.transpose()))
        self.KBinv = stabilizeMatrix(self.KBinv)
        
        self.KB = self.KB[keepInd][:,keepInd]
        self.BV = self.BV[keepInd]
        
    def computeWeightedDiv(self, hatalpha, hatC, removeInd):
        # computes the weighted divergence for removing a specific BV
        # currently uses matrix inversion and therefore somewhat slow
        
        hatalpha = extendVector(hatalpha, ind=removeInd)
        hatC = extendMatrix(hatC, ind=removeInd)        
        
        diff = self.alpha - hatalpha
        scale = np.dot(self.alpha.transpose(), np.dot(self.KB,self.alpha))
        
        Gamma = np.eye(self.BV.shape[0]) + np.dot(self.KB,self.C)
        Gamma = Gamma.transpose() / scale + np.eye(self.BV.shape[0])
        M = 2 * np.dot(Gamma,self.alpha) - (self.alpha + hatalpha)
        
        hatV = inv(hatC + self.KBinv)
        (s,logdet) = np.linalg.slogdet(np.dot(self.C + self.KBinv, hatV))
        
        if(s==1):
            w = np.trace(np.dot(self.C - hatC, hatV)) - logdet
        else:
            w = np.Inf
        
        return np.dot(M.transpose(), np.dot(hatV, diff)) + w
        
    def getUpdatedParams(self, removeInd):
        # computes updates for alpha and C after removing the given BV        
        
        numBV = self.BV.shape[0]
        keepInd = [i for i in range(numBV) if i != removeInd]
        a = self.alpha
        
        if(not self.weighted):
            # compute auxiliary variables
            q_star = self.KBinv[removeInd,removeInd]
            red_q = self.KBinv[keepInd][:,[removeInd]]
            c_star = self.C[removeInd,removeInd]
            red_CQsum = red_q + self.C[keepInd][:,[removeInd]]
        
            if(self.proj):
                hatalpha = (a[keepInd] - 
                    (a[removeInd] / (q_star + c_star)) * red_CQsum)
                hatC = (self.C[keepInd][:,keepInd] + 
                    (1 / q_star) * np.dot(red_q,red_q.transpose()) -
                    (1 / (q_star + c_star)) * np.dot(red_CQsum,red_CQsum.transpose()))
            else:
                tempQ = red_q / q_star
                hatalpha = a[keepInd] - a[removeInd] * tempQ
                red_c = self.C[removeInd,[keepInd]]
                hatC = (self.C[keepInd][:,keepInd] + 
                        c_star * np.dot(tempQ,tempQ.transpose()))
                tempQ = np.dot(tempQ, red_c)
                hatC = hatC - tempQ - tempQ.transpose()
        else:
            # compute auxiliary variables
            q_star = self.KBinv[removeInd,removeInd]
            red_q = self.KBinv[keepInd][:,[removeInd]]
            c_star = self.C[removeInd,removeInd]
            red_CQsum = red_q + self.C[keepInd][:,[removeInd]]
            Gamma = (np.eye(numBV) + np.dot(self.KB, self.C)).transpose()
            Gamma = (np.eye(numBV) + 
                Gamma / np.dot(a.transpose(), np.dot(self.KB, a)))
                
            hatalpha = (np.dot(Gamma[keepInd], a) - 
                np.dot(Gamma[removeInd], a) * red_q / q_star)
                    
            # this isn't rigorous...
            #extend = extendVector(hatalpha, ind=removeInd)
            hatC = self.C# + np.dot(2*np.dot(Gamma,a) - (a + extend),
                                       #(a - extend).transpose())
            hatC = (hatC[keepInd][:,keepInd] + 
                    (1 / q_star) * np.dot(red_q,red_q.transpose()) -
                    (1 / (q_star + c_star)) * np.dot(red_CQsum,red_CQsum.transpose()))
            
        return hatalpha, hatC
        
    def computeCov(self, x1, x2, is_self=False):
        # computes covariance between inputs x1 and x2
        #   returns a matrix of size (n1 x n2)
        # directly use covariance function from higher level gp
        if len(x1) == 0 or len(x2) == 0:
            return np.zeros((0, 1))
        else:
            x1_torch = torch.from_numpy(x1).to(dtype=self.dtype, device=self.device)
            x2_torch = torch.from_numpy(x2).to(dtype=self.dtype, device=self.device)
            K = self.covar(x1_torch, x2_torch).to_dense().detach().cpu().numpy()
            return K
        
        
        # (n1, dim) = x1.shape
        # n2 = x2.shape[0]
    
        # (hyp_ARD, hyp_coeff) = self.covar_params
    
        # b = np.exp(hyp_ARD)
        # coeff = np.exp(hyp_coeff)
    
        # # use ARD to scale
        # b_sqrt = np.sqrt(b)
        # x1 = x1 * b_sqrt
        # x2 = x2 * b_sqrt
    

        # x1_sum_sq = np.reshape(np.sum(x1 * x1, axis=1), (n1,1))
        # x2_sum_sq = np.reshape(np.sum(x2 * x2, axis=1), (1,n2))

        # K = -2 * np.dot(x1, x2.transpose())
        # K = K + x1_sum_sq + x2_sum_sq
        # K = coeff * np.exp(-.5 * K)
    
        # if(is_self):
        #     jitter = 1e-6
        #     K = K + jitter * np.eye(n1)
        # print(K.shape)
        # return K
        
def logLikelihood(noise, y, mu, var):
    sigX2 = noise + var
    K2 = -1 / sigX2
    K1 = -K2 * (y - mu)
    logLik = - (np.log(2*np.pi*sigX2) + (y - mu) * K1) / 2
    
    return logLik, K1, K2
    
def stabilizeMatrix(M):
    return (M + M.transpose()) / 2
    
def extendMatrix(M, ind=-1):
    if(ind==-1):
        M = np.concatenate((M,np.zeros(shape=(M.shape[0],1))),axis=1)
        M = np.concatenate((M,np.zeros(shape=(1,M.shape[1]))),axis=0)
    elif(ind==0):
        M = np.concatenate((np.zeros(shape=(M.shape[0],1)),M),axis=1)
        M = np.concatenate((np.zeros(shape=(1,M.shape[1])),M),axis=0)
    else:
        M = np.concatenate((M[:ind], np.zeros(shape=(1,M.shape[1])), M[ind:]),axis=0)
        M = np.concatenate((M[:,:ind], np.zeros(shape=(M.shape[0],1)), M[:,ind:]),axis=1)
    return M
    
def extendVector(v, val=0, ind=-1):
    if(ind==-1):
        if len(v) == 0:
            return np.array([[val]])
        return np.concatenate((v,[[val]]),axis=0)
    elif(ind==0):
        return np.concatenate(([[val]],v),axis=0)
    else:
        return np.concatenate((v[:ind],[[val]],v[ind:]),axis=0)
from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, dsigma_DVCS_HERA, M
import numpy as np
import pandas as pd


def PDF_theo(PDF_input: pd.DataFrame, Para: np.array):
    # [x, t, Q, f, delta_f, spe, flv] = PDF_input
    xs = PDF_input['x'].to_numpy()
    ts = PDF_input['t'].to_numpy()
    Qs = PDF_input['Q'].to_numpy()
    flvs = PDF_input['flv'].to_numpy()
    spes = PDF_input['spe'].to_numpy()
 
    xi = 0

    ps = np.where(spes<=1, 1, -1)
    spes = np.where(spes<=1, spes, spes-2)

    '''
    if(spe == 0 or spe == 1):
        spe, p = spe, 1

    if(spe == 2 or spe == 3):
        spe, p = spe - 2 , -1
    '''
    # Para: (4, 2, 5, 1, 4)

    Para_spe = Para[spes] # fancy indexing. Output (N, 3, 5, 1, 5)
    PDF_theo = GPDobserv(xs, xi, ts, Qs, ps)
    return PDF_theo.tPDF(flvs, Para_spe)  # array length N

tPDF_theo = PDF_theo
      

def GFF_theo(GFF_input: np.array, Para):
    # [j, t, Q, f, delta_f, spe, flv] = GFF_input
    js = GFF_input['j'].to_numpy()
    ts = GFF_input['t'].to_numpy()
    Qs = GFF_input['Q'].to_numpy()
    #fs = GFF_input['f'].to_numpy()
    #delta_fs = GFF_input['delta f'].to_numpy()
    flvs = GFF_input['flv'].to_numpy()
    spes = GFF_input['spe'].to_numpy()
    x = 0
    xi = 0   
    '''
    if(spe == 0 or spe == 1):
        spe, p = spe, 1

    if(spe == 2 or spe == 3):
        spe, p = spe - 2 , -1
    '''
    ps = np.where(spes<=1, 1, -1)
    spes = np.where(spes<=1, spes, spes-2)

    Para_spe = Para[spes] # fancy indexing. Output (N, 3, 5, 1, 5)
    GFF_theo = GPDobserv(x, xi, ts, Qs, ps)
    return GFF_theo.GFFj0(js, flvs, Para_spe) # (N)

def CFF_theo(xB, t, Q, Para_Unp, Para_Pol):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/(Q**2*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    Ht_Et = GPDobserv(x, xi, t, Q, -1)
    HCFF = H_E.CFF(Para_Unp[..., 0, :, :, :, :])
    ECFF = H_E.CFF(Para_Unp[..., 1, :, :, :, :])
    HtCFF = Ht_Et.CFF(Para_Pol[..., 0, :, :, :, :])
    EtCFF = Ht_Et.CFF(Para_Pol[..., 1, :, :, :, :])

    return [ HCFF, ECFF, HtCFF, EtCFF ] # this can be a list of arrays of shape (N)
    # return np.stack([HCFF, ECFF, HtCFF, EtCFF], axis=-1)

def DVCSxsec_theo(DVCSxsec_input: pd.DataFrame, CFF_input: np.array):
    # CFF_input is a list of np.arrays
    # [y, xB, t, Q, phi, f, delta_f, pol] = DVCSxsec_input    

    y = DVCSxsec_input['y'].to_numpy()
    xB = DVCSxsec_input['xB'].to_numpy()
    t = DVCSxsec_input['t'].to_numpy()
    Q = DVCSxsec_input['Q'].to_numpy()
    phi = DVCSxsec_input['phi'].to_numpy()
    #f = DVCSxsec_input['f'].to_numpy()
    pol = DVCSxsec_input['pol'].to_numpy()

    [HCFF, ECFF, HtCFF, EtCFF] = CFF_input # each of them have shape (N); scalar is also OK if we use 
    return dsigma_TOT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

def DVCSxsec_cost_xBtQ(DVCSxsec_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol):
    [xB, t, Q] = [DVCSxsec_data_xBtQ['xB'].iat[0], DVCSxsec_data_xBtQ['t'].iat[0], DVCSxsec_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol) # scalar for each of them
    # DVCS_pred_xBtQ = np.array(list(map(partial(DVCSxsec_theo, CFF_input = [HCFF, ECFF, HtCFF, EtCFF]), np.array(DVCSxsec_data_xBtQ))))
    DVCS_pred_xBtQ = DVCSxsec_theo(DVCSxsec_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    return np.sum(((DVCS_pred_xBtQ - DVCSxsec_data_xBtQ['f'])/ DVCSxsec_data_xBtQ['delta f']) ** 2 )

def DVCSxsec_HERA_theo(DVCSxsec_data_HERA: pd.DataFrame, Para_Unp, Para_Pol):
    # [y, xB, t, Q, f, delta_f, pol]  = DVCSxsec_data_HERA
    y = DVCSxsec_data_HERA['y'].to_numpy()
    xB = DVCSxsec_data_HERA['xB'].to_numpy()
    t = DVCSxsec_data_HERA['t'].to_numpy()
    Q = DVCSxsec_data_HERA['Q'].to_numpy()
    #f = DVCSxsec_data_HERA['f'].to_numpy()
    #delta_f = DVCSxsec_data_HERA['delta f'].to_numpy()
    pol = DVCSxsec_data_HERA['pol'].to_numpy()

    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, np.expand_dims(Para_Unp, axis=0), np.expand_dims(Para_Pol, axis=0))
    return dsigma_DVCS_HERA(y, xB, t, Q, pol, HCFF, ECFF, HtCFF, EtCFF)
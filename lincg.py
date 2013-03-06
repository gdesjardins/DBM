import copy
import theano
from theano import tensor
from theano.ifelse import ifelse
from theano.sandbox.scan import scan
from theano.printing import Print

def linear_cg_fletcher_reeves(compute_Gv, bs, xinit = None,
              rtol = 1e-6, maxit = 1000, damp=0,
              floatX = None, profile=0):
    """
    assume all are lists all the time
    Reference:
        http://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    n_params = len(bs)


    def loop(rz_old, *args):
        ps = args[:n_params]
        rs = args[n_params:2*n_params]
        xs = args[2*n_params:]
        _Aps = compute_Gv(*ps)
        Aps = [x + damp*y for x,y in zip(_Aps, ps)]
        alpha = rz_old/sum( (x*y).sum() for x,y in zip(Aps, ps))
        xs = [x + alpha * p for x,p in zip(xs,ps)]
        rs = [r - alpha * Ap for r, Ap, in zip(rs, Aps)]
        rz_new = sum( (r*r).sum() for r in rs)
        ps = [ r + rz_new/rz_old*p for r,p in zip(rs,ps)]
        return [rz_new]+ps+rs+xs, \
                theano.scan_module.until(abs(rz_new) < rtol)

    if xinit is None:
        r0s = bs
        _x0s = [tensor.unbroadcast(tensor.shape_padleft(tensor.zeros_like(x))) for x in bs]
    else:
        init_Gv = compute_Gv(*xinit)
        r0s = [bs[i] - init_Gv[i] for i in xrange(len(bs))]
        _x0s = [tensor.unbroadcast(tensor.shape_padleft(xi)) for xi in xinit]

    _p0s = [tensor.unbroadcast(tensor.shape_padleft(x),0) for x in r0s]
    _r0s = [tensor.unbroadcast(tensor.shape_padleft(x),0) for x in r0s]
    rz_old = sum( (r*r).sum() for r in r0s)
    _rz_old = tensor.unbroadcast(tensor.shape_padleft(rz_old),0)
    outs, updates = scan(loop,
                         states = [_rz_old] + _p0s + _r0s + _x0s,
                         n_steps = maxit,
                         mode = theano.Mode(linker='cvm'),
                         name = 'linear_conjugate_gradient',
                         profile=profile)
    fxs = outs[1+2*n_params:]
    return [x[0] for x in fxs]


def linear_cg_polyak_ribiere(compute_Gv, b, M=None, xinit = None,
                      rtol = 1e-16, maxit = 100000, floatX = None):
    """
    assume all are lists all the time
    Reference:
        http://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    n_params = len(b)
    def loop(*args):
        pk = args[:n_params]
        rk = args[n_params:2*n_params]
        zk = args[2*n_params:3*n_params]
        xk = args[-n_params:]
        A_pk = compute_Gv(*pk)
        alphak_num = sum((rk_ * zk_).sum() for rk_, zk_ in zip(rk,zk))
        alphak_denum = sum((A_pk_ * pk_).sum() for A_pk_, pk_ in zip(A_pk, pk))
        alphak = alphak_num / alphak_denum
        xkp1 = [xk_ + alphak * pk_ for xk_, pk_ in zip(xk, pk)]
        rkp1 = [rk_ - alphak * A_pk_ for rk_, A_pk_, in zip(rk, A_pk)]
        if M:
            zkp1 = [rkp1_ / m_ for rkp1_, m_ in zip(rkp1, M)]
        else:
            zkp1 = rkp1
        # compute beta_k using Polak-Ribiere
        betak_num = sum((zkp1_* (rkp1_ - rk_)).sum() for rkp1_,rk_,zkp1_ in zip(rkp1,rk,zkp1))
        betak_denum = alphak_num
        betak = betak_num / betak_denum
        pkp1 = [zkp1_ + betak * pk_ for zkp1_, pk_ in zip(zkp1,pk)]
        # compute termination critera
        rkp1_norm = sum((rkp1_**2).sum() for rkp1_ in rkp1)
        return pkp1 + rkp1 + zkp1 + xkp1,\
               theano.scan_module.until(abs(rkp1_norm) < rtol)

    if xinit is None:
        r0_temp = b
        x0 = [tensor.unbroadcast(tensor.shape_padleft(tensor.zeros_like(b_))) for b_ in b]
    else:
        init_Gv = compute_Gv(*xinit)
        r0_temp = [b[i] - init_Gv[i] for i in xrange(len(b))]
        x0 = [tensor.unbroadcast(tensor.shape_padleft(xinit_)) for xinit_ in xinit]

    r0 = [tensor.unbroadcast(tensor.shape_padleft(r0_temp_)) for r0_temp_ in r0_temp]
    if M:
        z0 = [tensor.unbroadcast(tensor.shape_padleft(r0_temp_ / m_)) for r0_temp_, m_ in zip(r0_temp, M)]
    else:
        z0 = r0
    p0 = z0

    outs, updates = scan(loop,
                         states = p0 + r0 + z0 + x0,
                         n_steps = maxit,
                         mode = theano.Mode(linker='c|py'),
                         name = 'linear_conjugate_gradient',
                         profile=0)
    fxs = outs[-n_params:]
    return [x[0] for x in fxs]

linear_cg = linear_cg_polyak_ribiere


import numpy
import time
import theano
import theano.tensor as T
from scipy import linalg
floatX = theano.config.floatX

from DBM import lincg

rng = numpy.random.RandomState(23091)
nparams = 1000

def init_psd_mat(size):
    temp = rng.rand(size, size)
    return numpy.dot(temp.T, temp)

def hilbert(size):
    h = numpy.zeros((size, size))
    for i in xrange(100):
        for j in xrange(100):
            h[i,j] = 1./ (i + j + 1.)
            if i==j: h[i,j] += 0.1
    return h

symb = {}
symb['L'] = T.matrix("L")
symb['g'] = T.vector("g")

vals = {}
vals['L'] = init_psd_mat(nparams).astype(floatX)
#vals['L'] = hilbert(nparams).astype(floatX)
vals['g'] = rng.rand(nparams).astype(floatX)

## now compute L^-1 g
vals['Linv_g'] = linalg.cho_solve(linalg.cho_factor(vals['L']), vals['g'])


def test_lincg_fletcher():
    rval = lincg.linear_cg_fletcher_reeves(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            rtol=1e-10,
            damp = 0.,
            maxit = 10000,
            floatX = floatX,
            profile=0)

    f = theano.function([symb['L'], symb['g']], rval[0])
    t1 = time.time()
    Linv_g = f(vals['L'], vals['g'])
    print 'test_lincg runtime (s):', time.time() - t1
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=5)


def test_lincg_fletcher_xinit():
    symb['xinit'] = T.vector('xinit')
    vals['xinit'] = rng.rand(nparams).astype(floatX)

    rval = lincg.linear_cg_fletcher_reeves(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            rtol=1e-10,
            damp = 0.,
            maxit = 10000,
            floatX = floatX,
            xinit = [symb['xinit']],
            profile=0)

    f = theano.function([symb['L'], symb['g'], symb['xinit']], rval[0])
    t1 = time.time()
    Linv_g = f(vals['L'], vals['g'], vals['xinit'])
    print 'test_lincg runtime (s):', time.time() - t1
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=5)


def test_lincg_polyak():
    rval = lincg.linear_cg_polyak_ribiere(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = None,
            rtol=1e-10,
            maxit = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g']], rval)
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['L'], vals['g'])
    print 'test_lincg_polyak runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=5)


def test_lincg_polyak_xinit():
    symb['xinit'] = T.vector('xinit')
    vals['xinit'] = rng.rand(nparams).astype(floatX)

    rval = lincg.linear_cg_polyak_ribiere(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = None,
            xinit = [symb['xinit']],
            rtol=1e-10,
            maxit = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g'], symb['xinit']], rval)
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['L'], vals['g'], vals['xinit'])
    print 'test_lincg_polyak runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=5)



def test_lincg_polyak_precond():
    symb['M'] = T.vector('M')
    vals['M'] = numpy.diag(vals['L'])

    rval = lincg.linear_cg_polyak_ribiere(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            M = [symb['M']],
            rtol=1e-10,
            maxit = 10000,
            floatX = floatX)

    f = theano.function([symb['L'], symb['g'], symb['M']], rval)
    t1 = time.time()
    [Linv_g, niter, rerr] = f(vals['L'], vals['g'], vals['M'])
    print 'test_lincg runtime (s):', time.time() - t1
    print '\t niter = ', niter
    print '\t residual error = ', rerr
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g'], decimal=5)

    ### test scipy implementation ###
    from scipy.sparse import linalg
    t1 = time.time()
    linalg.cg(vals['L'], vals['g'], maxiter=10000, tol=1e-10)
    print 'scipy.sparse.linalg.cg (no preconditioning): Elapsed ', time.time() - t1
    t1 = time.time()
    linalg.cg(vals['L'], vals['g'], maxiter=10000, tol=1e-10, M=numpy.diag(vals['M']))
    print 'scipy.sparse.linalg.cg (preconditioning): Elapsed ', time.time() - t1

import numpy
import time
import theano
import theano.tensor as T
from scipy import linalg

floatX = theano.config.floatX

from DBM import fisher
from DBM import natural

rng = numpy.random.RandomState(92832)
(M,N0,N1,N2) = (1024,10,11,12)

params = ['W','V','a','b','c']
nparam = N0*N1 + N1*N2 + N0 + N1 + N2

# initialize samples, parameter and x values
vals = {}
vals['v'] = rng.randint(low=0, high=2, size=(M,N0)).astype('float32')
vals['g'] = rng.randint(low=0, high=2, size=(M,N1)).astype('float32')
vals['h'] = rng.randint(low=0, high=2, size=(M,N2)).astype('float32')
vals['W'] = rng.rand(N0, N1).astype('float32')
vals['V'] = rng.rand(N1, N2).astype('float32')
vals['a'] = rng.rand(N0).astype('float32')
vals['b'] = rng.rand(N1).astype('float32')
vals['c'] = rng.rand(N2).astype('float32')
vals['x_W'] = rng.random_sample(vals['W'].shape).astype('float32')
vals['x_V'] = rng.random_sample(vals['V'].shape).astype('float32')
vals['x_a'] = rng.random_sample(vals['a'].shape).astype('float32')
vals['x_b'] = rng.random_sample(vals['b'].shape).astype('float32')
vals['x_c'] = rng.random_sample(vals['c'].shape).astype('float32')
vals['x'] = numpy.hstack((vals['x_W'].flatten(), vals['x_V'].flatten(),
                          vals['x_a'], vals['x_b'], vals['x_c']))

# compute sufficient statistics for each parameter
stats = {}
stats['W'] = vals['v'][:,:,None] * vals['g'][:,None,:]
stats['V'] = vals['g'][:,:,None] * vals['h'][:,None,:]
stats['a'] = vals['v']
stats['b'] = vals['g']
stats['c'] = vals['h']

# gradients are computed as the mean (acrss examples) of sufficient statistics
grads = {}
for param in params:
    grads[param] = stats[param].mean(axis=0)

# Initialize symbols for samples and parameters
symb = {}
for k in ['v','g','h','W','V','x_W','x_V']:
    symb[k] = T.matrix(k)
for k in ['a','b','c','x_a','x_b','x_c']:
    symb[k] = T.vector(k)

### numpy implementation ###
L = numpy.zeros((0,nparam), dtype='float32')
Minv = numpy.float32(1./M)
for i, pi in enumerate(params):
    dim = numpy.prod(vals[pi].shape)
    Li = numpy.zeros((dim,0))
    for j, pj in enumerate(params):
        lterm = (stats[pi] - grads[pi]).reshape(M,-1)
        rterm = (stats[pj] - grads[pj]).reshape(M,-1)
        Lij = Minv * numpy.dot(lterm.T, rterm)
        Li = numpy.hstack((Li, Lij))
    L = numpy.vstack((L, Li))

def test_compute_Lx():

    ## baseline result ##
    Lx = numpy.dot(L, vals['x'])
    Lx_w = Lx[:N0*N1].reshape(N0,N1)
    Lx_v = Lx[N0*N1 : N0*N1 + N1*N2].reshape(N1,N2)
    Lx_a = Lx[N0*N1 + N1*N2 : N0*N1 + N1*N2 + N0]
    Lx_b = Lx[N0*N1 + N1*N2 + N0 : N0*N1 + N1*N2 + N0 + N1]
    Lx_c = Lx[-N2:]

    # natural.compute_Lx implementation
    symb_inputs = [symb['v'], symb['g'], symb['h'],
                   symb['x_W'], symb['x_V'],
                   symb['x_a'], symb['x_b'], symb['x_c']]
    Lx = natural.compute_Lx(*symb_inputs)
    t1 = time.time()
    f = theano.function(symb_inputs, Lx)
    print 'natural.compute_Lx elapsed: ', time.time() - t1
    rvals = f(vals['v'], vals['g'], vals['h'],
              vals['x_W'], vals['x_V'],
              vals['x_a'], vals['x_b'], vals['x_c'])
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[4], decimal=3)

    # fisher.compute_Lx implementation
    energies = - T.sum(T.dot(symb['v'], symb['W']) * symb['g'], axis=1) \
               - T.sum(T.dot(symb['g'], symb['V']) * symb['h'], axis=1) \
               - T.dot(symb['v'], symb['a']) \
               - T.dot(symb['g'], symb['b']) \
               - T.dot(symb['h'], symb['c'])

    symb_params = [symb['W'], symb['V'], symb['a'], symb['b'], symb['c']]
    symb_x = [symb['x_W'], symb['x_V'], symb['x_a'], symb['x_b'], symb['x_c']]
    LLx = fisher.compute_Lx(energies, symb_params, symb_x)

    f_inputs = [symb['v'], symb['g'], symb['h']] + symb_params + symb_x
    f = theano.function(f_inputs, LLx)
    
    t1 = time.time()
    rvals = f(vals['v'], vals['g'], vals['h'],
              vals['W'], vals['V'], vals['a'], vals['b'], vals['c'],
              vals['x_W'], vals['x_V'], vals['x_a'], vals['x_b'], vals['x_c'])

    ### compare both implementation ###
    print 'fisher.compute_Lx elapsed: ', time.time() - t1
    numpy.testing.assert_almost_equal(Lx_w, rvals[0], decimal=3)
    numpy.testing.assert_almost_equal(Lx_v, rvals[1], decimal=3)
    numpy.testing.assert_almost_equal(Lx_a, rvals[2], decimal=3)
    numpy.testing.assert_almost_equal(Lx_b, rvals[3], decimal=3)
    numpy.testing.assert_almost_equal(Lx_c, rvals[4], decimal=3)

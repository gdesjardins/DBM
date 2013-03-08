import numpy
import time
import theano
import theano.tensor as T
from scipy import linalg
floatX = theano.config.floatX

from DBM import fisher
from DBM import natural

rng = numpy.random.RandomState(92832)
(M,N0,N1,N2) = (256,784,1000,500)

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

# Initialize symbols for samples and parameters
symb = {}
for k in ['v','g','h','W','V','x_W','x_V']:
    symb[k] = T.matrix(k)
for k in ['a','b','c','x_a','x_b','x_c']:
    symb[k] = T.vector(k)

def test_runtime():

    ### theano implementation ###
    energies = - T.sum(T.dot(symb['v'], symb['W']) * symb['g'], axis=1) \
               - T.sum(T.dot(symb['g'], symb['V']) * symb['h'], axis=1) \
               - T.dot(symb['v'], symb['a']) \
               - T.dot(symb['g'], symb['b']) \
               - T.dot(symb['h'], symb['c'])

    # Fisher Implementation
    symb_params = [symb['W'], symb['V'], symb['a'], symb['b'], symb['c']]
    symb_x = [symb['x_W'], symb['x_V'], symb['x_a'], symb['x_b'], symb['x_c']]
    f_inputs = [symb['v'], symb['g'], symb['h']] + symb_params + symb_x
    fisher_Lx = fisher.compute_Lx(energies, symb_params, symb_x)
    fisher_func = theano.function(f_inputs, fisher_Lx)

    samples = [symb['v'], symb['g'], symb['h']]
    symb_weights = [symb['x_W'], symb['x_V']]
    symb_biases = [symb['x_a'], symb['x_b'], symb['x_c']]
    f_inputs = [symb['v'], symb['g'], symb['h']] + symb_weights + symb_biases
    natural_Lx = natural.generic_compute_Lx(samples, symb_weights, symb_biases)
    natural_func = theano.function(f_inputs, natural_Lx)
    
    t1 = time.time()
    fisher_rval = fisher_func(vals['v'], vals['g'], vals['h'],
              vals['W'], vals['V'], vals['a'], vals['b'], vals['c'],
              vals['x_W'], vals['x_V'], vals['x_a'], vals['x_b'], vals['x_c'])
    print 'Fisher runtime (s): ', time.time() - t1

    t1 = time.time()
    nat_rval = natural_func(vals['v'], vals['g'], vals['h'],
              vals['x_W'], vals['x_V'], vals['x_a'], vals['x_b'], vals['x_c'])
    print 'Natural runtime (s): ', time.time() - t1

    ### make sure the two return the same thing ###
    for (rval1, rval2) in zip(fisher_rval, nat_rval):
        numpy.testing.assert_almost_equal(rval1, rval2, decimal=2)

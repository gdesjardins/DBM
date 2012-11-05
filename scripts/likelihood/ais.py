"""
This script implements Annealed Importance Sampling for estimating the partition
function of a 3-layer DBM, as described in:
On the Quantitative Analysis of Deep Belief Networks, Salakhutdinov & Murray
Deep Boltzmann Machines, Salakhutdinov & Hinton.

The DBM energy function being considered is the following:
    E(v, h1, h2) = - v^T W1 h1 - h1^T W2 h2 - v^T b0 - h1^T b1 - h2^T b2

This implementation works by interpolating between:

* the target distribution p_B(h1), obtained by marginalizing h0 and h2:
  p_B(h1) = \sum_{h0,h2} p(v, h1, h2)
          = 1/Z_B \exp(\sum_j h1_j b1_j)
                  \prod_i (1 + \exp(\beta_k \sum_j (b0_i + h1_j W1_ij)))
                  \prod_l (1 + \exp(\beta_k \sum_j (b2_l + h1_j W2_jl)))

* the baserate distribution p_A(h1):
  p_A(h1) = 1/Z_A \exp(- h1^T b1_A)

* using the interpolating distributions p_k(h^1) defined as:
  p_k*(h1) \proto p*_A^(1 - \beta_k) p*_B^(beta_k).

To maximize accuracy, the model p_A is chosen to be the maximum likelihood
solution when all weights are null (i.e. W1=W2=0). We refer to b1_A as the
baserate biases. They can be computed directly from the dataset. Note that
because all h1 units of p_A are independent, we can easily compute the partition
function Z_A, as:

    Z_A = 2^N0 2^N2 \prod_j (1 + exp(b1_j))

where N0 and N2 are the number of hidden units at layers 0 and 2.

The ratio of partition functions Z_B / Z_A is then estimated by averaging the
importance weights w^(m), given by:
    
            p*_1(h_0) p*_2(h_1)     p*_K(h_{K-1})
    w^(m) = --------- --------- ... -----------------,  where h_k ~ p_k(h1).
            p*_0(h_0) p*_1(h_1)     p*_{K-1}(h_{K-1})

    log w^(m) = FE_0(h_0) - FE_1(h_0) + FE_1(h_1) - FE_2(h_1) + etc.
"""
import numpy
import logging
import optparse

import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.datasets import mnist

floatX = theano.config.floatX
logging.basicConfig(level=logging.INFO)

def neg_sampling(dbm, nsamples, beta=1.0, h1bias_a=None):
    """
    Generate a sample from the intermediate distribution defined at inverse
    temperature `beta`, starting from state `nsamples`. See file docstring for
    equation of p_k(h1).

    Inputs
    ------
    dbm: dbm.DBM object
        DBM from which to sample.
    nsamples: array-like object of shared variables.
        nsamples[i] contains current samples from i-th layer.
    beta: scalar
        Temperature at which we should sample from the model.

    Returns
    -------
    new_nsamples: array-like object of symbolic matrices
        new_nsamples[i] contains new samples for i-th layer.
    """
    assert len(nsamples) == 3

    new_nsamples = [nsamples[i] for i in xrange(dbm.depth)]

    # contribution from model B, at temperature beta_k
    new_nsamples[0] = dbm.sample_hi_given(new_nsamples, 0, beta)
    new_nsamples[2] = dbm.sample_hi_given(new_nsamples, 2, beta)
    temp =  dbm.hi_given(new_nsamples, 1, beta, apply_sigmoid=False)
    # contribution from model A, at temperature (1 - beta_k)
    temp += h1bias_a * (1. - beta)
    h1_mean = T.nnet.sigmoid(temp)
    # now compute actual sample
    new_nsamples[1] = dbm.theano_rng.binomial(
                        size = (dbm.batch_size, dbm.n_u[1]),
                        n=1, p=h1_mean, dtype=floatX)
    return new_nsamples


def free_energy_at_beta(h1_sample, beta, h1bias_a=None):
    """
    Computes the free-energy of the sample `h1_sample`, for model p_k(h1).

    Inputs
    ------
    h1_sample: theano.shared
        Shared variable representing a sample of layer h1.
    beta: T.scalar
        Inverse temperature beta_k of model p_k(h1) at which to measure the free-energy.

    Returns
    -------
    Symbolic variable, free-energy of sample `h1_sample`, at inv. temp beta.
    """
    layer0_temp = model.bias[0] + T.dot(h1_sample, model.W[1].T)
    layer2_temp = model.bias[2] + T.dot(h1_sample, model.W[2])
    fe_bp_h1  = - T.sum(T.nnet.softplus(beta * layer0_temp), axis=1) \
                - T.sum(T.nnet.softplus(beta * layer2_temp), axis=1) \
                - T.dot(h1_sample, model.bias[1]) * beta \
                - T.dot(h1_sample, h1bias_a) * (1. - beta)
    return fe_bp_h1


def compute_log_ais_weights(free_energy_fn, sample_fn, betas):
    """
    Compute log of the AIS weights.
    TODO: remove dependency on global variable model.

    Inputs
    ------
    free_energy_fn: theano.function
        Function which, given temperature beta_k, computes the free energy
        of the samples stored in model.samples. This function should return
        a symbolic vector.
    sample_fn: theano.function
        Function which, given temperature beta_k, generates samples h1 ~
        p_k(h1). These samples are stored in model.nsamples[1].

    Returns
    -------
    log_ais_w: T.vector
        Vector containing log ais-weights.
    """
    # Initialize log-ais weights.
    log_ais_w = numpy.zeros(model.batch_size, dtype=floatX)
    # Iterate from inv. temperature beta_k=0 to beta_k=1...
    for i in range(len(betas) - 1):
        bp, bp1 = betas[i], betas[i+1]
        log_ais_w += free_energy_fn(bp) - free_energy_fn(bp1)
        sample_fn(bp1)
        if i % 1e3 == 0:
            logging.info('Temperature %f ' % bp1)
    return log_ais_w


def estimate_from_weights(log_ais_w):
    """ Safely computes the log-average of the ais-weights.

    Inputs
    ------
    log_ais_w: T.vector
        Symbolic vector containing log_ais_w^{(m)}.

    Returns
    -------
    dlogz: scalar
        log(Z_B) - log(Z_A).
    var_dlogz: scalar
        Variance of our estimator.
    """
    # Utility function for safely computing log-mean of the ais weights.
    ais_w = T.vector()
    max_ais_w = T.max(ais_w)
    dlogz = T.log(T.mean(T.exp(ais_w - max_ais_w))) + max_ais_w
    log_mean = theano.function([ais_w], dlogz, allow_input_downcast=False)

    # estimate the log-mean of the AIS weights
    dlogz = log_mean(log_ais_w)

    # estimate log-variance of the AIS weights
    # VAR(log(X)) \approx VAR(X) / E(X)^2 = E(X^2)/E(X)^2 - 1
    m = numpy.max(log_ais_w)
    var_dlogz = (log_ais_w.shape[0] *
                 numpy.sum(numpy.exp(2 * (log_ais_w - m))) /
                 numpy.sum(numpy.exp(log_ais_w - m)) ** 2 - 1.)

    return dlogz, var_dlogz


def compute_log_za(pa_h1_bias):
    """
    Compute the exact partition function of model p_A(h1).
    """
    log_za = numpy.sum(numpy.log(1 + numpy.exp(pa_h1_bias)))
    log_za += model.n_u[0] * numpy.log(2)
    log_za += model.n_u[2] * numpy.log(2)
    return log_za


def compute_test_set_likelihood(model, energy_fn, inference_fn, log_z, test_x):
    """
    Compute test set likelihood as below, where q is the variational
    approximation to the posterior p(h1,h2|v).

        ln p(v) \approx \sum_h q(h) E(v,h1,h2) + H(q) - ln Z

    See section 3.2 of DBM paper for details.

    Inputs:
    -------
    model: dbm.DBM
    energy_fn: theano.function
        Function which computes the (temperature 1) energy of the samples stored
        in model.samples. This function should return a symbolic vector.
    inference_fn: theano.function
        Inference function for DBM. Function takes a T.matrix as input (data)
        and returns a list of length `length(model.n_u)`, where the i-th element
        is an ndarray containing approximate samples of layer i.
    log_z: scalar
        Estimate partition function of `model`.
    test_x: numpy.ndarray
        Test set data, in dense design matrix format.

    Returns:
    --------
    Scalar, representing negative log-likelihood of test data under the model.
    """
    i = 0.
    likelihood = 0
    for i in xrange(0, len(test_x), model.batch_size):

        # recast data as floatX and apply preprocessing if required
        x = numpy.array(test_x[i:i + model.batch_size, :], dtype=floatX)

        # perform inference
        psamples = inference_fn(x)

        # entropy of h(q) adds contribution to variational lower-bound
        hq = 0
        for psample in psamples[1:]:
            temp = - psample * numpy.log(1e-5 + psample) \
                   - (1.-psample) * numpy.log(1. - psample + 1e-5)
            hq += numpy.sum(temp, axis=1)

        # copy into negative phase buffers to measure energy
        for ii, psample in enumerate(psamples):
            model.nsamples[ii].set_value(psample)

        # compute sum of likelihood for current buffer
        x_likelihood = numpy.sum(-energy_fn(1.0) + hq - log_z)

        # perform moving average of negative likelihood
        # divide by len(x) and not bufsize, since last buffer might be smaller
        likelihood = (i * likelihood + x_likelihood) / (i + len(x))

    return likelihood


def main(model, data):
    """
    Compute estimate of log-partition function and likelihood of data.X.

    Inputs:
    -------
    model: dbm.DBM
    data: pylearn2 dataset

    Returns:
    --------
    nll: scalar
        Negative log-likelihood of data.X under `model`.
    logz: scalar
        Estimate of log-partition function of `model`.
    """

    ##########################
    ## BUILD THEANO FUNCTIONS
    ##########################
    beta = T.scalar()
    
    # Build function to retrieve energy.
    E = model.energy(model.nsamples, beta)
    energy_fn = theano.function([beta], E)

    # Build inference function.
    new_psamples = model.e_step(model.psamples, n_steps=model.pos_mf_steps)
    inference_fn = theano.function([model.input], new_psamples)

    # Configure baserate bias for h1.
    temp = numpy.asarray(data.X, dtype=floatX)
    mean_data = numpy.mean(temp, axis=0)
    psamples = inference_fn(mean_data[None,:])
    mean_pos_h1 = numpy.minimum(psamples[1], 1-1e-5)
    mean_pos_h1 = numpy.maximum(mean_pos_h1, 1e-5)
    h1bias_a = -numpy.log(1./mean_pos_h1[0] - 1.)

    # Build Theano function to sample from interpolating distributions.
    updates = {}
    new_nsamples = neg_sampling(model, model.nsamples, beta=beta, h1bias_a=h1bias_a)
    for (nsample, new_nsample) in zip(model.nsamples, new_nsamples):
        updates[nsample] = new_nsample
    sample_fn = theano.function([beta], [], updates=updates, name='sample_func')

    ### Build function to compute free-energy of p_k(h1).
    fe_bp_h1 = free_energy_at_beta(model.nsamples[1], beta, h1bias_a)
    free_energy_fn = theano.function([beta], fe_bp_h1)


    ###########
    ## RUN AIS
    ###########

    # Generate exact sample for the base model.
    for i, nsample_i in enumerate(model.nsamples):
        bias = h1bias_a if i==1 else model.bias[i].get_value()
        hi_mean_vec = 1. / (1. + numpy.exp(-bias))
        hi_mean = numpy.tile(hi_mean_vec, (model.batch_size, 1))
        r = numpy.random.random_sample(hi_mean.shape)
        hi_sample = numpy.array(hi_mean > r, dtype=floatX)
        nsample_i.set_value(hi_sample)


    # default configuration for interpolating distributions
    betas = numpy.cast[floatX](
        numpy.hstack((numpy.linspace(0, 0.5, 1e3),
                     numpy.linspace(0.5, 0.9, 1e4),
                     numpy.linspace(0.9, 1.0, 1e4))))

    log_ais_w = compute_log_ais_weights(free_energy_fn, sample_fn, betas)
    dlogz, var_dlogz = estimate_from_weights(log_ais_w)
    log_za = compute_log_za(h1bias_a)
    log_z = log_za + dlogz
    logging.info('log_z = %f' % log_z)
    logging.info('log_za = %f' % log_za)
    logging.info('dlogz = %f' % dlogz)
    logging.info('var_dlogz = %f' % var_dlogz)

    nll = compute_test_set_likelihood(model, energy_fn, inference_fn, log_z, data.X)
    logging.info('Test set likelihood = %f' % nll)

    return (nll, log_z)


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option('-m', '--model', action='store', type='string', dest='path')
    parser.add_option('--dataset', action='store', type='string', dest='dataset')
    (opts, args) = parser.parse_args()

    # Load model and retrieve parameters.
    model = serial.load(opts.path)
    # Load dataset.
    assert opts.dataset in ['train','test']
    dataset = mnist.MNIST(opts.dataset)

    main(model, dataset)
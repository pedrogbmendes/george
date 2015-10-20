import george
import numpy as np
import matplotlib.pyplot as plt
import emcee



ndims = 5		# total number of dimensions
which_dimension=3
degree=4		# if the degree is to low, it gets numerically unstable as many configurations have essentially zero probability


kernel = george.kernels.BayesianLinearRegressionKernel(ndims,which_dimension,degree)
gp = george.GP(kernel)



# just to check that you can form product and sum kernels
kernel2 = george.kernels.ExpSquaredKernel(0.5,ndims) * kernel
kernel3 = george.kernels.ExpSquaredKernel(0.5,ndims) + kernel

num_points = 10
X = np.array([np.sort(np.random.rand(num_points)*0.8) for d in range(ndims)]).T
Y = np.sort(np.random.rand(num_points))[::-1]

# kernel parameter DO NOT live on a log scale
gp.kernel[:] = np.array([0.1]*(degree+1))
# initialize kernel computation
gp.compute(X, 0.1)


plt.scatter(X[:,which_dimension],Y)
plt.show()



def lnprob(p):
	# Update the kernel and compute the lnlikelihood.

	# prior are not trivial here! A lot of the configuration yield a 
	logprior = -np.sum(p**2/10)
	gp.kernel[:] = p
	lh = logprior + gp.lnlikelihood(Y, quiet=True)
	print(p, lh)
	return(lh)


# Set up the sampler.
nwalkers = 2*(degree+1)
sampler = emcee.EnsembleSampler(nwalkers, degree+1, lnprob)



# if you want to do the kernel with (1, (1-x)^2) as basis functions,
# set degree = 2 and initialize the walkers such that the parameters obey
# gp.kernel[1] = -2 gp.kernel[2]
p0 = [np.random.randn(degree+1) for i in range(nwalkers)]



print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 200)

print("Running production chain")
chain, _, _=sampler.run_mcmc(p0, 300)




print(sampler.chain[2,-1])
gp.kernel.pars = sampler.chain[2, -1]


t = np.array([np.linspace(0,1,20) for d in range(ndims)]).T
mu, cov = gp.predict(Y, t)
std = np.sqrt(np.diag(cov))


plt.fill_between(t[:,which_dimension], mu-std, mu+std, alpha=0.3)
plt.scatter(X[:,which_dimension],Y)
plt.plot(t[:,which_dimension],mu)
plt.show()

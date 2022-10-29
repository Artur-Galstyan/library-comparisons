from jax import grad, jit


@jit
def f(x):
    return x**4 - x**2 + 15


f_prime = grad(f)
f_prime_prime = grad(f_prime)


print(f_prime(5.0))
print(f_prime_prime(5.0))

DEAP-SSF
========

![Language: Python](https://img.shields.io/badge/language-python-blue.svg)

**DEAP-SSF** Simple Stochastic Fitness. A simple and unobtrusive way to use [DEAP framework](https://github.com/deap/deap) under stochastic uncertain.

### Basic example
Find the "magicbox" contains the higher number. But the getter is noisy...
```
    class MagicBox:
        
```


### Basic usage
The package provides `StochasticFitness` class, with almost the same api of [DEAP.base.Fitness](https://deap.readthedocs.io/en/master/api/base.html). You can use that at the same way, but:

* you can't set directly the fitness value, that is calculated automatically from the samples (but, ad in `DEAP.base.Fitness`, you must invalidate them if makes change to genotype!);
* you must provides the samples, calling `addSamples` method.

A relevant number of samples can be added immediately after fitness invalidation (due to crossover or mutation), but if the sampling function is not so fast it's convenient to call it lazily. **DEAP-SSF** suggest when it is time to add new samples to an individual (see next paragraph).

### Reliability

The class provides a `reliability` attribute, an indicator of the quality of the solutions. The reliability 

### Basic concept
In real-world problems, loss function can return a lot of different value, from which can be calculated some stats. Usually, better **sample mean** means better solutions, and lower **sample variance** means better solution ([risk adversion](https://en.wikipedia.org/wiki/Risk_aversion)).

Also, a solution is better or worse than another with a confidence value. A low confidence value can mean: 1) the solutions are very similar 2) there are too few samples.



## Example

### Italian trucks's hold problem
Like a classical knapsack problem, we want to maximize the stored value of items keeping the weight under the trucks limit. But... we could take a pothole!

From benchmarks, the probability and the impact on weight to take a pothole are:

_ | No pothole | Small pothole | Medium pothole | Huge pothole
--- | --- | --- | --- | ---
Weight | 1.00 x | 1.02 x | 1.05 x | 1.25 x
Occurrency | 60 % | 20 % | 15 % | 5 %

So, define:
```
def evalShipsHold(items):
    weight, value = 0.0, 0.0
    
    for item in items:
        weight += item.weight
        value += item.value
    
    # if weight is greater than max, the truck
    # can't start and then the value is zero
    if weight > MAXWEIGHT:
        return 0 
    
    # now, draw one sample from the pothole weight-multipliers
    distr = [1.00, 1.02, 1.05, 1.25] 
    p = [0.60, 0.20, 0.15, 0.5] 
    [mul] = random.choice(distr, weights=p, k=1)
    
    # update weight according with multiplier
    weight = weight * mul
    
    # if weight become greater than max during trip,
    # the truck breaks, the accident destroy the items
    # and you must refund them
    if weight > MAXWEIGHT:
        return -1 * value
        
    # finally, if the shipment ends...
    return value
```

And simply use:

```
    toolbox.register("sample", paSamples, sym=sym)
    eaSimpleStoch()
```


### Contacts and license
* Massimo Rebuglio [massimo.rebuglio@polito.it](mailto:massimo.rebuglio@polito.it) 

DEAP-SSF is [free and open-source software](https://en.wikipedia.org/wiki/Free_and_open-source_software), and it is distributed under the permissive [EUPL License](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12).
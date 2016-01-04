# pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-public-methods
"""
Population.py
"""

import bisect
import tabulate
import numpy as np
import random as r
from multiprocessing import Pool

from eugene.Util import ProgressBar, rmse
from eugene.Tree import random_tree
from eugene.List import random_list
from eugene.String import random_string
from eugene.Individual import Individual

def par_gene_expression(par_input):
    """Create function to run gene expression  in parallel"""
    # Separate inputs
    individual, error_function, target = par_input
    # Calculate expression
    gene_expression = individual.compute_gene_expression(error_function, target)
    return gene_expression

def par_obj_func(par_input):
    """Create function to run objective functino in parallel"""
    # Separate inputs
    objective_function, expression, expression_scale = par_input
    # Calculate expression
    fitness = objective_function(expression, expression_scale)
    return fitness

class Population(object):
    """
    Defines Population of Individuals with ability to create generations and evaluate fitness.

    Parameters
    individual_type      : Data structure of individual: 'tree', 'list', 'string', defaults to 'tree'
    init_population_size : The number of Individuals to create in the initial population, defaults to 1000
    item_factory         : Factory function to create new Individuals, assumed to be fully or semi random, defaults to None
    eval_function        : Converts the encoding of an individual (tree of strings, list of instructions) into the actual
                           solution or result, defaults to None
    objective_function   : Calculates the fitness or performance of solution (e.g. weighted combination of error and
                           complexity), defaults to None
    error_function       : Calculates the error between an individual solution and the target, defaults to eugene.Util.rmse
                           (Standard Root Mean Squared Error)
    target               : The 'ideal' solution or the thing you are trying to evolve 'towards', used to calculate solution
                           error, defaults to None
    max_generations      : The maximum number of generations the population is allowed to evolve, defaults to 1000
    init_ind_size        : Size limit of each individual, set number of times item_factory is called during creation.  This will
                           set the max depth for tree type individuals and the max length for string and list type individuals,
                            defaults to 3
    stagnation_timeout   : The number of generations the average fitness can remain the same before Population growth stagnates
                           and the evolution ends, defaults to 20
    rank_pressure        : Selective Pressure used rank_roulette selection process (currently not used), defaults to 2.0
    elitism              : Percent of Top Ranked Population that passes to next generation, defaults to 0.02
    replication          : Percent of Population that duplicates itself each generation, defaults to 0.28
    mating               : Percent of Population that mates each generations, defaults to 0.6
    mutation             : Percent of Population that will mutate, defaults to 0.1
    parallel             : Perform evolution in parallel, defaults to False
    pruning              : Removes inefficiencies in Tree based Individuals, defaults to False

    """

    def __init__(
            self,
            individual_type='tree',
            init_population_size=1000,
            item_factory=None,
            eval_function=None,
            objective_function=None,
            error_function=rmse,
            target=None,
            max_generations=1000,
            init_ind_size=3,
            stagnation_timeout=20,
            rank_pressure=2.0,
            elitism=0.02,
            replication=0.28,
            mating=0.6,
            mutation=0.1,
            parallel=False,
            pruning=False
        ):
        # parameters
        self.individual_type = individual_type
        self.init_population_size = init_population_size
        self.item_factory = item_factory
        self.eval_function = eval_function
        self.objective_function = objective_function
        self.error_function = error_function
        self.target = target
        self.max_generations = max_generations
        self.init_ind_size = init_ind_size
        self.stagnation_timeout = stagnation_timeout
        self.rank_pressure = rank_pressure
        self.elitism = elitism
        self.replication = replication
        self.mating = mating
        self.mutation = mutation
        self.parallel = parallel
        self.pruning = pruning
        # initialize variables
        self.created = False
        self.individuals = []
        self.ranking = []
        self.generation = 0
        self.expression_scale = np.array([1.0, 1.0, 1.0])
        self.history = {
            'fitness' : [],
            'error' : [],
            'time' : [],
            'complexity' : [],
            'most_fit': []
        }
        # cached values
        self._fitness = np.array([])
        self._ranked = False

    @property
    def size(self):
        """Return the size of the population"""
        return len(self.individuals)

    @property
    def fitness(self):
        """Return the fitness of each individual in population"""
        if self._fitness.shape == (0, ):
            self.calc_fitness()
        return self._fitness

    @property
    def stagnate(self):
        """
        Determine if the population has stagnated and reached local min where average fitness over last n generations has
        not changed.
        """
        if self.generation <= self.stagnation_timeout:
            return False
        else:
            last_gen2 = self.history['fitness'][(self.generation - 2 - self.stagnation_timeout):(self.generation - 2)]
            last_gen1 = self.history['fitness'][(self.generation - 1 - self.stagnation_timeout):(self.generation - 1)]
            return (last_gen2 == last_gen1) and not np.isinf(last_gen1).all()

    def describe(self):
        """Print out all data"""
        self.describe_init()
        print '\n'
        self.describe_current()

    def describe_init(self):
        """Print out parameters used to intialize population"""
        print '\nPopulation Initialized w/ Parameters:'
        data = [
            ['Initial number of individuals:', self.init_population_size],
            ['Initial size of individuals:', self.init_ind_size],
            ['Max. number of generations:', self.max_generations],
            ['Parallel fitness turned on:', self.parallel],
            ['Stagnation factor:', self.stagnation_timeout],
            ['Percent of Elitism:', self.elitism],
            ['Percent of replication:', self.replication],
            ['Percent of mating:', self.mating],
            ['Percent of mutation:', self.mutation],
            ['Selective pressure:', self.rank_pressure]
        ]
        print tabulate.tabulate(data)

    def describe_current(self):
        """Print out status about current population"""
        print '\nCurrent Population Status:'
        # initialize VARIABLES
        data = [
            ['Current generation:', self.generation],
            ['Number of individuals:', self.size]
        ]
        # if fitness is empty
        if self.fitness.shape == (0,):
            data.extend([
                ['Max fitness:', 0.0],
                ['Average fitness:', 0.0],
                ['Min fitness:', 0.0],
            ])
        else:
            data.extend([
                ['Max fitness:', self.fitness.max()],
                ['Average fitness:', self.fitness.mean()],
                ['Min fitness:', self.fitness.min()],
            ])
        print tabulate.tabulate(data)
        if self.fitness.max() == self.fitness.mean() == self.fitness.min():
            print '\n'
            print 'Constant Fitness: It looks like the template objective functions is being used!'
            print '                  Please add your own with '

    # @profile
    def initialize(self, seed=None):
        """Initialize a population based on seed or random generation"""

        self.describe_init()
        self.created = True
        if seed:
            print '\nUsing seed for inital population'
            self.individuals = seed
        else:
            print '\nInitializing Population with Individuals composed of random {}s:'.format(self.individual_type.title())
            pb = ProgressBar(self.init_population_size)
            uid = 0
            while len(self.individuals) < self.init_population_size:
                if self.individual_type == 'tree':
                    # generate a random expression tree
                    tree = random_tree(self.init_ind_size)
                    # prune inefficiencies from the tree
                    if self.pruning:
                        tree.prune()
                    # create an individual from this expression tree
                    individual = Individual(tree)

                elif self.individual_type == 'list':
                    # generate a random list
                    l = random_list(self.init_ind_size, self.item_factory, self.eval_function, uid)
                    # create an individual from this list
                    individual = Individual(l)

                elif self.individual_type == 'string':
                    # generate a random string
                    s = random_string(self.init_ind_size, self.item_factory, self.eval_function, uid)
                    # create an individual from this string
                    individual = Individual(s)

                else:
                    print '\nWarning: individual not defined'

                # check for genes
                gene_expression = individual.compute_gene_expression(self.error_function, self.target)
                # if there is some non-infinite error, add to the population
                if not np.isinf(gene_expression[0]):
                    uid += 1
                    self.individuals.append(individual)
                    pb.animate(self.size)

        print '\n'
        self.describe_current()

    # @profile
    def calc_fitness(self):
        """Calculate the fitness of each individual."""

        if self.parallel:
            # Parallel gene expression
            pool = Pool()
            expression = pool.map(par_gene_expression, [(i, self.error_function, self.target) for i in self.individuals])
            pool.close()
            pool.join()
            expression_scale = np.array(np.ma.masked_invalid(expression).max(axis=0))
            # Parallel objective fitness
            pool = Pool()
            fitness = pool.map(par_obj_func, [(self.objective_function, e, expression_scale) for e in expression])
            pool.close()
            pool.join()

        else:
            # Parallel gene expression
            expression = np.array([i.compute_gene_expression(self.error_function, self.target) for i in self.individuals])
            expression_scale = np.array(np.ma.masked_invalid(expression).max(axis=0))
            # Parallel objective fitness
            fitness = self.objective_function(expression, expression_scale)

        # Cache results
        self._fitness = np.array(fitness)

        # Store history of results
        max_expr = np.array(np.ma.masked_invalid(expression).max(axis=0)) / expression_scale
        mean_expr = np.array(np.ma.masked_invalid(expression).mean(axis=0)) / expression_scale
        min_expr = np.array(np.ma.masked_invalid(expression).min(axis=0)) / expression_scale
        mfit = np.ma.masked_invalid(self._fitness)
        self.history['fitness'].append((mfit.max(), mfit.mean(), mfit.min()))
        self.history['error'].append((max_expr[0], mean_expr[0], min_expr[0]))
        self.history['time'].append((max_expr[1], mean_expr[1], min_expr[1]))
        self.history['complexity'].append((max_expr[2], mean_expr[2], min_expr[2]))
        self.history['most_fit'].append(self.most_fit())

        # Rank
        self.rank()

    # @profile
    def rank(self):
        """Create ranking of individuals"""
        if not self._ranked:
            self.ranking = zip(self.fitness, self.individuals)
            self.ranking.sort()
            self._ranked = True

    # @profile
    def roulette(self, number=None):
        """Select parent pairs based on roulette method (probability proportional to fitness)"""
        number = number if number else self.size
        selections = []

        # unpack
        ranked_fitness, ranked_individuals = (list(i) for i in zip(*self.ranking))
        ranked_fitness = np.array(ranked_fitness)

        # calculate weighted probability proportial to fitness
        fitness_probability = ranked_fitness / np.ma.masked_invalid(ranked_fitness).sum()
        cum_prob_dist = np.array(np.ma.masked_invalid(fitness_probability).cumsum())

        # randomly select two individuals with weighted probability proportial to fitness
        selections = [ranked_individuals[bisect.bisect(cum_prob_dist, r.random() * cum_prob_dist[-1])] for _ in xrange(number)]
        return selections

    def stochastic(self, number=None):
        """Select parent pairs based on stochastic method (probability uniform across fitness)"""
        number = number if number else self.size

        # unpack
        ranked_fitness, ranked_individuals = (list(i) for i in zip(*self.ranking))
        ranked_fitness = np.array(ranked_fitness)

        # calculate weighted probability proportial to fitness
        fitness_probability = ranked_fitness / np.ma.masked_invalid(ranked_fitness).sum()
        cum_prob_dist = np.array(np.ma.masked_invalid(fitness_probability).cumsum())

        # determine uniform points
        p_dist = 1 / float(number)
        p0 = p_dist * r.random()
        points = p0 + p_dist * np.array(range(0, number))

        # randomly select individuals with weighted probability proportial to fitness
        selections = [ranked_individuals[bisect.bisect(cum_prob_dist, p * cum_prob_dist[-1])] for p in points]
        return selections

    def tournament(self, number=None, tournaments=4):
        """Select parent pairs based on tournament method (random tournaments amoung individuals where fitness wins)"""
        number = number if number else self.size
        selections = []
        for _ in xrange(number):
            # select group of random competitors
            competitors = [self.ranking[i] for i in list(np.random.random_integers(0, self.size-1, tournaments))]
            # group compete in fitness tournament (local group sorting)
            competitors.sort()
            # select most fit from each group
            winner = competitors[-1]
            selections.append(winner[1])
        return selections

    def rank_roulette(self, number=None, pressure=2):
        """Select parent pairs based on rank roulette method (probability proportional to fitness rank)"""
        number = number if number else self.size
        selections = []

        # unpack
        _, ranked_individuals = (list(i) for i in zip(*self.ranking))

        # create a scaled rank by fitness (individuals already sorted, so just create rank range, then scale)
        n = self.size
        rank = range(1, n + 1)
        scaled_rank = 2.0 - pressure + (2.0 * (pressure - 1) * (np.array(rank) - 1) / (n - 1))

        # calculate weighted probability proportial to scaled rank
        scaled_rank_probability = scaled_rank / np.ma.masked_invalid(scaled_rank).sum()
        cum_prob_dist = np.array(np.ma.masked_invalid(scaled_rank_probability).cumsum())

        for _ in xrange(number):
            # randomly select individuals with weighted probability proportial to scaled rank
            p1 = ranked_individuals[bisect.bisect(cum_prob_dist, r.random() * cum_prob_dist[-1])]
            selections.append(p1)
        return selections

    # @profile
    def select(self, number=None):
        """Select individuals thru various methods"""
        selections = self.roulette(number)
        return selections

    # @profile
    def create_generation(self):
        """Create the next generations, this is main function that loops"""

        # determine fitness of current generations and log average fitness
        self.calc_fitness()

        # rank individuals by fitness
        self.rank()

        # create next generation
        elite_num = int(round(self.size * self.elitism))
        replicate_num = int(round(self.size * self.replication))
        mate_num = int(round(self.size * self.mating))
        mutate_num = int(round(self.size * self.mutation))
        # split mate_num in half_size (2 parents = 2 children)
        mate_num = int(round(mate_num/2.0))

        # propogate elite
        next_generation = [i[1] for i in self.ranking[-elite_num:]]

        # replicate
        next_generation.extend(self.select(replicate_num))

        # crossover mate
        parent_pairs = zip(self.select(mate_num), self.select(mate_num))
        child_pairs = [p1.crossover(p2) for p1, p2 in parent_pairs]
        children = [child for pair in child_pairs for child in pair]
        next_generation.extend(children)

        # mutate
        mutants = self.select(mutate_num)
        for m in mutants:
            m.mutate(self.pruning)
        next_generation.extend(mutants)

        # Keep population the same size
        self.individuals = next_generation[:self.size]

        # Reset all individual UIDs and eval flag
        for i, ind in enumerate(self.individuals):
            ind.chromosomes.uid = i
            ind.chromosomes.evaluated = False

        # clear cached values
        self._fitness = np.array([])
        self._ranked = False

        # log generation
        self.generation += 1

        return None

    # @profile
    def evolve(self, number_of_generations=None):
        """Evolve the Population (run the algorithm)"""

        number_of_generations = number_of_generations if number_of_generations else self.max_generations
        pb = ProgressBar(number_of_generations)

        while self.generation < number_of_generations and not self.stagnate:
            self.create_generation()
            pb.animate(self.generation)

        if self.stagnate:
            print 'population became stagnate'
        print self.generation, 'generations'

    def run(self, number_of_generations=None):
        """Alias for the evolve function"""
        self.evolve(number_of_generations=number_of_generations)

    def most_fit(self):
        """Return the most fit individual"""

        # make sure the individuals have been ranked
        self.rank()
        return self.ranking[-1][1]

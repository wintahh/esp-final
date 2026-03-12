import neat
import pickle
from demomunk import Game
import os
import tempfile

def generate_neat_config_object(num_inputs, num_outputs):
    config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 500
pop_size              = 250
reset_on_extinction   = False
no_fitness_termination = True

[DefaultGenome]
num_inputs            = {num_inputs}
num_outputs           = {num_outputs}
num_hidden            = 0
initial_connection    = full

enabled_default         = False
enabled_mutate_rate     = 0.1
enabled_rate_to_true_add  = 0
enabled_rate_to_false_add = 0

feed_forward          = True
compatibility_disjoint_coefficient = 1
compatibility_weight_coefficient = 1

activation_default=random
activation_mutate_rate= 0.1
activation_options=tanh sigmoid sin

aggregation_default=random
aggregation_mutate_rate=0.1
aggregation_options=sum product mean

bias_init_type = gaussian
bias_init_mean        = 0.0
bias_init_stdev       = 1.0
bias_max_value        = 20.0
bias_min_value        = -20.0
bias_mutate_power     = 0.3
bias_mutate_rate      = 0.8
bias_replace_rate     = 0.1

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 20
weight_min_value        = -20
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.9
weight_replace_rate     = 0.1

conn_add_prob         = 0.6
conn_delete_prob      = 0.25
node_add_prob         = 0.15
node_delete_prob      = 0.15

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

single_structural_mutation = false
structural_mutation_surer  = default

[DefaultSpeciesSet]
compatibility_threshold = 4.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 1

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.1
min_species_size   = 1
"""

    # tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    tmp.write(config_text)
    tmp.flush()
    tmp.close()

    # load config from tempfile
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        tmp.name
    )

    # delete tempfile
    os.unlink(tmp.name)

    return config

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game(render=False)
    fitness = game.run_genome(net)
    genome.fitness = fitness
    return fitness


def run_neat(generations=50, num_workers=None):
    game = Game(render=False)

    # temporary creature in order to generate config. 
    # kinda janky, but its whatever.
    game.spawn_creatures(1)
    creature = game.creatures[0]

    num_inputs = 3*len(creature.bodies) + 2
    num_outputs = len(creature.springs)

    game.creatures = [] # remove temp creature
    viewer = Game(render=True)

    config = generate_neat_config_object(num_inputs,num_outputs)

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # use all cores if num_workers=None
    if num_workers is None:
        from multiprocessing import cpu_count
        num_workers = cpu_count()

    print(f"Using {num_workers} parallel workers for evaluation.")

    pe = neat.ParallelEvaluator(num_workers, eval_genome)

    winner = None

    for gen in range(generations):    # every gen, get species_best
        print(f"\n---- Generation {gen} ----")

        # evaluate genomes
        genomes = list(pop.population.items())
        pe.evaluate(genomes, config)

        # collect best genome per species
        species_best = []

        for sid, species in pop.species.species.items(): # sid is species id
            members = species.members
            best = max(members.values(), key=lambda g: g.fitness if g.fitness is not None else -999999)
            species_best.append(best)

        print("Species best genomes:", len(species_best))
        nets = [neat.nn.FeedForwardNetwork.create(g, config) for g in species_best]

        viewer.run_multiple_genomes(nets)

        # save best overall genome
        best_genome = max(pop.population.values(), key=lambda g: g.fitness)
        winner = best_genome

        # reproduce next gen
        pop.population = pop.reproduction.reproduce(
            config, pop.species, config.pop_size, pop.generation
        )

        if not pop.species.species:
            pop.population = pop.reproduction.create_new(
                config.genome_type, config.genome_config, config.pop_size
            )

        pop.species.speciate(config, pop.population, pop.generation)

        pop.generation += 1

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome:\n", winner)

if __name__ == "__main__":
    run_neat(generations=1000000)


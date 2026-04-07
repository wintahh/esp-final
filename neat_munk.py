import neat
import pickle
import os
import tempfile
import json
import threading
from multiprocessing import cpu_count

from demomunk import Game

def build_neat_config(num_inputs, num_outputs): # wow im so good at coding
    config_text = f"""
[NEAT]
fitness_criterion       = max
fitness_threshold       = 500
pop_size                = 250
reset_on_extinction     = False
no_fitness_termination  = True

[DefaultGenome]
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
num_hidden              = 0
initial_connection      = full

enabled_default         = False
enabled_mutate_rate     = 0.1
enabled_rate_to_true_add  = 0
enabled_rate_to_false_add = 0

feed_forward            = True
compatibility_disjoint_coefficient = 1
compatibility_weight_coefficient   = 1

activation_default      = random
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid sin

aggregation_default     = random
aggregation_mutate_rate = 0.1
aggregation_options     = sum product mean

bias_init_type          = gaussian
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 20.0
bias_min_value          = -20.0
bias_mutate_power       = 0.3
bias_mutate_rate        = 0.8
bias_replace_rate       = 0.1

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 20
weight_min_value        = -20
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.9
weight_replace_rate     = 0.1

conn_add_prob           = 0.6
conn_delete_prob        = 0.25
node_add_prob           = 0.15
node_delete_prob        = 0.15

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
species_fitness_func    = max
max_stagnation          = 15
species_elitism         = 1

[DefaultReproduction]
elitism             = 2
survival_threshold  = 0.1
min_species_size    = 1
"""
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".cfg")
    tmp.write(config_text)
    tmp.close()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        tmp.name,
    )
    os.unlink(tmp.name)
    return config


def get_genome_dimensions(): # it took me hours to remember i had creature data and that i could use that instead of doing that jank that i did b4
    with open("creatures/creature3.json") as f: # TODO : sync this with creature in demomunk
        data = json.load(f)
    num_bodies  = len(data["bodies"])
    num_springs = sum(1 for j in data["joints"] if j["type"] == "pivot" and j.get("actuated", False))
    num_inputs  = 2 + num_bodies * 3
    num_outputs = num_springs
    return num_inputs, num_outputs

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = Game(render=False).run_genome(net)
    genome.fitness = fitness
    return fitness

def run_neat(generations=1000000, num_workers=None):
    num_inputs, num_outputs = get_genome_dimensions()
    config = build_neat_config(num_inputs, num_outputs)
    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())

    num_workers = num_workers or cpu_count()
    print(f"Using {num_workers} parallel workers.")

    evaluator = neat.ParallelEvaluator(num_workers, eval_genome)
    viewer = Game(render=True)
    best = None

    # evaluate gen 0 before the loop
    eval_thread = threading.Thread(target=evaluator.evaluate, args=(list(pop.population.items()), config))
    eval_thread.start()
    eval_thread.join()

    for gen in range(generations):
        print(f"\n---- Generation {gen} ----")

        species_best = [
            (sid, max(sp.members.values(), key=lambda g: g.fitness or -999999))
            for sid, sp in sorted(pop.species.species.items())
        ]
        sids = [sid for sid, _ in species_best]
        nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in species_best]
        best = max(pop.population.values(), key=lambda g: g.fitness or -999999)

        # reproduce to get next gen ready
        pop.population = pop.reproduction.reproduce(config, pop.species, config.pop_size, pop.generation)
        if not pop.species.species:
            pop.population = pop.reproduction.create_new(config.genome_type, config.genome_config, config.pop_size)
        pop.species.speciate(config, pop.population, pop.generation)
        pop.generation += 1

        # start evaluating next gen while current gen renders
        eval_thread = threading.Thread(target=evaluator.evaluate, args=(list(pop.population.items()), config))
        eval_thread.start()

        print(f"Rendering {len(species_best)} species representatives.")
        viewer.run_multiple_genomes(nets, sids)

        eval_thread.join()  # wait for eval if rendering finished first

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(best, f)

    print("\nBest genome:\n", best)

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(best, f)

    print("\nBest genome:\n", best)

if __name__ == "__main__":
    run_neat()

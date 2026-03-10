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
compatibility_threshold = 3.0

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

with open("genomes/creat3-frog.pkl", "rb") as f:
    winner = pickle.load(f)

game = Game(render=False)
num_inputs = 3*len(game.creature.bodies) + 2
num_outputs = len(game.creature.springs)

config = generate_neat_config_object(num_inputs, num_outputs)

print("\nLoaded genome:\n", winner)

net = neat.nn.FeedForwardNetwork.create(winner, config)

game = Game(render=True)
game.run_genome(net, render=True)
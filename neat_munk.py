import neat
import pickle
from demomunk import Game

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game(render=False)
    fitness = game.run_genome(net, render=False)
    genome.fitness = fitness
    return fitness

def run_neat(config_file, generations=50, num_workers=None):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Use all cores if num_workers=None
    if num_workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()

    print(f"Using {num_workers} parallel workers for evaluation.")

    pe = neat.ParallelEvaluator(num_workers, eval_genome)

    # Run NEAT evolution using parallel evaluation
    winner = pop.run(pe.evaluate, generations)

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome:\n", winner)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = Game(render=True)
    game.run_genome(net, render=True)

if __name__ == "__main__":
    config_path = "fig"
    run_neat(config_path, generations=15)

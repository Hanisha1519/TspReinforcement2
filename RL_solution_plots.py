from RL_Qnnet_training import init_model, is_state_final, total_distance
import os
from Q_nnet import *
import matplotlib.pyplot as plt
from generate_data import *
from Benchmark_genetic import genetic_algorithm_benchmark
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 22
FOLDER_NAME = './models'  # where to checkpoint the best models
Q_func, Q_net, optimizer, lr_scheduler = init_model(os.path.join(FOLDER_NAME, 'ep_3248_length_7.683841608464718.pt'))


def plot_sol(coords, mat, solution):
    plt.scatter(coords[:, 0], coords[:, 1], label=coords)
    n = len(coords)

    for idx in range(n - 1):
        i, next_i = solution[idx], solution[idx + 1]
        plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8,
                 label=coords)

    i, next_i = solution[-1], solution[0]
    plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8, )
    plt.plot(coords[solution[0], 0], coords[solution[0], 1], 'x', markersize=40)
    k = 0
    for i, j in zip(coords[:, 0], coords[:, 1]):
        #         print("i here", i)
        plt.text(i, j, '({})'.format(k))
        k = k + 1
    plt.show()

NR_NODES = 20
coords_list, W_np = get_graph_mat(n=NR_NODES)
plot_graph(coords_list, W_np)

best_state, best_fitness = genetic_algorithm_benchmark(coords_list)

W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)

solution = [random.randint(0, NR_NODES - 1)]
current_state = State(partial_solution=solution, W=W, coords=coords_list)
current_state_tsr = state2tens(current_state)


while not is_state_final(current_state):
    next_node, est_reward = Q_func.get_best_action(current_state_tsr,
                                                   current_state)

    solution = solution + [next_node]
    current_state = State(partial_solution=solution, W=W, coords=coords_list)
    current_state_tsr = state2tens(current_state)

plt.figure()
print("RL solution here", solution)
print("RL shortest length", total_distance(solution, W))

plt.title('RL model / len = {}'.format(total_distance(solution, W)))
plot_sol(coords_list, W, solution)

plt.figure()
# random_solution = list(range(NR_NODES))
plt.title('Genetic Algorithm / len = {}'.format(best_fitness))
plot_sol(coords_list, W, best_state)

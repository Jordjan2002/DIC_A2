import itertools
import numpy as np
import matplotlib.pyplot as plt
from train_dqn import train_agent
from train_dqn import plot_learning_curve

# Hyperparameter grid
alphas = [1e-4, 5e-4, 1e-3] #learning rate
epsilon_decays = [0.99, 0.999]
target_updates = [100, 500, 1000]
optimizers =["adam", "adamW", "sgd"]
random_seeds = [20, 200]

gamma = 0.99 #discount factor
batch_size = 64
embedding_dim = 256
margin = 0.2

episodes = 300

results = []
search_space = list(itertools.product(alphas, epsilon_decays, target_updates, optimizers, random_seeds))
total_runs = len(search_space)

for idx, (alpha, eps_decay, target_update, optimizer, random_seed) in enumerate(search_space, 1):
    print(f"""
        ============================== 
        Training [{idx}/{total_runs}]
        alpha={alpha}, epsilon_decay={eps_decay}, 
        target_update={target_update}, optimizer={optimizer}, random_seed={random_seed}
        ==============================""")

    agent, returns = train_agent(
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        batch_size=batch_size,
        epsilon_decay=eps_decay,
        target_update=target_update,
        optimizer=optimizer,
        embedding_dim=embedding_dim,
        margin=margin,
        early_stop_patience=100,          # ← tune this
        early_stop_threshold=-np.inf,     # ← or 0.0 if you expect > 0 return
        verbose=False
    )
    

    avg_return = np.mean(returns[-10:])  # Use last 10 episodes as performance metric

    results.append({
        "alpha": alpha,
        "gamma": gamma,
        "batch_size": batch_size,
        "epsilon_decay": eps_decay,
        "target_update":target_update,
        "optimizer":optimizer,
        "embedding_dim":embedding_dim,
        "margin":margin,
        "random_seed":random_seed,
        "avg_return": avg_return,
        "returns": returns
    })

# Sort and print best results
sorted_results = sorted(results, key=lambda x: x["avg_return"], reverse=True)

#print("\nTop 10 Hyperparameter Sets:")
for res in sorted_results[:10]:
    print(res)



# -------- PLOT TOP 5 LEARNING CURVES --------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

for i, res in enumerate(sorted_results[:5]):
    label = (f"{res['optimizer']}, lr={res['alpha']}, "
             f"eps_decay={res['epsilon_decay']}, random_seeds={res['random_seed']}")
    
    plt.plot(res["returns"], label=f"#{i+1}: {label}")

    window_size=100
    if len(res["returns"]) >= window_size:
        ma = np.convolve(
            res["returns"],
            np.ones(window_size) / window_size,
            mode="valid"
        )
        plt.plot(
            range(window_size - 1, len(res["returns"])),
            ma,
            label=f"#{i+1}: {window_size}-ep moving avg",
        )

plt.title("Top 5 Hyperparameter Setups - Learning Curves")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend(loc="upper left", fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

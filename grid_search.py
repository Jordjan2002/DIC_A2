import itertools
import numpy as np
import matplotlib.pyplot as plt
from train_dqn import train_agent

# Hyperparameter grid
#alphas = [1e-4, 5e-4, 1e-3] #learning rate
alphas = [0.001] #learning rate
gammas = [0.99] #discount factor
batch_sizes = [64]
#epsilon_decays = [0.95, 0.99, 0.995]
epsilon_decays = [0.998]
#target_updates = [100, 500, 1000]
target_updates = [1000]
#optimizers =["adam", "adamW", "sgd"]
optimizers =["adam"]
#embedding_dims = [128, 256, 512]
embedding_dims = [256]
#margins = [0.2, 0.5, 1.0]
margins = [0.2]

episodes = 500

results = []
search_space = list(itertools.product(alphas, gammas, batch_sizes, epsilon_decays, target_updates, optimizers, embedding_dims, margins))
total_runs = len(search_space)

for idx, (alpha, gamma, batch_size, eps_decay, target_update, optimizer, embedding_dim, margin) in enumerate(search_space, 1):
    print(f"""
        ============================== 
        Training [{idx}/{total_runs}]
        alpha={alpha}, gamma={gamma}, batch_size={batch_size}, epsilon_decay={eps_decay}, 
        target_update={target_update}, optimizer={optimizer}, embedding_dim={embedding_dim}, margin={margin}
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
        "avg_return": avg_return,
        "returns": returns
    })

# Sort and print best results
sorted_results = sorted(results, key=lambda x: x["avg_return"], reverse=True)

#print("\nTop 5 Hyperparameter Sets:")
#for res in sorted_results[:5]:
    #print(res)


# -------- PLOT TOP 5 LEARNING CURVES --------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

#for i, res in enumerate(sorted_results[:5]):
for i, res in enumerate(sorted_results[:1]):
    label = (f"{res['optimizer']}, lr={res['alpha']}, gamma={res['gamma']}, "
             f"bs={res['batch_size']}, eps_decay={res['epsilon_decay']}, "
             f"embed={res['embedding_dim']}, margin={res['margin']}")
    
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

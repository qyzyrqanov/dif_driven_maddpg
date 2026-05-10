import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnv
import tkinter as tk

from models.simpleactor import SimpleActor


class SharedActorWrapper:
    def __init__(self, actor, device):
        self.actor = actor
        self.device = device

    def choose_actions(self, obs, use_noise=False):
        """
        Args:
            obs (torch.Tensor): Tensor of shape [N, obs_dim]
        Returns:
            torch.Tensor: Tensor of shape [N, act_dim] on correct device
        """
        return self.actor.choose_action(obs.to(self.device), use_noise=use_noise, eval_mode=True)


def play(env:DiffDriveParallelEnv, agent:SharedActorWrapper, delay: float = 0.2):
    """
    Launch a Tkinter GUI embedding matplotlib to play with a multi-agent environment.

    Args:
        env: your DiffDriveParallelEnv environment
        agent: your agent (e.g., MADDPGSharedActorCritic) with choose_actions()
        delay: step delay in seconds
    """

    class EnvPlayer:
        def __init__(self, root):
            self.root = root
            self.running = False
            self.seed_var = tk.StringVar()
            self.state = None
            self.obs = None

            # --- Top Control Panel ---
            control_frame = tk.Frame(root)
            control_frame.pack()

            self.pause_btn = tk.Button(control_frame, text="▶ Run", width=10, command=self.toggle_run)
            self.pause_btn.grid(row=0, column=0)

            tk.Label(control_frame, text="Seed:").grid(row=0, column=1)
            self.seed_entry = tk.Entry(control_frame, textvariable=self.seed_var, width=10)
            self.seed_entry.grid(row=0, column=2)

            self.restart_btn = tk.Button(control_frame, text="⟲ Restart", width=10, command=self.restart)
            self.restart_btn.grid(row=0, column=3)

            # --- Agent Stats ---
            self.table = ttk.Treeview(root, columns=("reward", "action", "observation"), show='headings')
            self.table.heading("reward", text="Reward")
            self.table.heading("action", text="Action")
            self.table.heading("observation", text="Observation")
            self.table.pack(fill=tk.BOTH, expand=True)

            for agent_id in env.agents:
                self.table.insert("", "end", iid=agent_id, values=("", "", ""))

            # --- Matplotlib Canvas Embedding ---
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            env.fig = self.fig
            env.ax = self.ax
            self.canvas = FigureCanvasTkAgg(self.fig, master=root)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.restart()  # Reset initial state
            self.loop()

        def toggle_run(self):
            self.running = not self.running
            self.pause_btn.config(text="⏸ Pause" if self.running else "▶ Run")

        def restart(self):
            seed = self.seed_var.get()
            seed = int(seed) if seed.isdigit() else None
            self.state, self.obs = env.reset_tensor(seed=seed)
            self.running = False
            self.pause_btn.config(text="▶ Run")
            self.redraw()

        def redraw(self):
            self.ax.clear()
            env.render()  # uses self.ax
            self.canvas.draw()

        def loop(self):
            if self.running:
                with torch.no_grad():
                    actions = agent.choose_actions(self.obs, use_noise=False)
                    self.state, self.obs, rewards, dones = env.step_tensor(actions)

                self.redraw()

                # Update agent stats
                for i, agent_id in enumerate(env.agents):
                    rew = f"{rewards[i].item():.2f}"
                    act = ", ".join(f"{x:.2f}" for x in actions[i].tolist())
                    ob = ", ".join(f"{x:.2f}" for x in self.obs[i].tolist())
                    self.table.item(agent_id, values=(rew, act, ob))

            self.root.after(int(delay * 1000), self.loop)

    root = tk.Tk()
    root.title("Multi-Agent RL GUI Viewer")
    EnvPlayer(root)
    root.mainloop()

if __name__ == "__main__":
    env = DiffDriveParallelEnv()
    actor = SimpleActor(env.obs_dim, env.action_dim, device=env.device)
    actor.load_checkpoint()
    agent = SharedActorWrapper(actor, env.device)
    play(env, agent)

# RODENT SDK releases

Simulator packs and example notebooks for the **RODENT Python SDK** ([`rodent-sdk` on PyPI](https://pypi.org/project/rodent-sdk/)).

RODENT is a virtual rodent behaviour environment: researchers build arenas (open field, light/dark box, T-maze, mazes of their own), set lights, sounds, odours, doors and treatments, and run a simulated rat through them. The [RODENT website](https://rodent-tau.vercel.app/) is where projects and experiments live and where runs are replayed in 3D. The Python SDK runs the same simulation without graphics, from Google Colab or any Python, and can train your own PyTorch model to control the rat.

This repository is not the source code. It hosts the files the SDK and its users download:

| Folder or page | What it holds |
| --- | --- |
| [Releases](https://github.com/HuzaifaMajeed/rodent-sdk-releases/releases) | The **simulator pack** for each SDK version (`rodent-<version>.pck`), the compiled Godot simulation the SDK runs. |
| [`notebooks/`](notebooks) | Ready-to-run Colab notebooks and the example model file. |

## Quick start

Open a notebook in Google Colab, no installation needed:

| Notebook | What it does | Open |
| --- | --- | --- |
| **Getting started** | Sign in, choose an experiment, run it, look at the results and save the run to the website. | [Open in Colab](https://colab.research.google.com/github/HuzaifaMajeed/rodent-sdk-releases/blob/main/notebooks/getting_started.ipynb) |
| **Train your rat** | Train a PyTorch model to control the rat (reach a region, avoid light, follow an odour or a sound), look at any training episode, compare it with the built-in rat, and save and reuse the model. | [Open in Colab](https://colab.research.google.com/github/HuzaifaMajeed/rodent-sdk-releases/blob/main/notebooks/train_your_rat.ipynb) |

The easiest way in is from the website: open an experiment, click **Use in Python**, and copy the experiment code into the notebook.

Before the first run, add two Colab Secrets (key icon on the left): `RODENT_EMAIL` and `RODENT_PASSWORD`, your website login, with **Notebook access** switched on.

## Install it yourself

```bash
pip install rodent-sdk              # running experiments
pip install "rodent-sdk[train]"     # plus Gymnasium, NumPy, pandas and matplotlib for training
```

Python 3.9 or later on Linux, Windows or macOS. PyTorch is needed only for training (Colab already has it).

```python
import rodent_sdk as rodent

# Offline, no account needed
run = rodent.run("light_dark_box", steps=3000, seed=42)
print(run.region_table())

# With the website
client = rodent.login()
exp = client.project("My study").experiment("Baseline")
run = exp.run(steps=5000)
run.save()                      # appears under Review on the website
```

Train your own model (the full worked example is the **Train your rat** notebook):

```python
env = exp.training_env(max_steps=150, reward=my_reward, actions="discrete", log="training")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(1)      # forward

env.log.episode(522).plot()                                 # any episode, rebuilt exactly
project.save_model("Dark seeker", model, training=env.log)  # shared with the project
```

## The example model

[`notebooks/rat_model.py`](notebooks/rat_model.py) is the model the training notebook uses: a PPO actor-critic in PyTorch, commented throughout. The notebook downloads it next to itself. To train your own model with the same notebook, replace this file with one that defines `RatModel` with three methods:

| Method | What it does |
| --- | --- |
| `forward(observations)` | 31 observation values in, a score for each of the 5 moves out. |
| `act(observation, best=False)` | The move to make (0 stay, 1 forward, 2 left, 3 right, 4 run). RODENT also calls this when the model drives the rat in a run. |
| `learn(episodes)` | Improve from a batch of finished episodes. |

## How the simulator is downloaded and checked

The package on PyPI is small: it holds the Python code plus the address and SHA-256 fingerprint of its simulator pack. On first use the SDK:

1. downloads the Godot engine version that exported the pack, and the pack from this repository's release;
2. checks the pack's SHA-256 against the fingerprint built into the package, and refuses and deletes it if they differ;
3. caches both, so later runs start straight away.

Because each SDK version names one exact pack, the same experiment and seed give the same run on any computer, and a notebook pinned to a version (for example `pip install rodent-sdk==0.4.2`) keeps getting the same simulation.

## Versions

Packs are never changed or deleted once published. A new pack is published only when the simulation itself changes; releases that change only Python reuse the previous pack.

| SDK version | Simulator pack | Main changes |
| --- | --- | --- |
| 0.3.1 | v0.3.1 | First release on PyPI: the slim package, with the pack downloaded from here and checked. |
| 0.4.0 | v0.4.0 | Model training: a Gymnasium environment, episode logs with exact rebuilds, project models with versions, model-driven runs, stimulus heat maps, review links. Light travels in straight lines and stops at walls; sound and odour weaken through walls; odour spreads as a gas. |
| 0.4.1 | v0.4.0 (reused) | Training episodes can be saved to Review. |
| 0.4.2 | v0.4.0 (reused) | A model with an `act()` method chooses its own moves in runs, seeded so runs repeat. |
| 0.4.3 | v0.4.0 (reused) | See the [release notes](https://github.com/HuzaifaMajeed/rodent-sdk-releases/releases). |
| 0.4.4 | v0.4.4 | See the [release notes](https://github.com/HuzaifaMajeed/rodent-sdk-releases/releases). |

Versions before 0.3.1 were distributed from the project's own storage and remain available there for older notebooks.

## Documentation

- **Documentation site:** https://virtual-rodent-docs.pages.dev/ (user guide, Python SDK reference, training guide, the user manual and a cheat sheet as PDFs).
- **Website:** https://rodent-tau.vercel.app/
- **PyPI:** https://pypi.org/project/rodent-sdk/

## A note on what the results mean

RODENT's built-in rat follows a simple statistical model, and a model you train learns whatever its reward pays for in the arena it trained in. Neither is a validated model of real rat behaviour. Use RODENT to build, compare and reproduce virtual experiments, and report the model, reward and settings with any result.

## Licence and authors

The RODENT SDK is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Built for COMS3011A (Software Design) at the University of the Witwatersrand by Huzaifa Majeed, Ahmed Majeed, Muhammed Abed and Zidan Fajandar. The Python SDK, its release pipeline and model training were built by Huzaifa Majeed.

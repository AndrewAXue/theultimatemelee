# CS 486 Final Project

A reinforcement learning AI trained on Super Smash Bros Melee.

# Installation

1. Install Dolphin / Slippi:
Slippi is a version of Dolphin built specifically for machine learning purposes. You can download it [here](https://slippi.gg/netplay)
2. Install Custom Dolphin Fork:
To train this model, you'll need our [custom fork of the libmelee](https://github.com/AndrewAXue/libmelee) 
(credits to the [original repo](https://github.com/altf4/libmelee)). Our fork adds the ability to choose enemy CPU difficulty.
    * Clone the libmelee fork
    * Follow the instructions outlined in the README of that repository to get Dolphin set up correctly.
    * Install this fork of Dolphin/Libmelee by running `pip install -e .`
3. Install pip requirements
Using the requirements file in this repository, run `pip install requirements.txt`

# To Train Using Deep Q-Networks
Run `python3 DQNTrainer.py`

# To Train Using Policy Gradients
run `python3 Policy.py`

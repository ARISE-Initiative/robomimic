# API for Multi-Task Evaluation
---
We need methods for summarizing performance of multi-task agents.  The hope is to achieve accurate checkpoint selection with a limited computation budget (evaluating on all tasks might take too long).  In order to build this system out, we need to think about how it will be used.

## Current API for Evaluation:
``train_utils.rollout_with_stats(policy, envs, use_goals, horizon, num_episodes)``
* **policy** - trained multi-task policy
* envs -  specified in config.experiment.additional_envs
* use_goals - If true, goal is provided by env.get_goal()
* horizon and num_episodes are the same for all environments
* Evaluation takes timesteps = len(envs) * num_episodes * horizon

Others: \
``run_trained_agent.py --agent --env --horizon ``\
validation loss (is there any other useful metric to report?)


## Proposed API:
Rollout with stats is largely unchanged, but we need a different goal, horizon, and num_episodes specified for each task.  That is to say, len(envs) = len(goals) = len(horizons)...\

```train_utils.rollout_with_stats(policy, envs, goals, horizons, num_episodes)```
* **policy** - trained multi-task policy
* **envs** - the environmet used for task evaluation
* **goals** - the goal for each task
* **num_episodes** - number of rollouts for each task

New function: \
``get_evaluation_tasks(dataset, eval_description, total_episodes)``- returns (envs, goals, horizons, num_episodes) which should be fed into train_utils.rollout_with_stats.
* **dataset** - this is the dataset used for training the policy, these are all the tasks that the model was trained on
* **evaluation discription** - this is the most ambiguous part of the document.  Open to suggestions, but I think this is how users should specify their desired evaluation method.
    * **default** - run the main evaluation method that we propose in the paper.  Need to have a larger discussion about what this will actually be.  If we are not in the buisness of task design, then this should select a representative subset of tasks from the training dataset.  See Vaibhav's post in slack that goes over a couple different ways that we could break these tasks into different categories.
    * **random** - randomly select a subset of tasks from the training dataset
    * **all** - use all tasks from the training dataset
    * **custom** - user explicitly specifies the tasks and their frequency
    * **language discription** - the user provides a language discription of what they want their agent to do.  We return appropriate evaluation tasks.
* **total_episodes** - total number of evaluation episodes across all of the tasks

## Current HDF5 Structure:
HDF5 API:
- File 
    - data (.attrs includes env_args, same for all of the demos)
        - demo_0
        - demo_1

## Proposed HDF5 Structure:
I think the HDF5 structure need to be reformulated to emphasize tasks.  Here is one proposed method.  I believe a task can be uniquely described given an environment and a goal.  If we are going to break tasks into categories, those decisions should be made based on the information held in the .attrs fields.  
HDF5 API:
- File
    - env_0 (.attrs includes env_args)
        - goal_0 (.attrs includes image, text, and low dim goal specification)
            - demo_0
            - demo_1
        - goal_1
            - demo_0
            - demo_1
    - env_1
    - env_2

<!-- ## General Comments
Evaluating Multiple Tasks:
* One of the main contributions for the paper should be an in depth benchmark
* Default Multi-Task Evaluation
    * Run our benchmark
* Random
    * Randomly choose environment and goal for each rollout
* Custom
    * Specify which environments and goals to use as well as their frequency

Questions
* How do we want to categorize the environemnts
* Do we want to split our environment and goals into categories?
    * Difficulty Level - Easy, Medium, Hard
    * Skill Level
        * Picking (block lift)
        * Insertion (square)
        * Manipulation (mug flipping)
        * Short/Long Horizon for all of these
 -->

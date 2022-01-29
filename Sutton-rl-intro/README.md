## reinforcement-learning-an-introduction

A**.ipynb files demonstrate how to use **Algorithms** to tackle problems like gambler and herman

c**.py files are the code correspond to each **chapter**.  

c00_* is the basic structure code for this project

### Convention

1. Env has two methods, in which state is an **int** start from 0, action also is an **int** start from 0. Env class wraps all Environment specific implements.
 - get_all_state() -> List[int]
 - get_all_state_action() -> Dict[int, List[int]]

2. The step method in Env receive action_idx parameter, which is the index of the `List[int]` in `Dict[int, List[int]]`
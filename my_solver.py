# -*- coding: utf-8 -*-
"""
Created on  Feb 27 2018

@author: frederic

Scaffholding code for CAB320 Assignment One

This is the only file that you have to modify and submit for the assignment.

"""

import numpy as np

import itertools

import generic_search

from assignment_one import (TetrisPart, AssemblyProblem, offset_range,
    #                            display_state,
                            make_state_canonical, play_solution,
    #                            load_state, make_random_state
                            )


# ---------------------------------------------------------------------------

def print_the_team():
    '''
    Print details of the members of your team 
    (full name + student number)
    '''
    print('Manuel Concepcion, 10156208')
    print('Petr Ungar, ')


#    print('Ada Lovelace, 12340001')
#    print('Grace Hopper, 12340002')
#    print('Maryam Mirzakhani, 12340003')

# ---------------------------------------------------------------------------

def appear_as_subpart(some_part, goal_part):
    '''    
    Determine whether the part 'some_part' appears in another part 'goal_part'.
    
    Formally, we say that 'some_part' appears in another part 'goal_part',
    when the matrix representation 'S' of 'some_part' is a a submatrix 'M' of
    the matrix representation 'G' of 'goal_part' and the following constraints
    are satisfied:
        for all indices i,j
            S[i,j] == 0 or S[i,j] == M[i,j]
            
    During an assembly sequence that does not use rotations, any part present 
    on the workbench has to appear somewhere in a goal part!
    
    @param
        some_part: a tuple representation of a tetris part
        goal_part: a tuple representation of another tetris part
        
    @return
        True if 'some_part' appears in 'goal_part'
        False otherwise    
    '''
    # Step 1: Find the location of the first element of some_part in goal_part. This is stored in idx (numpy array)
    # Step 2: Try to match some_part in goal_part (only looking at the nonzero elements of some_part).
    part = np.array(some_part)  # HINT
    goal = np.array(goal_part)
    # Check if the array has just a single element.
    if part.size == 1:
        # Check if the element is in goal.
        if part[0] in goal:
            return True
        else:
            return False
    else:
        idx = np.where(goal == part[0][0])
        for i in range(0, idx[0].size):
            try:
                goal_subarray = goal[idx[0][i]:part.shape[0] + idx[0][i], idx[1][i]:part.shape[1] + idx[1][i]]
                if np.array_equal(part[part.nonzero()], goal_subarray[part.nonzero()]):
                    return True
            except IndexError:
                pass
    return False


# ---------------------------------------------------------------------------

def cost_rotated_subpart(some_part, goal_part):
    '''    
    Determine whether the part 'some_part' appears in another part 'goal_part'
    as a rotated subpart. If yes, return the number of 'rotate90' needed, if 
    no return 'np.inf'
    
    The definition of appearance is the same as in the function 
    'appear_as_subpart'.
                   
    @param
        some_part: a tuple representation of a tetris part
        goal_part: a tuple representation of another tetris part
    
    @return
        the number of rotation needed to see 'some_part' appear in 'goal_part'
        np.inf  if no rotated version of 'some_part' appear in 'goal_part'
    
    '''
    # Create a TetrisPart to rotate some_part
    some_part_tetris = TetrisPart(some_part)
    # Check if the array has just a single element.
    if np.array(some_part).size == 1:
        # Check if the element is in goal.
        if some_part[0] in goal_part:
            return 0
        else:
            return np.inf
    # Check the four possible rotations
    possible_rotations = 4
    for rot in range(0, possible_rotations):
        if rot == 0:
            some_part_tetris.get_frozen()
            if appear_as_subpart(some_part_tetris.frozen, goal_part):
                return rot
        else:
            some_part_tetris.rotate90()
            some_part_tetris.get_frozen()
            if appear_as_subpart(some_part_tetris.frozen, goal_part):
                return rot
    return np.inf


# ---------------------------------------------------------------------------

class AssemblyProblem_1(AssemblyProblem):
    '''
    
    Subclass of 'assignment_one.AssemblyProblem'
    
    * The part rotation action is not available for AssemblyProblem_1 *

    The 'actions' method of this class simply generates
    the list of all legal actions. The 'actions' method of this class does 
    *NOT* filtered out actions that are doomed to fail. In other words, 
    no pruning is done in the 'actions' method of this class.
        
    '''

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        # Call the parent class constructor.
        # Here the parent class is 'AssemblyProblem' 
        # which itself is derived from 'generic_search.Problem'
        super(AssemblyProblem_1, self).__init__(initial, goal, use_rotation=False)

    def actions(self, state):
        """
        Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        
        @param
          state : a state of an assembly problem.
        
        @return 
           the list of all legal drop actions available in the 
            state passed as argument.
        
        """
        # Check if all elements from state are equal
        # In case yes use itertools.combinations to avoid repetitions.
        if state[1:] == state[:-1]:
            valid_moves = [(a, b, c) for a, b in itertools.combinations(state, 2) for c in
                            range(offset_range(a, b)[0], offset_range(a, b)[1])
                            if c is not None]
        else:
            valid_moves = [(a, b, c) for a, b in itertools.permutations(state, 2) for c in
                            range(offset_range(a, b)[0], offset_range(a, b)[1])
                            if c is not None]
        return valid_moves
    def result(self, state, action):
        """
        Return the state (as a tuple of parts in canonical order)
        that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
        
        @return
          a state in canonical order
        """
        assert action in self.actions(state)
        pa, pu, offset = action
        # Rename variables with meaningful names
        tetris = TetrisPart(part_above=pa, part_under=pu, offset=offset)
        new_part = tetris.get_frozen()
        tetris.frozen = None
        updated_state = [a for a in state if a not in [pa, pu]]
        updated_state.append(new_part)
        return make_state_canonical(tuple(updated_state))


# ---------------------------------------------------------------------------

class AssemblyProblem_2(AssemblyProblem_1):
    '''
    
    Subclass of 'assignment_one.AssemblyProblem'
        
    * Like for AssemblyProblem_1,  the part rotation action is not available 
       for AssemblyProblem_2 *

    The 'actions' method of this class  generates a list of legal actions. 
    But pruning is performed by detecting some doomed actions and 
    filtering them out.  That is, some actions that are doomed to 
    fail are not returned. In this class, pruning is performed while 
    generating the legal actions.
    However, if an action 'a' is not doomed to fail, it has to be returned. 
    In other words, if there exists a sequence of actions solution starting 
    with 'a', then 'a' has to be returned.
        
    '''

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        # Call the parent class constructor.
        # Here the parent class is 'AssemblyProblem' 
        # which itself is derived from 'generic_search.Problem'
        super(AssemblyProblem_2, self).__init__(initial, goal)

    def actions(self, state):
        """
        Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        
        A candidate action is eliminated if and only if the new part 
        it creates does not appear in the goal state.
        """

        valid_moves = [(a, b, c) for a, b in itertools.permutations(state, 2) for c in
                       range((offset_range(a, b)[0]), (offset_range(a, b)[1]))
                       if c is not None and
                       appear_as_subpart(TetrisPart(part_under=b, part_above=a, offset=c).get_frozen(), self.goal[0])]
        return valid_moves


# ---------------------------------------------------------------------------

class AssemblyProblem_3(AssemblyProblem_1):
    '''
    
    Subclass 'assignment_one.AssemblyProblem'
    
    * The part rotation action is available for AssemblyProblem_3 *

    The 'actions' method of this class simply generates
    the list of all legal actions including rotation. 
    The 'actions' method of this class does 
    *NOT* filter out actions that are doomed to fail. In other words, 
    no pruning is done in this method.
        
    '''

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        # Call the parent class constructor.
        # Here the parent class is 'AssemblyProblem' 
        # which itself is derived from 'generic_search.Problem'
        super(AssemblyProblem_3, self).__init__(initial, goal)
        self.use_rotation = True
        self.magic_num = -47 # Identifier that an action is a rotation

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        
        Rotations are allowed, but no filtering out the actions that 
        lead to doomed states.

        The actions that lead to rotation are defined as a tuple of the form:
        action = (rotated(piece),index(piece),magic_num)
        """
        # First the drop of one piece into other
        valid_moves1 = AssemblyProblem_1.actions(self, state)
        # Rotation of one of the pieces of state
        valid_moves2 = []
        # Check if all elements are equal. If all pieces are the same it does not matter which one we rotate
        are_equal = False
        if state[1:] == state[:-1]:
            are_equal = True
        for i in range(0, len(state)):
            tetris_piece = TetrisPart(state[i])
            tetris_piece.rotate90()
            piece = tetris_piece.get_frozen()
            valid_moves2.append((piece, i, self.magic_num))
            if are_equal:
                break
        valid_moves = valid_moves1 + valid_moves2

        return valid_moves

    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        The action can be a drop or rotation.        
        """
        # Here a workbench state is a frozenset of parts
        assert action in self.actions(state)
        if self.magic_num in action:
            rotated_piece = action[0]
            idx_rot = action[1] # Index of rotated piece
            updated_state = [state[i] if i != idx_rot else rotated_piece for i in range(0, len(state))]
            return make_state_canonical(updated_state)
        else:
            return AssemblyProblem_1.result(self, state, action)



# ---------------------------------------------------------------------------

class AssemblyProblem_4(AssemblyProblem_3):
    '''
    
    Subclass 'assignment_one.AssemblyProblem3'
    
    * Like for its parent class AssemblyProblem_3, 
      the part rotation action is available for AssemblyProblem_4  *

    AssemblyProblem_4 introduces a simple heuristic function and uses
    action filtering.
    See the details in the methods 'self.actions()' and 'self.h()'.
    
    '''

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        # Call the parent class constructor.
        # Here the parent class is 'AssemblyProblem' 
        # which itself is derived from 'generic_search.Problem'
        super(AssemblyProblem_4, self).__init__(initial, goal)

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        
        Rotations are allowed, but no filtering out the actions that 
        lead to doomed states.

        The actions that lead to rotation are defined as a tuple of the form:
        action = (rotated(piece),index(piece),magic_num)
        """
        # First the drop of one piece into other
        valid_moves1 = AssemblyProblem_2.actions(self, state)
        # Rotation of one of the pieces of state
        valid_moves2 = []
        # Check if all elements are equal. If all pieces are the same it does not matter which one we rotate
        are_equal = False
        if state[1:] == state[:-1]:
            are_equal = True
        for i in range(0, len(state)):
            tetris_piece = TetrisPart(state[i])
            tetris_piece.rotate90()
            piece = tetris_piece.get_frozen()
            valid_moves2.append((piece, i, self.magic_num))
            if are_equal:
                break
        valid_moves = valid_moves1 + valid_moves2

        return valid_moves

    def h(self, n):
        '''
        This heuristic computes the following cost; 
        
           Let 'k_n' be the number of parts of the state associated to node 'n'
           and 'k_g' be the number of parts of the goal state.
          
        The cost function h(n) must return 
            k_n - k_g + max ("cost of the rotations")  
        where the list of cost of the rotations is computed over the parts in 
        the state 'n.state' according to 'cost_rotated_subpart'.
        
        
        @param
          n : node of a search tree
          
        '''

        k_n = len(n.state)
        k_g = len(self.goal)
        rot_cost = []

        for rotation in n.state:
            rot_cost.append(cost_rotated_subpart(rotation, self.goal[0]))

        return k_n - k_g + max(rot_cost)
    
    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        The action can be a drop or rotation.        
        """
        # Here a workbench state is a frozenset of parts
        assert action in self.actions(state)
        if self.magic_num in action:
            rotated_piece = action[0]
            idx_rot = action[1] # Index of rotated piece
            updated_state = [state[i] if i != idx_rot else rotated_piece for i in range(0, len(state))]
            return make_state_canonical(updated_state)
        else:
            return AssemblyProblem_2.result(self, state, action)


# ---------------------------------------------------------------------------

def solve_1(initial, goal):
    '''
    Solve a problem of type AssemblyProblem_1
    
    The implementation has to 
    - use an instance of the class AssemblyProblem_1
    - make a call to an appropriate functions of the 'generic_search" library
    
    @return
        - the string 'no solution' if the problem is not solvable
        - otherwise return the sequence of actions to go from state
        'initial' to state 'goal'
    
    '''

    print('\n++  busy searching in solve_1() ...  ++\n')

    assembly_problem = AssemblyProblem_1(initial, goal)  # HINT
    sol_ts = generic_search.breadth_first_tree_search(assembly_problem)
    if sol_ts is None:
        return ('no solution')
    else:
        return sol_ts.solution()


# ---------------------------------------------------------------------------

def solve_2(initial, goal):
    '''
    Solve a problem of type AssemblyProblem_2
    
    The implementation has to 
    - use an instance of the class AssemblyProblem_2
    - make a call to an appropriate functions of the 'generic_search" library
    
    @return
        - the string 'no solution' if the problem is not solvable
        - otherwise return the sequence of actions to go from state
        'initial' to state 'goal'
    
    '''

    print('\n++  busy searching in solve_2() ...  ++\n')
    assembly_problem = AssemblyProblem_2(initial, goal)  # HINT
    sol_ts = generic_search.breadth_first_tree_search(assembly_problem)
    if sol_ts is None:
        return ('no solution')
    else:
        return sol_ts.solution()


# ---------------------------------------------------------------------------

def solve_3(initial, goal):
    '''
    Solve a problem of type AssemblyProblem_3
    
    The implementation has to 
    - use an instance of the class AssemblyProblem_3
    - make a call to an appropriate functions of the 'generic_search" library
    
    @return
        - the string 'no solution' if the problem is not solvable
        - otherwise return the sequence of actions to go from state
        'initial' to state 'goal'
    
    '''
    print('\n++  busy searching in solve_3() ...  ++\n')
    assembly_problem = AssemblyProblem_3(initial, goal)  # HINT
    sol_ts = generic_search.breadth_first_graph_search(assembly_problem)
    if sol_ts is None:
        return ('no solution')
    else:
        return sol_ts.solution()



# ---------------------------------------------------------------------------

def solve_4(initial, goal):
    '''
    Solve a problem of type AssemblyProblem_4
    
    The implementation has to 
    - use an instance of the class AssemblyProblem_4
    - make a call to an appropriate functions of the 'generic_search" library
    
    @return
        - the string 'no solution' if the problem is not solvable
        - otherwise return the sequence of actions to go from state
        'initial' to state 'goal'
    
    '''

    #         raise NotImplementedError
    print('\n++  busy searching in solve_4() ...  ++\n')
    assembly_problem = AssemblyProblem_4(initial, goal)  # HINT
    sol_ts = generic_search.astar_graph_search(assembly_problem)
    if sol_ts is None:
        return ('no solution')
    else:
        return sol_ts.solution()


# ---------------------------------------------------------------------------



if __name__ == '__main__':
    pass

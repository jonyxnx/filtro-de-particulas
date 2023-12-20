import numpy as np
import random
from matplotlib import pyplot as plt 
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

#FUNCTIONS FOR THE PARTICLE FILTER

def brownian_motion(dt,N, grid):

    '''
    Function to simulate Brownian motion.

    PARAMETERS:
        dt (float): step size
        N (int): number of steps
        grid (np.array): the maze in which we move

    RETURNS:
        d (np.array): the trajectory of the movement
        dx (np.array): the derivative along the x-axis of the trajectory
        dy (np.array): the derivative along the y-axis of the trajectory
    '''

    n = len(grid)
    start_position = [np.random.uniform(0, n), np.random.uniform(0,n)]

    dx = np.random.normal(0,np.sqrt(dt),N)
    dy = np.random.normal(0,np.sqrt(dt),N)
    
    d1 = start_position[0]
    d2 = start_position[1]

    d = [[d1,d2]]

    for i in range(0,N):
        xa = d[-1][0]
        ya = d[-1][1]

        d.append(rebote_maze(grid, xa, ya, dx[i], dy[i]))
    
    return d, dx, dy


def brownian_motion_filter(dt,N, grid, m, graficar = False, ruido = 1, stop = False):
    
    '''
    Function to simulate random movement and apply the filter simultaneously.

    PARAMETERS:
        dt (float): step size
        N (int): number of steps
        grid (np.array): the maze in which we move
        m (int): number of particles
        noise (float): amount of noise added to the observed signal
        stop (bool): if True, the filter stops when it gets very close to the real position

    RETURNS:
        d (np.array): the trajectory of the movement
        dx (np.array): the derivative along the x-axis of the trajectory
        dy (np.array): the derivative along the y-axis of the trajectory
        mean (list): the estimated process
        obs (list): observations with noise generated by the trajectory
    '''
        
    n = len(grid)

    par = [[np.random.uniform(0, n), np.random.uniform(0,n)] for i in range(m)]

    start_position = [np.random.uniform(2, n-2), np.random.uniform(2,n-2)]

    dx = np.random.normal(0,np.sqrt(dt),N)
    dy = np.random.normal(0,np.sqrt(dt),N)
    
    dx = np.clip(dx, a_min = -.5, a_max = .5)
    dy = np.clip(dy, a_min = -.5, a_max = .5)

    par = np.array(par)

    d1 = start_position[0]
    d2 = start_position[1]

    d = [[d1,d2]]

    estimado = []
    observaciones = []
    for i in range(0,N):

    #Particle update:
        #Calculate weughts:
        pesos = []
        observacion = [d[-1][0]+np.random.normal(0,ruido), d[-1][1]+np.random.normal(0,ruido)]
        observaciones.append(observacion)

        for j in range(m): 
            w = 1/(1+np.exp((par[j][0]-observacion[0])**2+(par[j][1]-observacion[1])**2))
            pesos.append(w)
        
        pesos = pesos/sum(pesos)    
        
        par = remuestreo(pesos, par)
        
        par = par + np.random.normal(0,.1, par.shape)

        par = np.clip(par, a_min = [0,0], a_max = [n,n])
        
        xa = d[-1][0]
        ya = d[-1][1]

        d.append(rebote_maze(grid, xa, ya, dx[i], dy[i]))

        if i%(N/10) == 0 and graficar == True:
            draw_maze(grid, create = False, particles = par, trajectory = d)
        e = [np.mean([j[0] for j in par]), np.mean([j[1] for j in par])]

        estimado.append(e)
        if np.linalg.norm(np.array(e)-np.array([xa, ya])) < .05 and stop == True:
            break

    return d, dx, dy, estimado, observaciones, i

def particle_filter(Y, W, N, f):

    '''     
    Particle Filter

    PARAMETERS:
        Y (list): list with observations
        W (function): calculates weights
        N (int): Number of particles
        f (function): initial distribution of particles

    RETURNS:
        X (list): estimated density
    '''

    X = f()

    for i in range(len(Y)):
        #Calculamos los pesos
        pesos = W(Y[i], X)
        #Hacemos el remuestreo
        X = remuestreo(W, X)
    return X

def remuestreo(pesos, X):

    '''
    Function for resampling
    
    PARAMETERS:
        pesos (list): list with weights
        X (list): particles
    RETURNS:
        X (list): list with new particles
    '''

    n = len(pesos)
    particulas_nuevas = []
    pesos = [0]+[sum(pesos[:i+1]) for i in range(n)]
    u0, s = np.random.uniform(), 0

    for j in [(u0+i)/n for i in range(n)]:
        while j > pesos[s]:
            s+=1
        particulas_nuevas.append(X[s-1])

    return np.array(particulas_nuevas)  


#FUNCTIONS FOR THE MAZE

def up(cell):

    '''
    
    Move to the above position

    PARAMETERS:
        cell  (tuple): current cell 

    RETURNS:
        (tuple): the above position
    '''

    return cell[0], cell[1] + 1

def down(cell):

    '''
    Move to the lower position.

    PARAMETERS:
        cell (tuple): current cell

    RETURNS:
        (tuple): The cell below the current cell
    '''

    return cell[0], cell[1] - 1

def right(cell):

    '''
    Move to the right position.

    PARAMETERS:
        cell (tuple): current cell

    RETURNS:
        (tuple): The cell to the right of the current cell
    '''
    return cell[0] + 1, cell[1]

def left(cell):

    '''
    Move to the left position.

    PARAMETERS:
        cell (tuple): current cell

    RETURNS:
        (tuple): The cell to the left of the current cell
    '''

    return cell[0] - 1, cell[1]

def get_random_path(start_cell: (int, int), unvisited: np.array, n) -> [(int, int)]:
    
    '''
    Generate a random walk.

    PARAMETERS:
        start_cell (tuple): cell where the walk begins
        unvisited (np.array): list of cells that have not been visited and can be traversed by the walk
        n (int): size of the maze being worked on

    RETURNS:
        path (list): sequence of cells on which the walk takes place
    '''

    path = [start_cell]

    while True:
        #Check all cells to which we can move
        possible_cells = possible_moves(n, start_cell)
        
        #Take one of those cells
        start_cell = random.choice(possible_cells)

        #If we havent visited it, we save the path and return the path
        if start_cell not in unvisited:
            path.append(start_cell)
            return path
        
        #If we already visited it, we restart the walk
        elif start_cell in path:
            path = path[:path.index(start_cell) + 1]

        #Otherwise, save it on the walk
        else:
            path.append(start_cell)

def link_cells(grid, cell1, cell2):

    '''
    This function modifies the matrix where the information of the maze is stored to save the generated paths.

    PARAMETERS:
        grid (np.array): a matrix of nxn with tuples of size 2, where the information of the maze is stored
        cell1 (tuple): the cell from which it starts
        cell2 (tuple): the cell to which it moves
    '''

    #Get the information of the cell
    i = cell1[0]
    j = cell1[1]
    
    k = cell2[0]
    l = cell2[1]
    

     # Removing the walls that have been passed through
    if i == k:
        if j == l+1:
            grid[k,l][1] = True
        else:
            grid[i,j][1] = True
    else:
        if i == k+1:
            grid[k,l][0] = True
        else:
            grid[i,j][0] = True 

def possible_moves(n, cell: (int, int)):

    '''
    This function checks that the movement does not take us out of the grid
    
    Parameters:
        n (int): size of the maze grid
        cell (tuple): position in the maze
    
    Returns:
        moves (list): a list of possible cells to move to
    '''

    #Make a list to save the possible moves
    moves = []


    for direction in [up, down, right, left]:
        new_cell = direction(cell)
         #If the robot does not go out of the grid by making the movement, we accept the move
        if 0 <= new_cell[0] < n and 0 <= new_cell[1] < n:
            moves.append(new_cell)

    #Return the possible moves
    return moves

def can_go_up(cell,maze):

    '''

    This function verifies if you can go up

    PARAMETERS:
        cell (tuple): current position
        maze (np.array): n x n matrix where the information of the maze is stored
    RETURNS:
        (bool): if there is a wall return FALSE
    '''


    i = cell[0]
    j = cell[1]
    return maze[i, j][1]

def can_go_left(cell, maze):

    '''

    This function verifies if you can go left

    PARAMETERS:
        cell (tuple): current position
        maze (np.array): n x n matrix where the information of the maze is stored
    RETURNS:
        (bool): if there is a wall return FALSE
    '''

    i, j = left(cell)
    return maze[i, j][0] 

def can_go_right(cell, maze):

    '''

    This function verifies if you can go right

    PARAMETERS:
        cell (tuple): current position
        maze (np.array): n x n matrix where the information of the maze is stored
    RETURNS:
        (bool): if there is a wall return FALSE
    '''


    i = cell[0]
    j = cell[1]
    return maze[i,j][0]

def can_go_down(cell, maze):

    
    '''

    This function verifies if you can go down

    PARAMETERS:
        cell (tuple): current position
        maze (np.array): n x n matrix where the information of the maze is stored
    RETURNS:
        (bool): if there is a wall return FALSE
    '''

    i, j = down(cell)
    return maze[i, j][1]

def generate_maze(n, draw = False):
    
    '''
    Function to create a random maze.

    PARAMETERS:
        n (int): the size we want the maze to have
        draw (bool): TRUE if we want to draw the maze
    RETURNS:
        grid (np.array): an array that stores the maze information
    '''


    #Make an n x n matrix
    grid = np.zeros((n, n, 2), bool)

    #Make a list with unvisited cells
    unvisited_indices = [x for x in np.ndindex((n, n))]
    
    #Shuffle th unvisited locations
    random.shuffle(unvisited_indices)

    #Remove one element on the shuffled list
    unvisited_indices.pop()

    #Make a loop to ensure we visit all the cells
    while len(unvisited_indices) > 0:
        
        #Take the first cell
        cell = unvisited_indices[0]

        #Make the random walk
        path = get_random_path(cell, unvisited_indices, n)

        
        #Save the information on the matrix
        for i in range(len(path) - 1):
            link_cells(grid, path[i], path[i + 1])

            unvisited_indices.remove(path[i])

    #Option to visually display the created maze
    if draw == True:
        draw_maze(grid, None, False)
    return grid

def draw_maze(grid, robot = None, create = True, particles = [], path = [],  trajectory = [], estimate = [], observacion = []):
    

    '''
    Create a function to help visualize the maze:

    PARAMETERS:
        grid (matrix): helps define the maze
        robot (tuple): represents the position of the robot
        create (bool): True if we want to create a robot at a random position
        particles (list): particles we want to draw
        path (list): discrete path we want to draw through maze cells
        trajectory (list): continuous trajectory we want to plot
        estimate (list): continuous trajectory we want to draw
        observation (list): list of objects we observe

    RETURNS:
        (cell): returns the position of the robot in case it was created
    '''

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    width = grid.shape[0]
    height = grid.shape[1]
    
    #Plot the edges of the maze
    plt.plot([0, width], [0, 0], color="black")
    plt.plot([0, 0], [0, height], color="black")
    plt.plot([0, width], [height, height], color="black")
    plt.plot([width, width], [0, height], color="black")
    
    #Plot the maze with the information on the matrix
    for j in range(height):
        for i in range(width):
            value = grid[i, j]
            if not value[0]:
                
                plt.plot([i + 1, i + 1], [j, j + 1], color="black")

            if not value[1]:
                
                plt.plot([i, i + 1], [j + 1, j + 1], color="black")
    
    #If path is not empty, we plot it
    for k in path:
        i, j = k
        rect = Rectangle((i, j), 1, 1, linewidth=1, edgecolor='none', facecolor='yellow')
        ax.add_patch(rect)
    
    #Create a random robot
    if create == True:
        n = len(grid)
        i = random.randint(1, n-1)
        j = random.randint(1, n-1)
        robot = (i,j)
    
    #If there is a robot, we plot it
    if  robot != None:
        i, j = robot
        circulo = Circle((i+.5, j+.5), .5, linewidth=1, edgecolor='none', facecolor='red')
        ax.add_patch(circulo)
        
    #Add the particles 
    for elemento in particles:
        i, j = elemento
        circulo = Circle((i, j), .1, linewidth=.2, edgecolor='none', facecolor='blue')
        ax.add_patch(circulo)
    
    if len(trajectory) != 0:
        plt.plot([t[0] for t in trajectory], [t[1] for t in trajectory], color = "green", linewidth = 1, label = "Real Trajectory")

    if len(estimate)!=0:
        plt.plot([j[0] for j in estimate],[j[1] for j in estimate], label = "Estimate") 

    if len(observacion) != 0:
        plt.scatter([t[0] for t in observacion], [t[1] for t in observacion], s= .4, color = "red", linewidth = .8, label = "Signal")
    
    if len(observacion) != 0 or len(estimate)!=0 or len(trajectory) != 0:
        plt.legend(loc='lower right')
        
    plt.show()
    return robot

def rebote(tx, ty, dx,dy, l = False, u = False, d = False , r = False):

    '''
    This function calculates the rebound when simulated movement collides with a maze.

    PARAMETERS:
        tx (float): current position on the x-axis
        ty (float): current position on the y-axis
        dx (float): rate of change on the x-axis
        dy (float): rate of change on the y-axis
        l (bool): Indicates if it is possible to move to the left of the cell
        u (bool): Indicates if it is possible to move upward from the cell
        d (bool): Indicates if it is possible to move downward from the cell
        r (bool): Indicates if it is possible to move to the right of the cell

    RETURNS:
        (list): The rebound position
    ''' 


    #calculate the slope
    pendiente = dy / dx
    
    #Round it to avoid bugs
    if abs(pendiente) > 1000 or abs(pendiente) < .0001:
        return rebote(tx, ty, .001, .001, l, u, d, r)
    
    else:
        x = tx+dx 
        y = ty+dy

    #Calculate the y-intercept
    b = ty-pendiente*tx
    
    #Calculate the bound depending where the impact is
    if 0 >= x and 0 >= y:
        if b < 0:
            return [-x,-y]
        else:
            return [-x,-y]

    if 0 >= x and 1 <= y:
        if b < 1:
            return [-x,2-y]
        else:
            return [-x,2-y]

    if 0 >= y and 1 <= x:
        if pendiente+b < 0:
            return [2-x,-y]
        else:
            return [2-x,-y]
        
    if 1 <= y and 1 <= x:
        if pendiente+b < 1:
            return [2-x,2-y]
        else:
            return [2-x, 2-y]
        
    if 0 >= x and (not l):
        return [-x, y]
    
    if 0 >= y and not d:
        return [x, -y]
    
    if 1 <= x and (not r):
        return [2-x,y]
    
    if 1 <= y and not u:
        return [x, 2-y]
    
    return [tx+dx,ty+dy]   


def rebote_maze(maze, tx, ty, difx,dify):


    '''
    This function calculates the position after bouncing off a maze wall.

    PARAMETERS:
        maze (np.array): stores information about the maze
        tx (float): current position on the x-axis
        ty (float): current position on the y-axis
        difx (float): sensitive of change on the x-axis
        dify (float): sensitive of change on the y-axis

    RETURNS:
        D (list): The position after bouncing off the wall
'''

    values = [can_go_left, can_go_up, can_go_down, can_go_right]
    
    #Calculate the current cell
    posx = np.floor(tx) 
    posy = np.floor(ty)
    

    tx = tx-posx
    ty = ty-posy

    cell = (int(posx), int(posy))
    
    pared = []

    #Verfies if there is a wall around
    for k in values:
        pared.append(k(cell, maze))

    izquierda = pared[0] 
    arriba = pared[1]
    abajo = pared[2]
    derecha = pared[3]

    #Aplly rebote function
    C = rebote(tx, ty, difx, dify, izquierda, arriba, abajo, derecha)
    
    D = [C[0]+ posx, C[1]+posy]
    return D

def solve(maze, particle):
    #Place the functions to check if it can move and the movement functions
    values = [can_go_left, can_go_up, can_go_down, can_go_right]
    direction = [left, up, down, right]

    #Create a queue with the particle
    Q = [particle]

    #Create a list to save the cells we have already passed through
    n = len(maze)
    explored = np.zeros((n,n), bool)

    #Initialize a variable to calculate the distance to the center and to signal when we find the exit
    distancia = 0
    find = False

    #Label the first particle as explored
    explored[particle] = True
    
    #Create a dictionary to store the information 
    rows, cols, _ = maze.shape
    parents = {cell: None for cell in np.ndindex((rows,cols))}  

    #Start a loop that end when we reach the exit
    while len(Q) > 0 and find == False:

        #Make a loop for every elemente in the query
        for i in range(len(Q)):
            
            #Pich the first elemente on the query
            v = Q[0]
            Q.pop(0)

            #Visit all the neighborghs, if we found the exit we break the loop
            if v == (n-1,0):
                find = True
                break

            # For all neighbors of the current cell, if they haven't been visited, we enqueue them to be visited in the next iteration and label them with the parent cell
            for value in values:
                if value(v, maze):
                    movimiento = direction[values.index(value)]
                    w = movimiento(v)
                    if not explored[w]:
                        explored[w] = True
                        parents[w] = v
                        Q.append(w)
        distancia += 1
    
    #We recover the way back with the dictionary
    path = []
    current = (n-1,0)
    while current is not None:
        path.append(current)
        current = parents[current]

    #Return the way back
    return path[::-1], len(path)
 


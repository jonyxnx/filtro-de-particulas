##Codigo del Proyecto

import numpy as np
import random
from matplotlib import pyplot as plt 
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

#FUNCIONES PARA EL FILTRO DE PARTÍCULAS

def brownian_motion(dt,N, grid):

    '''
    Función para simular el movimiento aleatorio

    PARAMETROS:
        dt (float): tamaño de paso
        N (int): numero de pasos
        grid (np.array): el laberinto en el nos movemos
    DEVUELVE:
        d (np.array): la trayectoria del movimiento
        dx (np.array): la derivada en el eje x de la trayectoria
        dy (np.array): la derivada en el eje y de la trayectoria
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
    Función para simular el movimiento aleatorio y aplicar el filtro a la vez

    PARAMETROS:
        dt (float): tamaño de paso
        N (int): numero de pasos
        grid (np.array): el laberinto en el nos movemos
        m (int): cantidad de particulas
        ruido (float): es la cantidad de ruido que añadimos la señal observada
        stop (bool): si es True, el filtro se detiene cuando se encuentra muy cerca de la posición real
    DEVUELVE:
        d (np.array): la trayectoria del movimiento
        dx (np.array): la derivada en el eje x de la trayectoria
        dy (np.array): la derivada en el eje y de la trayectoria
        mean (list): el proceso estimado
        obs (list): las observaciones con ruido generadas por la trayectoria

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

    #Actualizacion de las particulas:
        #Calculamos pesos:
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
    Filtro de partículas
    
    PARÁMETROS:
        Y (list): lista con las observaciones
        W (function): calcula los pesos 
        N (int): Cantidad de partículas
        f (function): distribución inicial de las partículas
    DEVUELVE:
        X (list): lista la densidad estimada

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
    Funcion para hacer remuestreo
    
    PARÁMETROS:
        pesos (list): lista con las pesos
        X (list): particulas
    DEVUELVE:
        X (list): lista con las particulas nuevas

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


#FUNCIONES PARA EL LABERINTO

def up(cell):

    '''
    
    Cambia la posicion actual a la posición superior

    PARAMETROS:
    cell  (tuple): celda actual

    DEVUELVE:
    (tuple): La celda superior a la celdad actual
    '''

    return cell[0], cell[1] + 1

def down(cell):

    '''
    
    Cambia la posición actual a la posición inferior

    PARAMETROS:
    cell (tuple): celda actual

    DEVUELVE:
    (tuple): La celda inferior a la celdad actual
    '''

    return cell[0], cell[1] - 1

def right(cell):

    '''
    
    Cambia la posición actual a la posición derecha

    PARAMETROS:
    cell: (tuple) celda actual

    DEVUELVE:
    (tuple) La celda a la derecha de la celdad actual
    '''

    return cell[0] + 1, cell[1]

def left(cell):

    '''
    
    Cambia la posición actual a la posición izquierda

    PARAMETROS:
    cell (tuple): celda actual

    DEVUELVE:
    (tuple): La celda a la derecha de la celdad actual
    '''

    return cell[0] - 1, cell[1]

def get_random_path(start_cell: (int, int), unvisited: np.array, n) -> [(int, int)]:
    
    '''
    
    Genera una caminata aleatoria

    PARAMETROS: 
        start_cell (tuple): celda en la que comenzamos la caminata
        unvisited (np.array): las lista de casillas que no hemos visitatado y por las que puede transitar la caminata
        n (int): corresponde al tamaño del laberinto sobre el que se está trabajando

    DEVUELVE:
        path (list): la secuencia de celdas sobre las que se realiza la caminata
    '''


    path = [start_cell]

    while True:
        #Checamos todas las casillas a las que nos podemos mover 
        possible_cells = possible_moves(n, start_cell)
        
        #Tomamos una de esas celdas
        start_cell = random.choice(possible_cells)

        #Si no la hemos visitado la guardamos en el camino y devolvemos el camino
        if start_cell not in unvisited:
            path.append(start_cell)
            return path
        
        #Si ya la visitamos en la camina, reiniciamos el camino
        elif start_cell in path:
            path = path[:path.index(start_cell) + 1]
        #En otro caso guardamos la caminata 
        else:
            path.append(start_cell)

def link_cells(grid, cell1, cell2):

    '''

    Esta función modifica la matriz donde se almacena la  información del laberinto para guardar los caminos que se van generando

    PARAMETROS:
        grid (np.array): una matriz de nxn con tuplas de tamaño 2, donde se almacena la información del laberinto
        cell1 (tuple): la celula de la que se parte
        cell2 (tuple): la celula a donde se pasa

    '''

    #Obtenemos la información de las celdas
    i = cell1[0]
    j = cell1[1]
    
    k = cell2[0]
    l = cell2[1]
    

    #Quitamos las paredes por donde se haya pasado
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
    Esta función se encarga de verificar que el movimiento que realizamos no nos saque del recuadro
    
    PARAMETROS:
        n (int): tamaño de la cuadricula del laberinto
        cell (tuple): posicion en el laberinto 
    
    DEVUELVE:
        moves (list): una lista con las posibles celdas hacia las que se puede mover 
    '''

    #Creamos una lista para almacenar los movimientos que puede hacer
    moves = []


    for direction in [up, down, right, left]:
        new_cell = direction(cell)
        #Si el robot no se sale del recuadro al hacer el movimiento lo movemos
        if 0 <= new_cell[0] < n and 0 <= new_cell[1] < n:
            moves.append(new_cell)

    #Devolvemos los posibles movimientos del robot
    return moves

def can_go_up(cell,maze):

    '''

    Esta función verifica con la información el laberinto si es posible realizar un movimiento hacia arriba

    PARAMETROS:
        cell (tuple): posición actual
        maze (np.array): es la matriz de nxn donde se almacena la información del laberinto
    DEVUELVE:
        (bool): a partir de la información del laberinto devuelve TRUE si no hay una pared, FALSE si exite una pared
    '''

    i = cell[0]
    j = cell[1]
    return maze[i, j][1]

def can_go_left(cell, maze):

    '''

    Esta función verifica con la información el laberinto si es posible realizar un movimiento hacia la izquierda

    PARAMETROS:
        cell (tuple): posición actual
        maze (np.array): es la matriz de nxn donde se almacena la información del laberinto
    DEVUELVE:
        (bool): a partir de la información del laberinto devuelve TRUE si no hay una pared, FALSE si exite una pared
    '''

    i, j = left(cell)
    return maze[i, j][0] 

def can_go_right(cell, maze):


    '''

    Esta función verifica con la información el laberinto si es posible realizar un movimiento hacia la derecha

    PARAMETROS:
        cell (tuple): posición actual
        maze (np.array): es la matriz de nxn donde se almacena la información del laberinto
    DEVUELVE:
        (bool): a partir de la información del laberinto devuelve TRUE si no hay una pared, FALSE si exite una pared
    '''


    i = cell[0]
    j = cell[1]
    return maze[i,j][0]

def can_go_down(cell, maze):

    
    '''

    Esta función verifica con la información el laberinto si es posible realizar un movimiento hacia abajo

    PARAMETROS:
        cell (tuple): posición actual
        maze (np.array): es la matriz de nxn donde se almacena la información del laberinto
    DEVUELVE:
        (bool): a partir de la información del laberinto devuelve TRUE si no hay una pared, FALSE si exite una pared
    '''


    i, j = down(cell)
    return maze[i, j][1]

def generate_maze(n, draw = False):
    
    '''
    Funcion para crear el laberinto aleatorio

    PARAMETROS:
        n (int): es el tamaño que queremos que tenga el laberinto
        draw (bool): TRUE si queremos dibujar un laberinto
    DEVUELVE:
        grid (np.array): es un arreglo que almacena la información del laberinto
    '''

    #Creamos una matriz de nxn con tuplas como entradas
    grid = np.zeros((n, n, 2), bool)

    #Guardamos las celdas que no hemos visitado
    unvisited_indices = [x for x in np.ndindex((n, n))]
    
    #Desordenamos los lugares sin visitar
    random.shuffle(unvisited_indices)

    unvisited_indices.pop()

    #Hacemos un ciclo para verificar que visitamos todas las celdas del laberinto
    while len(unvisited_indices) > 0:
        
        #Tomamos la primera celda
        cell = unvisited_indices[0]

        #Realizamos una caminata la caminata aleatoria sin ciclos
        path = get_random_path(cell, unvisited_indices, n)

        
        #Guardamos la información del laberinto en el grid
        for i in range(len(path) - 1):
            link_cells(grid, path[i], path[i + 1])

            unvisited_indices.remove(path[i])

    #Opcion de mostrar graficamente el laberinto creado        
    if draw == True:
        draw_maze(grid, None, False)
    return grid

def draw_maze(grid, robot = None, create = True, particles = [], path = [],  trajectory = [], estimate = [], observacion = []):
    

    '''
    Creamos una función que nos ayude a visualizar el laberinto:
    
    PARAMETROS:
        grid(matrix): nos ayuda a definir el laberinto
        robot(tuple): representa la posicion en la que se encuentra el robot
        create(bool): True si queremos crear un robot en una posición aleatoria
        particles(list): las particulas que queremos dibujar
        path(list): el camino discreto que queremos dibujar a través de las celdas del laberinto
        trajectory(list): trayectoria continua que queremos graficar
        estimate(list): trayectoria continua que que queremos dibujar
        observacion(list): lista de objetos que observamos

    DEVUELVE:
        (cell): regresa la posicion del robot en caso se que se haya creado uno
    '''
    #Creamos la figura
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    width = grid.shape[0]
    height = grid.shape[1]
    
    #Graficamos las orillas del laberinto
    plt.plot([0, width], [0, 0], color="black")
    plt.plot([0, 0], [0, height], color="black")
    plt.plot([0, width], [height, height], color="black")
    plt.plot([width, width], [0, height], color="black")
    
    #Graficamos el laberinto con la informacion de la variable grid
    for j in range(height):
        for i in range(width):
            value = grid[i, j]
            if not value[0]:
                
                plt.plot([i + 1, i + 1], [j, j + 1], color="black")

            if not value[1]:
                
                plt.plot([i, i + 1], [j + 1, j + 1], color="black")
    
    #Coloreamos el recuadro de salida del llaberinto en verde 
    #rect = Rectangle((width-1, 0), 1, 1, linewidth=2, edgecolor='none', facecolor='green')
    #ax.add_patch(rect)
    
    #En caso de que la variable path sea no vacia la graficamos
    for k in path:
        i, j = k
        rect = Rectangle((i, j), 1, 1, linewidth=1, edgecolor='none', facecolor='yellow')
        ax.add_patch(rect)
    
    #Creamos un robot aleatorio
    if create == True:
        n = len(grid)
        i = random.randint(1, n-1)
        j = random.randint(1, n-1)
        robot = (i,j)
    
    #Si existe un robot lo dibujamos
    if  robot != None:
        i, j = robot
        circulo = Circle((i+.5, j+.5), .5, linewidth=1, edgecolor='none', facecolor='red')
        ax.add_patch(circulo)
        
    #Añadimos las particulas al plot    
    for elemento in particles:
        i, j = elemento
        circulo = Circle((i, j), .1, linewidth=.2, edgecolor='none', facecolor='blue')
        ax.add_patch(circulo)
    
    if len(trajectory) != 0:
        plt.plot([t[0] for t in trajectory], [t[1] for t in trajectory], color = "green", linewidth = 1, label = "Trayectoria Real")

    if len(estimate)!=0:
        plt.plot([j[0] for j in estimate],[j[1] for j in estimate], label = "Estimación") 

    if len(observacion) != 0:
        plt.scatter([t[0] for t in observacion], [t[1] for t in observacion], s= .4, color = "red", linewidth = .8, label = "Señal")
    
    if len(observacion) != 0 or len(estimate)!=0 or len(trajectory) != 0:
        plt.legend(loc='lower right')
    plt.show()
    return robot

def rebote(tx, ty, dx,dy, l = False, u = False, d = False , r = False):
    
    '''
    Esta funcion calcula el rebote cuando el movimiento simulado choca con un laberinto
    
    PARAMETROS:
        tx (float): posicion actual en el eje x
        ty (float): posicion actual en el eje y 
        dx (float): razon de cambio en el eje x
        dy (float): razon de cambio en el eje y
        l (bool): Indica si se puede pasar a la izquierda de la celda
        u (bool): Indica si se puede pasar hacia arriba de la celda
        d (bool): Indica si se puede pasar hacia abajo de la celda
        r (bool): Indica si se puede pasar hacia la derecha de la celda

    DEVUELVE:
        (list): La posición de rebote

    '''    

    #calculamos la pendiente
    pendiente = dy / dx
    
    #La redondeamos para evitar bugs
    if abs(pendiente) > 1000 or abs(pendiente) < .0001:
        return rebote(tx, ty, .001, .001, l, u, d, r)
    
    else:
        x = tx+dx 
        y = ty+dy

    #Calculamos la ordenada al origen
    b = ty-pendiente*tx
    
    #Calculamos el rebote dependiendo del punto en donde rebote
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
    Esta funcion calcula la posicion tras rebotar en una pared del laberinto
    
    PARAMETROS:
        maze (np.array): almacena la información del laberinto
        tx (float): posicion actual en el eje x
        ty (float): posicion actual en el eje y
        difx (float): razon de cambio en el eje x
        dify (float): razon de cambio en el eje y
    DEVUELVE:
        D (list): Es la posición tras el rebote en la pared
    '''
    #Verificamos si hay una pared encima o no
    values = [can_go_left, can_go_up, can_go_down, can_go_right]
    
    #Calculamos la celda en la que se encuentra
    posx = np.floor(tx) 
    posy = np.floor(ty)
    

    tx = tx-posx
    ty = ty-posy

    cell = (int(posx), int(posy))
    
    pared = []

    #Verificamos hacia que lado se puede mover 
    for k in values:
        pared.append(k(cell, maze))

    izquierda = pared[0] 
    arriba = pared[1]
    abajo = pared[2]
    derecha = pared[3]

    #Aplicamos la funcion rebote
    C = rebote(tx, ty, difx, dify, izquierda, arriba, abajo, derecha)
    
    D = [C[0]+ posx, C[1]+posy]
    return D


#Necesitamos como input el laberinto y la particulas que ya localizo al robot
def solve(maze, particle):
    #Colocamos las funciones para verificar si puede moverse y las de movimiento
    values = [can_go_left, can_go_up, can_go_down, can_go_right]
    direction = [left, up, down, right]

    #Hacemos una cola con la particula
    Q = [particle]

    #Creamos una lista para guardar las celdas a las que ya pasamos
    n = len(maze)
    explored = np.zeros((n,n), bool)

    #Iniciamos una variable para calcular la distancia al centro y para avisar cuando encontremos la salida
    distancia = 0
    find = False

    #Etiquetamos la prirmera particula como explorada
    explored[particle] = True
    
    #Creamos un diccionario para guardar la informacion de donde precede cada celulal visitada
    rows, cols, _ = maze.shape
    parents = {cell: None for cell in np.ndindex((rows,cols))}  

    #Empezamos un ciclo que termina cuando visitemos todos los nodos o cuando encontremos la salida
    while len(Q) > 0 and find == False:

        #Hacemos un ciclo para cada elemento en la cola
        for i in range(len(Q)):
            
            #Escogemos una celda de la cola
            v = Q[0]
            Q.pop(0)

            #Visitamos los vecinos, cuando encotremos la salida detenemos el ciclo 
            if v == (n-1,0):
                find = True
                break

            #Para todos los vecinos de la celda en la que estamos, si no los hemos visitado, lo metemos a la cola para que sea visitado en la siguiente iteracion y lo etiquetamos con la celda-padre
            for value in values:
                if value(v, maze):
                    movimiento = direction[values.index(value)]
                    w = movimiento(v)
                    if not explored[w]:
                        explored[w] = True
                        parents[w] = v
                        Q.append(w)
        distancia += 1
    
    #Con ayuda del diccionario recuperamos el camino de regreso
    path = []
    current = (n-1,0)
    while current is not None:
        path.append(current)
        current = parents[current]

    #Devolvemos el camino de regreso
    return path[::-1], len(path)
 



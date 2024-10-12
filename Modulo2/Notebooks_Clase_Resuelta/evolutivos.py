import numpy as np
import copy
import random
import sys
import math

class AG:
    def __init__(self, fun_cost):
        self.fun_cost = fun_cost

    def roulete_wheel_selection(self, p):
        c = np.cumsum(p)
        r = sum(p)*np.random.rand()
    
        ind = np.argwhere(r<c)
        return ind[0][0]

    def crossover(self, p1,p2):
        c1=copy.deepcopy(p1)
        c2=copy.deepcopy(p2)
    
        alpha = np.random.uniform(0,1,*(c1['posicion'].shape))
        c1['posicion'] = alpha*p1['posicion'] +(1-alpha)*p2['posicion']
        c2['posicion'] = alpha*p2['posicion'] +(1-alpha)*p1['posicion']
    
        return c1,c2
    
    def mutate(self, c, mu, sigma):
        # mu es la media de la distribucion normal
        # sigma es la desviacion estandar
        # mutacion = gen_original + (tamaño de paso)*numero aleatorio con dist normal
        y = copy.deepcopy(c)
        flag = np.random.rand(*(c['posicion'].shape)) <= mu
        ind = np.argwhere(flag)
        y['posicion'][ind] += sigma*np.random.randn(*ind.shape)
        return y
        
    def bounds(self, c, varmin, varmax):
        c['posicion'] = np.maximum(c['posicion'], varmin)
        c['posicion'] = np.minimum(c['posicion'], varmax)
    
    def sort(self, arr):
        n = len(arr)
        for i in range(n-1):
            for j in range(0,n-i-1):
                if arr[j]['cost']>arr[j+1]['cost']:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    
    def ga(self, costfun, num_var, varmin, varmax, maxit, npop, num_hijos, mu, sigma, beta, x_points, y_points, N):
        # Inicializar la pobalcion
        poblacion = {}
        for i in range(npop):
            poblacion[i] = {'posicion': None, 'cost': None}
    
        bestsol = copy.deepcopy(poblacion)
        bestsol_cost = np.inf
        
        for i in range(npop):
            poblacion[i]['posicion'] = np.random.uniform(varmin, varmax, num_var)
            poblacion[i]['cost'] = costfun(poblacion[i]['posicion'], x_points, y_points, N)
    
            if poblacion[i]['cost'] < bestsol_cost:
                bestsol = copy.deepcopy(poblacion[i])
    
        print(f'best_sol: {bestsol}')
    
        bestcost = np.empty(maxit)
        bestsolution = np.empty((maxit, num_var))
    
        for it in range(maxit):
            # Calcular las probabilidades de ruleta
            costs = []
            for i in range(len(poblacion)):
                costs.append(poblacion[i]['cost'])
            costs = np.array(costs)
            avg_cost = np.mean(costs)
    
            if avg_cost != 0:
                costs = costs/avg_cost
            probs = np.exp(-beta*costs)
    
            for _ in range(num_hijos//2):
                # seleccion por ruleta
                p1 = poblacion[self.roulete_wheel_selection(probs)]
                p2 = poblacion[self.roulete_wheel_selection(probs)]
    
                # Crossover de lo padres
                c1, c2 = self.crossover(p1,p2)
                
                # Realizar la mutacion
                c1 = self.mutate(c1, mu, sigma)
                c2 = self.mutate(c2, mu, sigma)
                
                # Realizar el acotamiento de lo nuevos individuos
                self.bounds(c1, varmin, varmax)
                self.bounds(c2, varmin, varmax)
                
                #Evaluamos la funcion de costo
                c1['cost'] = costfun(c1['posicion'], x_points, y_points, N)
                c2['cost'] = costfun(c2['posicion'], x_points, y_points, N)
    
                if type(bestsol_cost)==float:
                    if c1['cost']<bestsol_cost:
                        bestsol_cost = copy.deepcopy(c1)
                else:
                    if c1['cost']<bestsol_cost['cost']:
                        bestsol_cost = copy.deepcopy(c1)
        
                if c2['cost']<bestsol_cost['cost']:
                    bestsol_cost = copy.deepcopy(c2)
                    
            # Juntar la poblacion de la generacion anterior con la nueva generacion
            poblacion[len(poblacion)] = c1
            poblacion[len(poblacion)] = c2
            
            # Ordenar la pobalcion tomando en cuenta la nueva poblacion agregada
            poblacion = self.sort(poblacion)
        
            # Almacenar en best cost, bestsoluction el historial de la optimizacion
            bestcost[it] = bestsol_cost['cost']
            bestsolution[it] = bestsol_cost['posicion']
            print(f'Iteracion: {it}, best_sol: {bestsolution[it]}, best_cost: {bestcost[it]}')
    
        out = poblacion  
        return(out, bestsolution, bestcost)

# Clase PSO
import random
import math # cos() for Rastrigin
import copy # array-copying convenience
import sys # max float


#particle class
class Particula:
    def __init__(self, fitness, x_points, y_points, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)

        # initialize position of the particle with 0.0 value
        self.position = [0.0 for i in range(dim)]

        # initialize velocity of the particle with 0.0 value
        self.velocity = [0.0 for i in range(dim)]

        # initialize best particle position of the particle with 0.0 value
        self.best_part_pos = [0.0 for i in range(dim)]

        # loop dim times to calculate random position and velocity
        # range of position and velocity is [minx, max]
        for i in range(dim):
            self.position[i] = ((maxx - minx)*self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx)*self.rnd.random() + minx)

        # compute fitness of particle
        self.fitness = fitness(self.position,x_points, y_points) # curr fitness

        # initialize best position and fitness of this particle
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness # best fitness

    # particle swarm optimization function
    def pso(fitness, x_points, y_points, max_iter, n, dim, minx, maxx):
        # hyper parameters
        w = 3 # inertia
        c1 = 2.3 # cognitive (particle)
        c2 = 2.5 # social (swarm)

        rnd = random.Random(0)

        # create n random particles
        swarm = [Particle(fitness, x_points, y_points, dim, minx, maxx, i) for i in range(n)]

        # compute the value of best_position and best_fitness in swarm
        best_swarm_pos = [0.0 for i in range(dim)]
        best_swarm_fitnessVal = sys.float_info.max # swarm best

        # computer best particle of swarm and it's fitness
        for i in range(n): # check each particle
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)

        # main loop of pso
        Iter = 0
        best_swarm_pos_hist = {}
        best_swarm_fitnessVal_hist = {}
        
        while Iter < max_iter:

            # after every 10 iterations
            # print iteration number and best fitness value so far
            best_swarm_pos_hist[Iter] = best_swarm_pos
            best_swarm_fitnessVal_hist[Iter] = best_swarm_fitnessVal
            if Iter % 10 == 0 and Iter > 1:
                print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)
                print(f'best_position: {best_swarm_pos}')

            for i in range(n): # process each particle

                # compute new velocity of curr particle
                for k in range(dim):
                    r1 = rnd.random() # randomizations
                    r2 = rnd.random()

                    swarm[i].velocity[k] = (
                                            (w * swarm[i].velocity[k]) +
                                            (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) +
                                            (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k]))
                                        )


                    # if velocity[k] is not in [minx, max]
                    # then clip it
                    if swarm[i].velocity[k] < minx:
                        swarm[i].velocity[k] = minx
                    elif swarm[i].velocity[k] > maxx:
                        swarm[i].velocity[k] = maxx


            # compute new position using new velocity
            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]

            # compute fitness of new position
            swarm[i].fitness = fitness(swarm[i].position,x_points, y_points)

            # is new position a new best for the particle?
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)

            # for-each particle
            Iter += 1
        #end_while
        return best_swarm_pos, best_swarm_pos_hist, best_swarm_fitnessVal_hist
        # end pso

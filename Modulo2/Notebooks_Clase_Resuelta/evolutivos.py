import numpy as np
import copy

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
class Particula:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        
        #Inicializar la posición de la particula con 0.0
        self.posicion = [0.0 for i in range (dim)]
        
        #Inicializar la velocidades de las particulas
        self.velocidad = [0.0 for i in range(dim)]
        
        #inicializamos la lista de la mejor posición de la particula, con ceros
        self.mejor_posicion = [0.0 for i in range(dim)]
        
        #Obtener de manera aleatoria velocidades y posiciones dependiendo de minx, maxx
        for i in range(dim):
            self.posicion[i] = ((maxx - minx)*self.rnd.random() + minx)
            self.velocidad[i] = ((maxx - minx)*self.rnd.random() + minx)
            
        # Evaluar las particulas en la función de costo
        self.fitness = fitness(self.posicion)
        
        #Inicializar la mejor posicion y el mejor fitness
        self.mejor_posicion_sig = copy.copy(self.posicion)
        self.mejor_fitnessVal_sig = self.fitness #mejor fitness 
        
    #Función del pso principal (donde se realiza la optimización)
    def pso(fitness, max_iter, n, dim, minx, maxx):
        # Hiperparámetros
        w = 0.729 #inercia
        c1 = 1.49445 # coeficiente cognitivo
        c2 = 1.49445 #coeficiente social
        
        rnd = random.Random(0)
        
        #Crear un enjambre de particulas de n elementos
        swarm = [Particula(fitness, dim, minx, maxx, 0) for i in range(n)]
        
        #Calcular el valor de la mejor posicion y el mejor fitness en el enjambre de particulas
        mejor_enjambre_sig = [0.0 for i in range(dim)]
        mejor_enjambre_fitnessVal = sys.float_info.max
        
        #Calcular la mejor particula con su fitness
        for i in range(n): #recorrer en cada particula
            if swarm[i].fitness < mejor_enjambre_fitnessVal:
                mejor_enjambre_fitnessVal = swarm[i].fitness
                mejor_enjambre_sig = copy.copy(swarm[i].posicion)
                
        # Loop principal del algoritmo de PSO
        
        Iter = 0
        mejor_enjambre_sig_hist = {}
        mejor_enjambre_fitnessVal_hist = {}
        
        while Iter < max_iter:
            mejor_enjambre_sig_hist[Iter] = mejor_enjambre_sig
            mejor_enjambre_fitnessVal_hist = mejor_enjambre_fitnessVal
            
            #imprimir por cada 10 iteraciones el número de iteración y el mejor fitness
            if Iter %10 ==0 and Iter>1:
                print(f"Iteración= {Iter}, mejor_fitness {mejor_enjambre_fitnessVal}")
                print(f"Mejor Posición = {mejor_enjambre_sig}")
            
            for i in range(n):
                #Calcular la velocidad de cada particula
                for k in range(dim):
                    r1 = rnd.random()
                    r2 = rnd.random()
                    
                    swarm[i].velocidad[k] = (
                                              (w*swarm[i].velocidad[k]) +
                                              (c1*r1*(swarm[i].mejor_posicion_sig[k] - swarm[i].posicion[k])) + 
                                              (c2*r2*(mejor_enjambre_sig[k] - swarm[i].posicion[k]))
                                            )
                    if swarm[i].velocidad[k] <minx:
                        swarm[i].velocidad[k] = minx
                    elif swarm[i].velocidad[k] > maxx:
                        swarm[i].velocidad[k] = maxx
                
                #calcular la nueva posición en función a la velocidad de cada particula
                for k in range(dim):
                    swarm[i].posicion[k] += swarm[i].velocidad[k]
                
                #Calcular el nuevo fitness de la nueva posicion
                swarm[i].fitness = fitness(swarm[i].posicion)
                
                #Es esta nueva posición una nueva mejor particula
                if swarm[i].fitness < swarm[i].mejor_fitnessVal_sig:
                    swarm[i].mejor_fitnessVal_sig = swarm[i].fitness
                    swarm[i].mejor_posicion_sig = copy.copy(swarm[i].posicion)
                
                #Es esta nueva posición una nueva mejor posición en todo el enjambre
                if  swarm[i].fitness < mejor_enjambre_fitnessVal:
                    mejor_enjambre_fitnessVal = swarm[i].fitness
                    mejor_enjambre_sig = copy.copy(swarm[i].posicion)
            
            #Incrementar el contador del while hasta n_iteraciones
            Iter +=1
        
        return mejor_enjambre_sig, mejor_enjambre_sig_hist, mejor_enjambre_fitnessVal_hist

import numpy as np
import copy

letras = ["A","B","C","D","E","F","G","H"]

def validar(state,crimen=2,min_porcent=0.1):
    """
    Determina si el estado es valido, es decir ningun nodo esta desconectado
    del resto y el porcentaje minimo de cada grupo se cumple por defecto 10%
    
    Devuelve True si es valido y False en caso contrario
    
    """
    result = True
    K = len(state["matrix"])
    for i in range(K+1):
        vector=[]
        for j in range(i):
            vector.append(state["matrix"][j][i-1-j])
        if i < K:
            vector+=state["matrix"][i]
        if sum(vector) == 0:
            result = False
            break
            
    P={x:0 for x in letras[:crimen]}
    for k in state["vector_crime"]:
        if k not in P:
            P[k]=1
        else:
            P[k]+=1
    P=np.array(P.values())*1.0/sum(P.values())    
    result2=True in (P < min_porcent)
    return result and not result2
            


def generate_estate_random(size,crimen=3):
    """
    Genera un estado para la simulacion del miedo al crimen determinado por una estructura
    de comunicacion y un vector que identifica la cantidad de crimen sufrido en un periodo de 
    tiempo
    
    devuelve un diccionario de la forma
    state={"matriz":array,"vector_crime":array(n)}
    """
    
    
    state={"matrix":[],"vector_crime":[np.random.choice(letras[:crimen]) for i in range(size)]}
    for i in range(size-1):
        vector=[]
        for j in range(size-i-1):
            vector.append(np.random.randint(2))
        state["matrix"].append(vector)
    
    return state
    
def inicializacion(n_estados,size,crimen=3,min_porcent=0.1):
    Poblacion=[]
    while(len(Poblacion) != n_estados):
        s=generate_estate_random(size,crimen)
        if validar(s,crimen,min_porcent) == True and s not in Poblacion:
            Poblacion.append(s)
    return Poblacion

def convert_matrix_to_vecinos(state):
    M=state["matrix"]
    C=state["vector_crime"]
    vecinos=[[i,c] for i,c in zip(range(len(M)+1), C)]
    for i in range(len(M)):
        for j in range(len(M[i])):
            if M[i][j] == 1 :
                vecinos[i].append(j+i+1)
                vecinos[j+i+1].append(i)
    return vecinos

def generate(vertices,psi,nu,T=200,s=np.array([None]),lamda={"A":0,"B":0.05,"C":0.5}):
    n=len(vertices)
    
    #periodos en semanas
    #T=150 #6 aos
    if s.all() == None:
        s = np.random.rand(n)  # vector PoS de las personas en el intante t, al principio aleatorio
        
        
    #psi = 0.9  # velocidad perdida de memoria
    #nu = 0.85  # Impacto de la inseguridad
    
    St = np.zeros((T,n ))  # PoS a lo largo del tiempo
    #identificacion de cada sujeto con su respectiva media de crimen
    
    St[0] = s
    
    for t in range(1,T):
        
        # Al inicio de cada periodo aplicamos la perdida de memoria
        s = psi * s       
        ## crimen
        for k in range(n):
            # numero de crimenes sufridos por la persona k 
            X = np.random.poisson(lamda[vertices[k][1]])
            # posicion hubo crimen o no
            I = 0
            if X >= 1:  # si hubo al menos un crimen I=1 de lo contrario I=0
                I = 1
            # efecto del crimen en la percepcion de k para el siguiente periodo
            s[k] = I + (1 - I) * s[k] 

        #comunicacion 

        #copia
        scopia=s.copy()
        
        for k in range(n):
            vecinos = vertices[k][2:]
            media = 0
            for vecino in vecinos:
                media+=scopia[vecino]
            media=media*1.0/len(vecinos)

            if scopia[k] > media:
                s[k]=nu*scopia[k]+(1-nu)*media
            else:
                s[k]=(1-nu)*scopia[k]+nu*media

        St[t] = s
        
        promedio=0
        for i in range(n):
            promedio+=np.mean(St.T[i][100:])
        promedio=promedio*1.0/n
        
    
    return St, promedio

def homofilia(vertices):
    edges={}
    for vertice in vertices:
        for vecino in vertice[2:]:
            if (vertice[0],vecino) in edges or (vecino,vertice[0]) in edges:
                pass
            else:
                edges[(vertice[0],vecino)]= 1*(vertice[1] == vertices[vecino][1])
    return sum(edges.values())*1.0/len(edges)

def funcion_objetivo(state,s0,tipo="h"):
    v=convert_matrix_to_vecinos(state)
    S=generate(v,0.98,0.85,T=200,s=s0)[1]
    if tipo == "h":
        return np.cos(S)+np.cos(homofilia(v))
    else:
        return np.cos(S)
        
      
def seleccion(Poblacion,s,tipo):
    fdp=[]
    for i in range(len(Poblacion)):
        fdp.append(funcion_objetivo(Poblacion[i],s,tipo))
    
    fdp=fdp/sum(fdp)

    return fdp

def sample(Poblacion,fdp):
    return np.random.choice(Poblacion,p=fdp)


def combinacion(state1,state2):
    
    k1=len(state1["matrix"])
    k2=k1+1
    
    hijo1=copy.deepcopy(state1)
    hijo2=copy.deepcopy(state2)
    
    rand1=np.random.randint(k1+1)
    
    
    for i in range(len(state1["matrix"])):
        for j in range(len(state1["matrix"][i])):
            if j < rand1:
                hijo1["matrix"][i][j]=state1["matrix"][i][j]
                hijo2["matrix"][i][j]=state2["matrix"][i][j]
            else:
                hijo1["matrix"][i][j]=state2["matrix"][i][j]
                hijo2["matrix"][i][j]=state1["matrix"][i][j]
                
    rand2=np.random.randint(k2+1)
    for i in range(len(state1["vector_crime"])):
        if i < rand2:
            hijo1["vector_crime"][i]=state1["vector_crime"][i]
            hijo2["vector_crime"][i]=state2["vector_crime"][i]
        else:
            hijo1["vector_crime"][i]=state2["vector_crime"][i]
            hijo2["vector_crime"][i]=state1["vector_crime"][i]
        
    return [hijo1,hijo2]    

def mutacion(state,crimen,probabilidad=None,min_porcent=0.1):
    
    copia=copy.deepcopy(state)
    
    if not probabilidad:
        probabilidad=1.0/len(state["vector_crime"])
    
    if np.random.binomial(1,probabilidad) == 1:
        rand1=np.random.randint(len(copia["vector_crime"]))
        anterior=copia["vector_crime"][rand1]
        
        while copia["vector_crime"][rand1] == anterior:
            copia["vector_crime"][rand1] = np.random.choice(letras[:crimen])
            
        rand2=np.random.randint(len(copia["matrix"]))
        rand3=np.random.randint(len(copia["matrix"][rand2]))
        
        if copia["matrix"][rand2][rand3] == 0:
            copia["matrix"][rand2][rand3] = 1
        else:
            copia["matrix"][rand2][rand3] = 0 
                     
        if validar(state=copia,crimen=crimen,min_porcent=min_porcent) == True:
            state=copia
    return state

def plot(vertices,s):
    import seaborn as sn
    import matplotlib.pyplot as plt
    colors=["R","B","G"]
    S=generate(vertices,0.9,0.85,100,s)[0].T
    
    promedio=0
    for i in range(20):
        promedio+=np.mean(S.T[i])
    promedio=promedio/20.0
        
    
    
    G={}
    for vertice in vertices:
        if vertice[1] not in G:
            G[vertice[1]]=[]
        G[vertice[1]].append(vertice[0])
        
    
    contador = 0
    for grupo in G.keys():
        sn.tsplot(S[G[grupo]],color=colors[contador],ci='sd')
        contador+=1

    plt.legend([str(k) for k in G.keys()])
    plt.xlabel("Time")
    plt.ylabel("Fear of crime")
    plt.show()
    return promedio
    
import numpy as np
import copy
import matplotlib.pyplot as plt
# from numba import jit


plt.ion()
letras = ["A","B","C","D","E","F","G","H"]

n=100
crimen = 3
min_porcent = 0.1

lamda={"A":0,"B":0.05,"C":0.5}
psi=0.9
nu=0.85
mu=1-nu
tipo="shg"
modelo="g_m_v" ## "g_1_v" ## 'random'


def read(file):
    # read the network
    filepath = file
    vertices = [];
    edges = [];
    readVertices = 0;
    readEdges = 0;
    with open(filepath) as fp:  
        line = fp.readline()
        cnt = 1
        while line:
            #print("Line {}: {}".format(cnt, line.strip()))

            if readVertices == 1:
                vertexData = line.strip().split(';');
                if len(vertexData)==4:
                    idxGroup = int(vertexData[3]);
                    vertices.append([int(vertexData[0]),idxGroup])

            if readEdges == 1:
                edgeData = line.strip().split(';');
                if len(edgeData)==2:
                    edges.append((int(edgeData[0]),int(edgeData[1])))
                    vertices[int(edgeData[0])].append(int(edgeData[1]));


            if line.strip() == '# Vertices':
#                 print('read vertices')
                readVertices = 1;
            if line.strip() == '# Edges':
                readVertices = 0;
                readEdges = 1;
#                 print('read edge')
            line = fp.readline()
            cnt += 1
            line.strip() 
    #array con 1 elemento: numero vertice, 2 elemento: grupo que se encuentra
    # resto son los vertices con los que se comunica
    #print(edges)
    for v in vertices:
        v[1]=letras[v[1]]
    return vertices,list(set(edges))


def porcentaje(vecinos):
    P=[]
    for i in vecinos:
        P.append(i[1])
    P=np.array(P)
    P_=np.unique(P,return_counts=True)
    return {label:p for label,p in zip(P_[0],P_[1]/len(vecinos))}


def convert_state_to_vecinos(state,dist_crimen,n=n):
    vecinos=[[i,c] for i,c in zip(range(n),dist_crimen)]
    for i in range(1,n):
        k=sum([j for j in range(n-i+1,n)])
        m=sum([j for j in range(n-i,n)])

        v=state[k:m]

        for j in range(len(v)):
            if v[j] == 1 :
                vecinos[i-1].append(j+i)
                vecinos[j+i].append(i-1)
    return vecinos  


def convert_vertices_to_graph(vertices,edges=None):
    import networkx as nx
    G = nx.Graph()
    for i in vertices:
        G.add_node(i[0],crime=i[1])
    if edges == None:
        edges=[]
        for i in vertices:
            for j in i[2:]:
                if (i[0],j) in edges or (j,i[0]) in edges:
                    pass
                else:
                    edges.append((i[0],j))
    G.add_edges_from(edges)
    return G


def validar_estado(estado,n=n):
    import networkx as nx
    
    vecinos=[[i,"A"] for i in range(n)]
    for i in range(1,n):
        k=sum([j for j in range(n-i+1,n)])
        m=sum([j for j in range(n-i,n)])

        v=estado[k:m]

        for j in range(len(v)):
            if v[j] == 1 :
                vecinos[i-1].append(j+i)
                vecinos[j+i].append(i-1)
    
    G=convert_vertices_to_graph(vecinos)
    
    return nx.is_connected(G)
    




def generate_estate(n=n,p=None):
    if not p :
        p=np.random.rand()
    total=int(n*(n-1)/2)
    estado=np.ndarray.tolist(np.random.binomial(n=1,p=p,size=total))
    contador=0
    while validar_estado(estado=estado,n=n) == False:
        if contador > 100:
            p=np.random.rand()
            contador=0
        estado=np.ndarray.tolist(np.random.binomial(n=1,p=p,size=total))        
        contador+=1
    return estado



def dist_crimen(crimen=crimen,n=n,porcentaje=np.array([None]),min_porcent=min_porcent):
    
    if porcentaje.all() == None:
        vector_random=np.random.rand(crimen)
        vector_random/=sum(vector_random)
    else:
        vector_random=porcentaje
        
    while True in (vector_random < min_porcent):
        vector_random=np.random.rand(crimen)
        vector_random/=sum(vector_random)
    distribucion_crimen=[]
    for i in vector_random:
        distribucion_crimen.append(int(round(i*n)))
    while 0 in distribucion_crimen or sum(distribucion_crimen) != n:
        vector_random=np.random.rand(crimen)
        vector_random/=sum(vector_random)
        while True in (vector_random < min_porcent):
            vector_random=np.random.rand(crimen)
            vector_random/=sum(vector_random)
        distribucion_crimen=[]
        for i in vector_random:
            distribucion_crimen.append(int(round(i*n)))
    r=[]
    for j in range(len(distribucion_crimen)):
        for i in range(distribucion_crimen[j]):
            r+=[letras[j]]
    return r
        


def convert_vecinos_to_solution(vecinos):
    n=len(vecinos)
    M=[[0]*n for i in range(n)]
    solution=[]
    for i in range(n):
        for j in vecinos[i][2:]:
            M[i][j]=1
            
    for i in range(len(M)):
        solution+=M[i][i+1:]
    return solution
    

def generate(vertices,psi=psi,nu=nu,mu=mu,T=200,s=np.array([None]),lamda=lamda,modelo=modelo,contar=None):
    n=len(vertices)
    
    
    if modelo == "g_m_v":
        Adj=(np.zeros((len(vertices),len(vertices))) == 1)
        n_vecinos=np.zeros(len(vertices))
        
    
    conteo=np.zeros(n)
    
    
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
        # media de crimen por persona
        g=[lamda[vertices[i][1]] for i in range(n) ]
        g=np.array(g)
        # numero de crimenes sufridos por persona
        ## np.random.poisson(g)
        # # posicion hubo crimen o no
        I=(np.random.poisson(g) > 0)*1
        # efecto del crimen 
        s=I+(1-I)*s

        #comunicacion 

        #copia
        scopia=s.copy()
        
        if modelo == "g_m_v":
            
            for i in range(n):
                n_vecinos[i]=len(vertices[i][2:])
                Adj[i][vertices[i][2:]]=True
            
            # media opiniones vecinos 
            media=(scopia*Adj).sum(axis=1)/n_vecinos
            # pesos segun comparacion
            w=(scopia > media)*mu + (scopia < media)*nu
            # Efecto comunicacion
            s=scopia-w*(scopia-media)
        
            
                    
        elif modelo == "g_1_v":
            
            comu=np.random.permutation(n)[:int(n*0.1)]
            
            salto=[]
            
            for k in comu:
                
                if k in salto:
                    continue
                    
                if len(set(vertices[k][2:])-set(salto)) > 0:
                    
                    aux = np.random.choice(vertices[k][2:])

                    if scopia[k] > scopia[aux]:
                        s[k]=(1-mu)*scopia[k]+mu*scopia[aux]
                        s[aux]=nu*scopia[k]+(1-nu)*scopia[aux]
                    else:
                        s[k]=(1-nu)*scopia[k]+nu*scopia[aux]
                        s[aux]=mu*scopia[k]+(1-mu)*scopia[aux]
                        
                    conteo[k]+=1
                    conteo[aux]+=1
                        
                    salto.append(k)
                    salto.append(aux)
                    
        elif modelo == "random":
            
            #numero de personas que se comunican
            n_comu=int(n*0.2)
            if n_comu % 2 == 1:
                n_comu+=1
            # indices personas que se comunican
            comu=np.random.permutation(n)[:n_comu]
            # primer persona de la pareja
            first=comu[int(n_comu/2):]
            # segunda persona de la pareja
            second=comu[:-int(n_comu/2)]
            # pesos persona uno a su opinion
            w1=(s[first]>s[second])*mu +(s[first]<s[second])*nu
            # pesos persona dos a su opinion
            w2=(s[second]>s[first])*mu +(s[second]<s[first])*nu
            # actualizacion todas las personas que se comunican
            s[first]=scopia[first]-w1*(scopia[first]-scopia[second])
            s[second]=scopia[second]-w2*(scopia[second]-scopia[first])
            
            conteo[first]+=1
            conteo[second]+=1
                        
                    
        else:
            continue
            
            

        St[t] = s
        promedio=np.mean(St.T[:,int(T/2):])
        
    if contar == None:
        return St, promedio
    else:
        return St, conteo

def homofilia(G):
    return sum([G.node[i[0]] == G.node[i[1]] for i in G.edges])/len(G.edges)


def draw_graph(G,fear,crime=crimen,labels=False,save=False,file="",legends=None):
    import networkx as nx
    
    s=1000*fear+100
    colors  = {"A":"green","B":"blue","C":"red","D":"orange","E":"purple","F":"pink","G":"yellow"}
    
    labels=[]
    color=[colors[G.node[i]['crime']] for i in G.nodes]
    pos = nx.spring_layout(G)
    plt.figure(figsize=(30,20))
    nx.draw(G,pos=pos,with_labels=labels,node_size=s,node_color=color,alpha=0.5,font_color="w")
    
    
    from matplotlib.lines import Line2D
    
    if legends == None:
        legends={x:x for x in letras}

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=legends[key],
                              markerfacecolor=colors[key], markersize=40, alpha=0.5) for key in letras[:crime] ]

    # Create the figure
        
    plt.legend(handles=legend_elements, loc='best',fontsize=40)
    
    if save == True :
        plt.savefig(file)
    
    plt.show();
    
    
def assor(G):
    from networkx.algorithms.assortativity import attribute_assortativity_coefficient
    a=attribute_assortativity_coefficient(G=G,attribute='crime')
    return 0.5*a+0.5

def mixing_matrix(G,crimen=crimen):
    from networkx.algorithms.assortativity.mixing import attribute_mixing_matrix
    m={x:i for x,i in zip(letras[:crimen],range(crimen))}
    return attribute_mixing_matrix(G=G,attribute="crime",mapping=m)

def normpdf(x, mean, sd):
    import math
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def funcion_objetivo(state,s0,distribucion_crimen,n=n,tipo=tipo,lamda=lamda,psi=psi,nu=nu,mu=mu,modelo=modelo):
    
    v=convert_state_to_vecinos(n=n,state=state,dist_crimen=distribucion_crimen)
    S=generate(v,psi,nu,mu,T=200,s=s0,lamda=lamda,modelo=modelo)[1]
    g=[]
    for node in v:
        g.append(len(node[2:]))                
    gmean=np.mean(g)
    
    gg=normpdf(gmean,5,0.5)+0.1
    
    
    A=assor(convert_vertices_to_graph(v))
    
    if tipo == "sag":
        return (1-S)*(1-abs(A-0.8))*gg
    elif tipo == "sg":
        return (1-S)*gg
    elif tipo == "sa":
        return (1-S)*(1-abs(A-0.8))
    else:
        return (1-S)
                 
                 

def inicializacion(individuos,n=n,crimen=crimen,min_porcent=min_porcent):
    Poblacion=[]
    for i in range(individuos):
        Poblacion.append(generate_estate(n=n))
    return Poblacion


def seleccion(Poblacion,s,distribucion_crimen,tipo=tipo,lamda=lamda,nu=nu,mu=mu,psi=psi,modelo=modelo):
    n=len(s)
    fdp=[]
    for i in range(len(Poblacion)):
        fdp.append(funcion_objetivo(n=n,state=Poblacion[i],s0=s,tipo=tipo,distribucion_crimen=distribucion_crimen,
                                    lamda=lamda,nu=nu,mu=mu,psi=psi,modelo=modelo))
    fdp=fdp+abs(min(fdp))+1
    fdp=fdp/sum(fdp)

    return fdp

 
def sample(Poblacion,fdp):
    P=[i for i in range(len(Poblacion))]
    return Poblacion[np.random.choice(P,p=fdp)]

 
def combinacion(state1,state2,n=n):
    total=n*(n-1)/2+1
    point_cross=np.random.randint(total)   
    
    h1=state1[:point_cross]+state2[point_cross:]
    h2=state2[:point_cross]+state1[point_cross:]
    
    while (validar_estado(h1,n=n) and validar_estado(h2,n=n)) == False:
        point_cross=np.random.randint(total)   
        h1=state1[:point_cross]+state2[point_cross:]
        h2=state2[:point_cross]+state1[point_cross:]
    
    
    return h1,h2



 
def mutacion(state,n=n,probabilidad=None):
    
    copia=copy.deepcopy(state)
    if not probabilidad:
        probabilidad=1.0/len(state)
        
    if np.random.binomial(1,probabilidad) == 1:
        rand1=np.random.randint(len(state))
        copia[rand1]=(copia[rand1]+1)%2
    
        while validar_estado(copia,n=n) == False:
            copia=copy.deepcopy(state)
            rand1=np.random.randint(len(state))
            copia[rand1]=(copia[rand1]+1)%2
            
        return copia
    
    else:
        return state
    
   
    
        
def plot(vertices,s,lamda=lamda,psi=psi,nu=nu,mu=mu,modelo=modelo,T=220,save=False,f="",legends=None,draw=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    colors  = {"A":"green","B":"blue","C":"red","D":"orange","E":"purple","F":"pink","G":"yellow"}
    S,promedio=generate(vertices=vertices,psi=psi,nu=nu,
                        mu=mu,T=T,s=s,lamda=lamda,modelo=modelo)
    S=S.T#[:,int(T):]
        
    if draw == False:
        return S#promedio
    else:
        
        if legends == None:
            legends={x:x for x in letras[:len(lamda)]}
            
            
        
        grado_grupo =np.array([[len(vertice[2:]),legends[vertice[1]]] for vertice in vertices])            
        V=pd.DataFrame(grado_grupo,columns=['Grade','Group']).astype({'Grade': 'int64'})
        
        plt.figure(figsize=(10,3))
        A=pd.DataFrame()
        A['node']=np.array([[i]*T for i in range(S.shape[0])]).flatten()
        A['Time']=list(np.arange(T))*S.shape[0]
        A['Fear of Crime']=S.flatten()
        A['Group']=V.Group[A.node].values

        sns.lineplot(data=A,x='Time',y='Fear of Crime',hue='Group',ci='sd',
                     hue_order=legends.values(),palette=list(colors.values())[:len(legends)],legend='full')
        plt.legend(loc='upper left',ncol=4,bbox_to_anchor=(0,1.23))
        plt.ylim(0,1)
        if save == True:
            plt.savefig(f)
        plt.show()
        return promedio




def grafica_grado_nodos(vecinos):
   
    grade={}
    for v in vecinos:
        g=len(v[2:])
        if g not in grade:
            grade[g]=1
        else:
            grade[g]+=1
    Y=np.array(grade.values())*1.0/sum(grade.values())
    X=grade.keys()
    plt.plot(np.log(X),np.log(Y))
    plt.title("Distribucion Grado Nodos (Log-Log)")
    plt.xlabel("Log(k)")
    plt.ylabel("Log(P(k))")
    plt.show();
    


from networkx.algorithms import approximation, assortativity,centrality, cluster, distance_measures, link_analysis, smallworld
from networkx.classes import function

def ver_medidas(G):
    print(function.info(G))
    """
    Numero minimo de nodos que deben ser removidos para desconectar G
    """
    print("Numero minimo de nodos que deben ser removidos para desconectar G :"+str(approximation.node_connectivity(G)))

    """
    average clustering coefficient of G.
    """
    print("average clustering coefficient of G: "+str(approximation.average_clustering(G)))

    """
    Densidad de un Grafo
    """
    print("Densidad de G: "+str(function.density(G)))

    """
    Assortativity measures the similarity of connections in
    the graph with respect to the node degree.
    Valores positivos de r indican que existe una correlacion entre nodos 
    con grado similar, mientras que un valor negativo indica
    correlaciones entre nodos de diferente grado
    """

    print("degree assortativity:"+str(assortativity.degree_assortativity_coefficient(G)))

    """
    Assortativity measures the similarity of connections
    in the graph with respect to the given attribute.
    """

    print("assortativity for node attributes: "+str(assortativity.attribute_assortativity_coefficient(G,"crime")))

    """
    Grado promedio vecindad
    """
    plt.plot(assortativity.average_neighbor_degree(G).values())
    plt.title("Grado promedio vecindad")
    plt.xlabel("Nodo")
    plt.ylabel("Grado")
    plt.show();

    """
    Grado de Centralidad de cada nodo
    """

    plt.plot(centrality.degree_centrality(G).values())
    plt.title("Grado de centralidad")
    plt.xlabel("Nodo")
    plt.ylabel("Centralidad")
    plt.show();


    """
    Calcular el coeficiente de agrupamiento para nodos
    """

    plt.plot(cluster.clustering(G).values())
    plt.title("coeficiente de agrupamiento")
    plt.xlabel("Nodo")
    plt.show();

    """
    Media coeficiente de Agrupamiento
    """
    print("Coeficiente de agrupamiento de G:"+str(cluster.average_clustering(G)))

    """
    Centro del grafo
    El centro de un grafo G es el subgrafo inducido por el 
    conjunto de vertices de excentricidad minima.

     La  excentricidad  de  v  in  V  se  define  como  la
     distancia maxima desde v a cualquier otro vertice del 
     grafo G siguiendo caminos de longitud minima.
    """

    print("Centro de G:"+ str(distance_measures.center(G)))

    """
    Diametro de un grafo
    The diameter is the maximum eccentricity.
    """
    print("Diametro de G:"+str(distance_measures.diameter(G)))


    """
    Excentricidad de cada Nodo
    The eccentricity of a node v is the maximum distance
    from v to all other nodes in G.
    """
    plt.plot(distance_measures.eccentricity(G).values())
    plt.title("Excentricidad de cada Nodo")
    plt.xlabel("Nodo")
    plt.show();

    """
    Periferia 
    The periphery is the set of nodes with eccentricity equal to the diameter.
    """
    print("Periferia de G:")
    print(distance_measures.periphery(G))

    """
    Radio
    The radius is the minimum eccentricity.

    """

    print("Radio de G:"+str(distance_measures.radius(G)))

    """
    PageRank calcula una clasificacion de los nodos
    en el grafico G en funcion de la estructura de 
    los enlaces entrantes. Originalmente fue disenado
    como un algoritmo para clasificar paginas web.
    """

    plt.plot(link_analysis.pagerank_alg.pagerank(G).values())
    plt.title("Puntaje de cada Nodo")
    plt.xlabel("Nodo")
    plt.show();

    """
    Coeficiente de Small World.
    A graph is commonly classified as small-world if sigma>1.

    """

    print("Coeficiente de Small World: " + str(smallworld.sigma(G)))

    """
    The small-world coefficient (omega) ranges between -1 and 1.
    Values close to 0 means the G features small-world characteristics.
    Values close to -1 means G has a lattice shape whereas values close
    to 1 means G is a random graph.
    """
    print("Omega coeficiente: "+str(smallworld.omega(G)))
    

def run(folder,total_generaciones,individuos,porcentaje_crimen,s0,n=n,crimen=crimen,tipo_objetivo=tipo,lamda=lamda,min_porcent=min_porcent,modelo=modelo,nu=nu,mu=mu):
    
    import sys, os, shutil
    import matplotlib as mpl
    
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    archivo = open(os.path.join(folder,"solucion_"+str(tipo_objetivo)+"__"+str(modelo)+".txt"),'a')
    
    archivo.write("# nodos: "+ str(n)+ '\n')
    archivo.write("# grupos de crimen: "+ str(crimen)+ '\n')
    archivo.write("funcion objetivo: "+ str(tipo_objetivo)+ '\n')
    archivo.write("# de cromosomas por generacion: "+ str(individuos)+ '\n')
    archivo.write("minimo porcentaje de grupo: "+ str(min_porcent)+ '\n')
    archivo.write("Modelo utilizado: "+ str(modelo)+ '\n')
    archivo.write("valor de nu: "+ str(nu)+ '\n')
    archivo.write("valor de mu: "+ str(mu)+ '\n')
  
    archivo.write("-----------------------------------------------------------------"+'\n')

    distr_crimen=dist_crimen(crimen=crimen,n=n,
                          porcentaje=porcentaje_crimen,
                          min_porcent=min_porcent)
    
    Poblacion=inicializacion(individuos=individuos,
                         n=n,
                         crimen=crimen,
                         min_porcent=min_porcent)
    
    print("Poblacion inicial terminado")

    archivo.write("Media miedo inicial: "+ str(np.mean(s0))+ '\n')
    
    p_gen=[]
    
    for i in range(total_generaciones):
        archivo.write("-----------------------------------------------------------------------"+ '\n')
        archivo.write("Generacion "+ str(i)+ '\n')
        print("Generacion "+ str(i)+"/"+str(total_generaciones)+ '\n')

        puntaje_generacion=[]
        for estado in Poblacion:
            puntaje_generacion.append(funcion_objetivo(state=estado,
                                                       distribucion_crimen=distr_crimen,
                                                       s0=s0,
                                                       n=n,
                                                       tipo=tipo,
                                                       lamda=lamda,psi=psi,nu=nu,mu=mu,modelo=modelo))

        best_cromosome_generation=np.argsort(puntaje_generacion)[-1]

#         print("Mejor de la Generacion:")
#         print(Poblacion[best_cromosome_generation])

        media_crimen=plot(vertices=convert_state_to_vecinos(state=Poblacion[best_cromosome_generation],
                                                            dist_crimen=distr_crimen,n=n),
                          s=s0,lamda=lamda,psi=psi,nu=nu,mu=mu)

        archivo.write("Media de crimen:" + str(media_crimen)+ '\n')
        print("Media de crimen:" + str(media_crimen))
      
        archivo.write("Puntaje Generacion "+str(np.mean(puntaje_generacion))+ '\n')
        print("Puntaje Generacion "+str(np.mean(puntaje_generacion)))
        p_gen.append(np.mean(puntaje_generacion))

        #funcion densidad de probabilidad para muestrear estados de la poblacion actual que depende de su desempeno
        fdp=seleccion(Poblacion=Poblacion,
                      s=s0,
                      distribucion_crimen=distr_crimen,
                      tipo=tipo,
                      lamda=lamda,
                      nu=nu,mu=mu,psi=psi,modelo=modelo)

        #Hijos de la poblacion actual
        Nueva_Generacion=[]
        # Combinacion 
        while len(Nueva_Generacion) != len(Poblacion):
            padre1=sample(Poblacion=Poblacion,fdp=fdp)
            padre2=sample(Poblacion=Poblacion,fdp=fdp)
            hijos=combinacion(state1=padre1,state2=padre2,n=n)
            contador=0
            while (validar_estado(estado=hijos[0],n=n) and validar_estado(estado=hijos[1],n=n)) == False:
                hijos=combinacion(state1=padre1,state2=padre2,n=n)
                if contador == 200:
                    hijos=[padre1,padre2]
                    break
                contador+=1
            Nueva_Generacion+=hijos

        # Mutacion
        for i in range(len(Nueva_Generacion)):
            Nueva_Generacion[i]=mutacion(Nueva_Generacion[i],n=n)

        # Reemplazo

        total = Poblacion+Nueva_Generacion

        fo=[]
        for t in total:
            fo.append(funcion_objetivo(state=t,
                                       distribucion_crimen=distr_crimen,
                                       s0=s0,
                                       n=n,
                                       tipo=tipo,
                                       lamda=lamda,psi=psi,nu=nu,mu=mu))
        order=np.argsort(fo)[::-1]
        best=order[:len(Poblacion)]
        for i in range(len(Poblacion)):
            Poblacion[i]=total[best[i]]
            
    archivo.write("////////////////////////////////////////////////////////////////"+ '\n')
    
    puntaje_generacion=[]
    for estado in Poblacion:
        puntaje_generacion.append(funcion_objetivo(state=estado,
                                                   distribucion_crimen=distr_crimen,
                                                   s0=s0,
                                                   n=n,
                                                   tipo=tipo,
                                                   lamda=lamda,psi=psi,nu=nu,mu=mu))

    best_cromosome_generation=np.argsort(puntaje_generacion)[-1]
    
    media_crimen=plot(convert_state_to_vecinos(state=Poblacion[best_cromosome_generation],
                                               dist_crimen=distr_crimen,n=n),
                      s0,lamda,psi,nu,mu=mu)
    
    print("Mejor Solucion:")
    archivo.write("Media de crimen:" + str(media_crimen)+ '\n')
    print("Media de crimen:" + str(media_crimen))
    
    archivo.close()

    np.save(os.path.join(folder,"solucion_"+str(tipo_objetivo)+"__"+str(modelo)),Poblacion[best_cromosome_generation])
    np.save(os.path.join(folder,"solucion_"+str(tipo_objetivo)+"__"+str(modelo)+"__"+"p_gen"),p_gen)
    
    
    
    return Poblacion[best_cromosome_generation], distr_crimen, p_gen


    

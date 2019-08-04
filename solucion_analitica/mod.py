import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# warnings.filterwarnings("always")

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
                    edges.append([int(edgeData[0]),int(edgeData[1])])
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
    return vertices,edges

def generate(file,lamda,colores=None,legend=None,pesos=True):
    vertices,edges = read(file)
    n=len(vertices)
    #vector identificacion de grupo
    g=np.zeros(n)
    for i in range(n):
        g[i]=vertices[i][1]
    # cantidad de grupos
    m=int(max(g)+1)
    #porcentaje de cada grupo
    q=np.zeros(m)
    for i in range(m):
        q[i]=sum((g==i)*1.0)/n
    #periodos en semanas
    T=312 #6 aos
    s = np.random.rand(n)  # vector PoS de las personas en el intante t, al principio aleatorio
    psi = 0.9  # velocidad perdida de memoria
    nu = 0.85  # Impacto de la inseguridad
    
    St = np.zeros((T,n ))  # PoS a lo largo del tiempo
    #identificacion de cada sujeto con su respectiva media de crimen
    for i in range(n):
        for j in range(m):
            if g[i]==j:
                g[i]=lamda[j]
    St[0] = s
    
    for t in range(1,T):
        
        # Al inicio de cada periodo aplicamos la perdida de memoria
        s = psi * s

        
        ## crimen
        for k in range(n):
            # numero de crimenes sufridos por la persona k 
            X = np.random.poisson(g[k])
            # posicion hubo crimen o no
            I = 0
            if X >= 1:  # si hubo al menos un crimen I=1 de lo contrario I=0
                I = 1
            # efecto del crimen en la percepcion de k para el siguiente periodo
            s[k] = I + (1 - I) * s[k] 

        #comunicacion 

        #copia
        scopia=s.copy()
        
        if pesos == True:

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
                    
        else:
            
            for k in range(n):
                vecinos = vertices[k][2:]
                media = 0
                for vecino in vecinos:
                    media+=scopia[vecino]
                media=(media+scopia[k])*1.0/(len(vecinos)+1)


        St[t] = s
    print("Tamano poblacion")
    print(n)
    print("Porcentaje grupos con distinta media de crimen")
    print(q)
    print("Vector media de crimen por grupos")
    print(lamda)
    print("velocidad de olvido")
    print(psi)
    print("impacto de la inseguridad nu")
    print(nu)
    
    
    
          

#     %matplotlib inline 
    plt.figure(figsize=(20,5))
    print("Grafica PoS individual")
    plt.plot(St[104:],alpha=0.1)
    plt.plot(np.mean(St[104:],axis=1),'blue',linewidth=8)
    plt.show()
    sns.set(color_codes=True)
    plt.figure(figsize=(20,5))
    print("PoS media por grupos")
#     colores=["Red","Blue","Green","yellow","k"]
    G=[]
    for i in range(m):
        grupo=np.zeros((T,int(n*q[i])))
        contador=0
        for k in range(n):
            if g[k]==lamda[i]:
                grupo[:,contador]=St[:,k]
                contador=contador+1
        #plt.plot(grupo,alpha=0.1)
        if colores != None:
            sns.tsplot(data=grupo[104:].T,ci='sd',color=colores[i])
        else:
            sns.tsplot(data=grupo[104:].T,ci='sd')
            
        G.append(grupo[104:].T)
    h = plt.gca().get_lines()
    plt.legend(handles=h,
               labels=legend,
                       ncol=3,fontsize=20,bbox_to_anchor=(0, -0.3),
                       loc=2, borderaxespad=0)
    #plt.legend(["susceptible", "immune", "highly susceptible"],ncol=3,fontsize=20,bbox_to_anchor=(0, -0.3), loc=2, borderaxespad=0)
    plt.axvline(52, color = 'black',alpha=0.3)
    plt.axvline(104, color = 'black',alpha=0.3)
    plt.axvline(156, color = 'black',alpha=0.3)
    plt.xlabel("Time (years)",fontsize=20)
    plt.ylabel("PoS",fontsize=20)
    ax = plt.axes()
    ax.xaxis.set_ticks([0, 52, 104, 156, 208])
    ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    ax.tick_params(labelsize=15)
    plt.text(-25, 1.02,'Insecure', fontsize=20)
    plt.text(-25, 0,'Secure', fontsize=20)
    plt.show()
#     plt.savefig('expgraph1.pdf',bbox_inches="tight")
    return G

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


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

def generate(file,graph,lamda,colores=None,legend=None):
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
    nu = 0.9  # Impacto de la inseguridad
    mu = 0.1  # Resistencia a la inseguridad
    St = np.zeros((T,n ))  # PoS a lo largo del tiempo
    #identificacion de cada sujeto con su respectiva media de crimen
    for i in range(n):
        for j in range(m):
            if g[i]==j:
                g[i]=lamda[j]
    homofilia=np.zeros(T)
    St[0] = s
    parescom=np.zeros(T)#cantidad pares de comunicacion en cada periodo
    commismogrupo=np.zeros(T)#cantidad de comunicaciones en el mismo grupo
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

            #escogemos las parejas de comunicacion 
            
            
            paleatorio=np.random.permutation(np.arange(n))#personas grupo 1 en orden aleatorio
            persona1=paleatorio[:int(n*0.1)] #primer 10%
            
            if graph == True:
                persona2=np.zeros_like(persona1)
            else:
                persona2=paleatorio[-int(n*0.1):]#ultimo 10%


            for k in range(len(persona1)):
                #para grupo 1
                aux1=persona1[k]
                if graph == True:
                    aux2=np.random.choice(vertices[aux1][2:])
                    contador=0
                    while aux2 in persona1 or aux2 in persona2:
                        aux2=np.random.choice(vertices[aux1][2:])
                        contador=contador+1
                        if contador>5:
                            break
                else:
                    aux2=persona2[k]
                    
                if s[aux1]<s[aux2]:
                    s[aux1] = s[aux1] - nu * (s[aux1] - s[aux2])
                    s[aux2] = s[aux2] - mu * (s[aux2] - s[aux1])
                else:
                    s[aux1] = s[aux1] - mu * (s[aux1] - s[aux2])
                    s[aux2] = s[aux2] - nu * (s[aux2] - s[aux1])

                if g[aux1]==g[aux2]:
                    commismogrupo[t]=commismogrupo[t]+1

            parescom[t]=len(persona1)
            homofilia[t]=commismogrupo[t]/parescom[t]
            
            St[t] = s
    print("Tamano poblacion")
    print(n)
    print("cantidad de pares de comunicacion")
    plt.plot(parescom[1:])
    plt.show()
    print("Porcentaje grupos con distinta media de crimen")
    print(q)
    print("Vector media de crimen por grupos")
    print(lamda)
    print("velocidad de olvido")
    print(psi)
    print("impacto de la inseguridad nu")
    print(nu)
    print("Resistencia a la inseguridad mu")
    print(mu)
    print("Homofilia")
    print(np.mean(homofilia))
          

#     %matplotlib inline 
    plt.figure(figsize=(20,5))
    #print("Grafica PoS individual")
    #plt.plot(St[104:],alpha=0.1)
    #plt.plot(np.mean(St[104:],axis=1),'blue',linewidth=8)
    #plt.show()
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

def distributions(G,H,colores=None,legend=None,save=None,ymin=0,ymax=0):
    
    
    fig = plt.figure()
   
    
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Community Interactions')
    
    grupos=[]
    for k in range(len(G)):
        p=G[k].shape[0]
        g=np.zeros(p)
        for j in range(p):
            g[j]=np.mean(G[k][j])
        grupos.append(g)
        if colores != None:
            sns.distplot(g,
                         hist_kws={'alpha':0.5,
                                   'histtype':'stepfilled',
                                   'density':True},
                         color=colores[k],
                         vertical=True)
        else:
            sns.distplot(g,
                         hist_kws={'alpha':0.5,
                                   'histtype':'stepfilled',
                                   'density':True},                         
                         vertical=True)
            
    plt.gca().invert_xaxis()
    plt.ylabel('Fear of Crime')
    ax1.set_ylim([ymin, ymax])
#     ax1.set_xlim([0, 30])
    
    
    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Random Interactions')
    
    ax2.set_ylim([ymin, ymax])
#     ax2.set_xlim([30, 0])
    ax2.axes.get_yaxis().set_visible(False)
    
    grupos=[]
    for k in range(len(H)):
        p=H[k].shape[0]
        g=np.zeros(p)
        for j in range(p):
            g[j]=np.mean(H[k][j])
        grupos.append(g)
    
        if colores != None:
            sns.distplot(g,
                         hist_kws={'alpha':0.5,
                                   'histtype':'stepfilled',
                                   'density':True},
                         color=colores[k],
                         vertical=True)
        else:
            sns.distplot(g,
                         hist_kws={'alpha':0.5,
                                   'histtype':'stepfilled',
                                   'density':True},                         
                         vertical=True)
    # for i in range(5):
    #     print("mean")
    #     print(np.mean(grupos[i]))
    #     print("std")
    #     print(np.std(grupos[i]))
#     plt.xlabel('Density')
    
    sns.set_style("white")
    
    plt.grid(False)
    plt.legend(legend)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    fig.text(0.5, 0, 'Density', ha='center')
#     plt.savefig("dist_5g_nog.pdf",bbox_inches="tight")
    plt.savefig(save,bbox_inches="tight")




from pos import *
import matplotlib.pyplot as plt
def grafica_grado_nodos(solucion):
    
    vecinos=convert_matrix_to_vecinos(solucion)
    grade={}
    for v in vecinos:
        g=len(v[1:])
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
    plt.show()
    
    
def draw_graph(solucion,fear):
    import networkx as nx
    G = nx.Graph()
    nodes=[]
    for i in range(len(solucion["vector_crime"])):
        nodes.append((i,solucion["vector_crime"][i]))
    for i in nodes:
        G.add_node(i[0],crime=i[1])

    edges=[]
    for i in range(len(solucion["matrix"])):
        for j in range(len(solucion["matrix"][i])):
            if solucion["matrix"][i][j] == 1 :
                edges.append((i,i+j+1))
    G.add_edges_from(edges)
    
    s=1000*fear+100
    color=[]
    for i in range(G.number_of_nodes()):
        if G.node[i]["crime"] == "A":
            color.append("G")
        elif G.node[i]["crime"] == "B":
            color.append("B")
        else:
            color.append("R")


    nx.draw(G,with_labels=True,node_size=s,node_color=color,alpha=0.7,font_color="w")
    plt.show()
    return G

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
    plt.show()

    """
    Grado de Centralidad de cada nodo
    """

    plt.plot(centrality.degree_centrality(G).values())
    plt.title("Grado de centralidad")
    plt.xlabel("Nodo")
    plt.ylabel("Centralidad")
    plt.show()


    """
    Calcular el coeficiente de agrupamiento para nodos
    """

    plt.plot(cluster.clustering(G).values())
    plt.title("coeficiente de agrupamiento")
    plt.xlabel("Nodo")
    plt.show()

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
    plt.show()

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
    plt.show()

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

import sys, os, shutil
from pos import *
import matplotlib as mpl
mpl.use('Agg')


nodos=int(sys.argv[1])
crimen=int(sys.argv[2])
tipo_objetivo=str(sys.argv[3])
folder=sys.argv[4]
total_generaciones=int(sys.argv[5])
n_estados=int(sys.argv[6])
   

#nodos=100
#crimen=6
if crimen == 2 :
    lamda={"A":0,"B":0.05}
elif crimen == 3 :
    lamda={"A":0,"B":0.05,"C":0.5}
elif crimen == 6:
    lamda={"A":0,"B":0.025,"C":0.05,"D":0.15,"E":0.25,"F":0.5}
else:
    lamda={"A":0,"B":0.025,"C":0.05,"D":0.15,"E":0.25,"F":0.5}


min_porcent=0.15
#tipo_objetivo="h"

if not os.path.exists(folder):
    os.mkdir(folder)
    
archivo = open(os.path.join(folder,"solucion_"+str(tipo_objetivo)+".txt"),'a')

archivo.write("# nodos: "+ str(nodos)+ '\n')
archivo.write("# grupos de crimen: "+ str(crimen)+ '\n')
archivo.write("funcion objectivo: "+ str(tipo_objetivo)+ '\n')
archivo.write("# de cromosomas por generacion: "+ str(n_estados)+ '\n')
archivo.write("-----------------------------------------------------------------"+'\n')
#Inicializacion de la poblacion
P=inicializacion(n_estados=n_estados,crimen=crimen,min_porcent=min_porcent,size=nodos)
#Estado inicial miedo al crimen para cada nodo
s0=np.random.rand(nodos)

archivo.write("Media miedo inicial: "+ str(np.mean(s0))+ '\n')

#total_generaciones=50
#### for generaciones

for i in range(total_generaciones):
    
    archivo.write("-----------------------------------------------------------------------"+ '\n')
    archivo.write("Generacion "+ str(i)+ '\n')
    print("Generacion "+ str(i)+"/"+str(total_generaciones)+ '\n')
    
    puntaje_generacion=[]

    
    for estado in P:
        puntaje_generacion.append(funcion_objetivo(estado,s0,tipo_objetivo,lamda))
        
    best_cromosome_generation=np.argsort(puntaje_generacion)[-1]
    
    
#     print("Mejor de la Generacion:")
#     print(P[best_cromosome_generation])
    
    media_crimen=plot(convert_matrix_to_vecinos(P[best_cromosome_generation]),s0,lamda)
    
    
    archivo.write("Media de crimen:" + str(media_crimen)+ '\n')
    
    homo=homofilia(convert_matrix_to_vecinos(P[best_cromosome_generation]))
    
    archivo.write("Homofilia: "+str(homo)+ '\n')
    
    archivo.write("Puntaje Generacion "+str(np.mean(puntaje_generacion))+ '\n')


    #funcion densidad de probabilidad para muestrear estados de la poblacion actual que depende de su desempeo
    fdp=seleccion(Poblacion=P,s=s0,tipo=tipo_objetivo,lamda=lamda)

    #Hijos de la poblacion actual
    Nueva_Generacion=[]

    # Combinacion 
    while len(Nueva_Generacion) != len(P):
        padre1=sample(Poblacion=P,fdp=fdp)
        padre2=sample(Poblacion=P,fdp=fdp)
        hijos=combinacion(state1=padre1,state2=padre2)
        contador=0
        while (validar(state=hijos[0],crimen=crimen,min_porcent=min_porcent) and validar(state=hijos[1],crimen=crimen,min_porcent=min_porcent)) == False:
            if contador == 200:
                hijos=[padre1,padre2]
                break
            hijos=combinacion(state1=padre1,state2=padre2)
            contador+=1
        Nueva_Generacion+=hijos

    # Mutacion
    for i in range(len(Nueva_Generacion)):
        Nueva_Generacion[i]=mutacion(Nueva_Generacion[i],crimen)

    # Reemplazo

    total = P+Nueva_Generacion

    fo=[]
    for t in total:
        fo.append(funcion_objetivo(t,s0,tipo_objetivo,lamda))
    order=np.argsort(fo)[::-1]
    best=order[:len(P)]
    for i in range(len(P)):
        P[i]=total[best[i]]
        
        
archivo.write("////////////////////////////////////////////////////////////////"+ '\n')
puntaje_generacion=[]
for estado in P:
    puntaje_generacion.append(funcion_objetivo(estado,s0,tipo_objetivo,lamda))
best_cromosome_generation=np.argsort(puntaje_generacion)[-1]
# print("Mejor Solucion:")
# print(P[best_cromosome_generation])

media_crimen=plot(convert_matrix_to_vecinos(P[best_cromosome_generation]),s0,lamda)

archivo.write("Media de crimen:" + str(media_crimen)+ '\n')

homo=homofilia(convert_matrix_to_vecinos(P[best_cromosome_generation]))

archivo.write("Homofilia: "+str(homo)+ '\n')

archivo.close()

np.save(os.path.join(folder,"solucion_"+str(tipo_objetivo)),P[best_cromosome_generation])
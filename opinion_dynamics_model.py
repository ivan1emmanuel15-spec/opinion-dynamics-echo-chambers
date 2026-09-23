# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:31:57 2026

@author: Ivan1
"""
#%reset -f
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
import numpy as np
from scipy.stats import skew, kurtosis
from matplotlib.ticker import MultipleLocator

#############USUARIOS Y PERIODOS#############
n=1000
T=20
tau=0.5

############CREACION DEL GRAFO###############
Grafo= nx.barabasi_albert_graph(n, m=3, seed=42) 
#Grafo = nx.erdos_renyi_graph(n, 0.009, directed=False, seed=42)
Grafodir= nx.DiGraph() 
Grafodir.add_nodes_from(Grafo.nodes())


############CREENCIAS Y PUBLICACION
np.random.seed(123)
s=np.random.uniform(-1,1,n)
sor=s.copy()

plt.figure(figsize=(8, 5), dpi=150)
plt.hist(s, bins=50, edgecolor='black')
plt.xlabel('Opinión', fontsize=13)
plt.ylabel('Frecuencia', fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.xlim(-1, 1)
plt.tight_layout()
plt.show()



#############TOLERANCIA DE LOS USUARIOS
t=np.random.uniform(tau,tau,n)
print("Tolerancia de los usuarios= ",t[0] )
print("---------------------------------------------------")
print("Nodos:", Grafo.number_of_nodes())
print("Aristas:", Grafo.number_of_edges())
print("Suma de grados:", sum(dict(Grafo.degree()).values()))
#########REGLA DE SEGUIMIENTO#################
degree=dict(Grafo.degree())

print("El nodo mas importante de la red original",max(degree, key=degree.get))

print("---------------------------------------------------")

sumadegrados=sum(degree.values())
print(sumadegrados)
grado={}                                                                                                 

for i in degree:
    grado[i]=degree[i]/sumadegrados

for u, v in Grafo.edges():
    if grado[u] <=grado[v]+min(grado.values()):
        Grafodir.add_edge(u, v)
        
    if grado[v]<=grado[u]+min(grado.values()):
        Grafodir.add_edge(v,u)


print("Aristas:", Grafodir.number_of_edges())
print("Suma de grados:", sum(dict(Grafodir.in_degree()).values()))


pos = nx.spring_layout(
    Grafodir,
    seed=42
)
############MATRIZ DE ADYACENCIA GRAFO DIRIGIDO###############
nodos = sorted(Grafodir.nodes())
A = nx.to_numpy_array(Grafodir, nodelist=nodos)

##############NODOS MAS IMPORTANTES gdir###################
followers = dict(Grafodir.in_degree())
ordenados = sorted(followers, key=followers.get, reverse=True)

maxim = ordenados[0]
segundo_maxim = ordenados[1]
tercer_maxim=ordenados[2]


print("El nodo mas importante despues de la regla se asignacion","\n",maxim)
#print("Grado del nodo mas importante ", followers[maxim])




###################HISTORIAL DE OPINIONES##################
S = np.zeros((n, T + 1))
S[:, 0] = s

###################VECTOR DE DIFUCION##################
r1=np.zeros(n)
r1[maxim]=1

r2=np.zeros(n)
r2[segundo_maxim]=1


r3=np.zeros(n)
r3[tercer_maxim]=0



###################MATRIZ DIAGONAL DE OPINIONES Y MATRIZ DE UNOS##################
U=np.ones((n,n))

P1=0.461954
np.random.uniform(np.maximum(s[maxim]-t[maxim],-1),np.minimum(s[maxim]+t[maxim],1))
P2=-0.3173158
np.random.uniform(np.maximum(s[segundo_maxim]-t[segundo_maxim],-1),np.minimum(s[segundo_maxim]+t[segundo_maxim],1))
#P3=-0.617086
#np.random.uniform(np.maximum(s[tercer_maxim]-t[tercer_maxim],-1),np.minimum(s[tercer_maxim]+t[tercer_maxim],1))

#publicacion en el intervalo de aleatorio
#P=np.random.uniform(np.maximum(s[ale]-t[ale],-1),np.minimum(s[ale]+t[ale],1)) 
sigma1=P1
sigma2=P2
#sigma3=P3
p1=np.full(n,P1)
p2=np.full(n,P2)
#p3=np.full(n,P3)

print("El valor de la publicacion es= ", sigma1)
print("El valor de la publicacion es= ", sigma2)
#print("El valor de la publicacion es= ", sigma3)




#print("--------------------------------------------------------")
# ==========================================
# FUNCIÓN PARA CALCULAR bNN
# ==========================================




conexiones=0
desconexiones=0
######################DINAMICA DE OPINION####################
enlaces=np.zeros(T+1)
enlaces[0]=np.sum(A)
print("Cantidad de enlaces iniciales= ", enlaces[0])

informados1=np.zeros(T+1)
informados1[0]=np.sum(r1)

informados2=np.zeros(T+1)
informados2[0]=np.sum(r2)


informados3=np.zeros(T+1)
informados3[0]=np.sum(r3)

Lconexiones=[]
Ldesconexiones=[]

A_hist = []
A_hist.append(A.copy())


def calcular_bNN(A_t, s_t):

    bNN = np.full(len(s_t), np.nan)

    for i in range(len(s_t)):

        # Nodos que i sigue en el periodo t
        vecinos = np.where(A_t[i] > 0)[0]

        if len(vecinos) > 0:
            bNN[i] = np.mean(s_t[vecinos])

    return bNN

def calcular_by(A_t, s_t):

    bNN = calcular_bNN(A_t, s_t)

    # Solo se consideran nodos que tienen vecinos
    mask = ~np.isnan(bNN)

    b = s_t[mask]
    bNN = bNN[mask]

    # Proyección sobre la primera dimensión
    # después de la rotación de 45 grados

    by = (b + bNN) / np.sqrt(2)

    return by

def calcular_BChom(A_t, s_t):

    by = calcular_by(A_t, s_t)

    n = len(by)

    # Se necesitan al menos 4 observaciones
    if n < 4:
        return np.nan

    # Asimetría
    g = skew(by)

    k = kurtosis(by, fisher=True)

    # Corrección por tamaño de muestra
    correccion = (
        3 * (n - 1)**2
        / ((n - 2) * (n - 3))
    )

    # Bimodality coefficient
    BC = (g**2 + 1) / (k + correccion)

    return BC


BChom = []


bc = calcular_BChom(A, s)

BChom.append(bc)

print("BC_hom inicial=", bc)

for a in range(T):
    B = np.diag(s)

    
    
    # =====================================================
    # DIFUSIÓN DE LA PUBLICACIÓN 1
    # =====================================================
    
    # Nodos expuestos por la red
    expuestos1 = np.minimum(A @ r1 + r1, 1)
    expuestos2 = np.minimum(A @ r2 + r2, 1)
    #expuestos3 = np.minimum(A @ r3 + r3, 1)
    
    # La publicación debe estar dentro del intervalo de tolerancia
    acepta1 = np.abs(p1 - s) < t
    acepta2 = np.abs(p2 - s) < t
    #acepta3 = np.abs(p3 - s) < t
    
    # Solo se informan los expuestos que aceptan la publicación
    
    r1 = np.where((expuestos1 == 1) & acepta1, 1, r1)
    r2 = np.where((expuestos2 == 1) & acepta2, 1, r2)
    #r3 = np.where((expuestos3 == 1) & acepta3, 1, r3)
    
    informados1[a+1] = np.sum(r1)
    informados2[a+1] = np.sum(r2)
    #informados3[a+1] = np.sum(r3)
    
    
    # Solo los informados que aceptan actualizan opinión
    mask1 = (r1 == 1) & acepta1
    mask2 = (r2 == 1) & acepta2
    #mask3 = (r3 == 1) & acepta3
    
    
   
    s_old = s.copy()
    s_new = s_old.copy()
    
    s_new[mask1] += (
        p1[mask1] - s_old[mask1]
    ) * np.abs(p1[mask1])
    s = np.clip(s_new, -1, 1)
    s_new[mask2] += (
        p2[mask2] - s_old[mask2]
    ) * np.abs(p2[mask2])
    s = np.clip(s_new, -1, 1)
# =============================================================================
#     s_new[mask3] += (
#         p3[mask3] - s_old[mask3]
#     ) * np.abs(p3[mask3])
#     
# =============================================================================
    s = np.clip(s_new, -1, 1)
    
    
    A_new=A.copy()
    B=np.diag(s)
    des = A @ B - B @ A
    con=B@U-U@B
    
    D_t = np.sum(np.abs(des)) / np.sum(A)
    
    #print("Distancia promedio= ",D_t)
    # Desconexió
    for i in range(n):
        for j in range(n):
            if abs(des[i,j]) > t[i]:
                A_new[i,j] = 0
                desconexiones=desconexiones+1
    #G_new = nx.from_numpy_array(A_new, create_using=nx.DiGraph())
    #gradon = np.array([G_new.in_degree(i) for i in range(n)])
    
        
    
    # Conexion
    # Conexión por recomendación social e ideología
    for i in range(n):
        
        for j in range(n):

            # evitar autoconexión
            if i == j:
                continue

            # j debe ser alguien que i sigue
            if A_new[i, j] == 0:
                continue

            # candidatos k: i no sigue a k, pero j sí sigue a k
            candidatos = np.where(A_new[i] - A_new[j] == -1)[0]

            for k in candidatos:

                # evitar que i se siga a sí mismo
                if i == k:
                    continue

                # condición ideológica entre i y k
                if abs(con[i, k]) < t[i]:
                    A_new[i, k] = 1
                    conexiones=conexiones+1                    
        
    Lconexiones.append(conexiones)
    Ldesconexiones.append(desconexiones)
       
    
    A = A_new.copy()
    enlaces[a+1] = np.sum(A)
    A_hist.append(A.copy())
    S[:,a+1] = s


    bc = calcular_BChom(A, s)

    BChom.append(bc)

    
    B=np.diag(s)
    des = A @ B - B @ A
    con=B@U-U@B
    
    D_t = np.sum(np.abs(des)) / np.sum(A)
    #print("Distancia promedio= ",D_t)
    if np.max(np.abs(S[:, a+1] - S[:, a])) < 0.001:
        #print("El sistema convergió en el periodo", a+1)
        tcon=a+1
        break
    else:
        tcon=a+1
Afin = A.copy()
print("BC_hom final =", bc)
G_fin = nx.from_numpy_array(Afin, create_using=nx.DiGraph())

print("Cantidad de enlaces finales= ", enlaces[tcon])


print("Cantidad de conexiones= ", conexiones )

print("Cantidad de desconexiones=",desconexiones )


#######################################################

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(BChom) + 1),
    BChom,
    label=r'$BC_{hom}$'
)

plt.axhline(
    5/9,
    linestyle='--',
    label=r'Umbral $5/9$'
)

plt.xlabel('Periodo')
plt.ylabel(r'$BC_{hom}$')

plt.legend()

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\BChome.pdf",
    bbox_inches='tight'
)



plt.tight_layout()
plt.show()









###############################################################

plt.rcParams.update({
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,

    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,

    'lines.linewidth': 1.5,

    # Texto y gráficos vectoriales
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
#########################################################################
plt.tight_layout()

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\conexiones_desconexiones.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\conexiones_desconexiones.pdf",
    bbox_inches='tight'
)

plt.show()
plt.close()
#########################################################################
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(Lconexiones) + 1),
    Lconexiones,
    label='Conexiones'
)

plt.plot(
    range(1, len(Ldesconexiones) + 1),
    Ldesconexiones,
    label='Desconexiones'
)

plt.xlabel('Periodo')
plt.ylabel('Número de enlaces')

plt.xlim(1, len(Lconexiones))

plt.legend()

plt.tight_layout()

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\conexiones_desconexiones.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\conexiones_desconexiones.pdf",
    bbox_inches='tight'
)

plt.show()
plt.close()
#################################################################################

plt.figure(figsize=(8, 5))

plt.plot(
    range(tcon),
    informados1[:tcon],
    label='Publicación 1'
)

plt.plot(
    range(tcon),
    informados2[:tcon],
    label='Publicación 2'
)

plt.plot(
    range(tcon),
    informados3[:tcon],
    label='Publicación 3'
)

plt.xlabel('Periodo')
plt.ylabel('Número de informados')

plt.xlim(0, tcon)

plt.ylim(
    0,
    max(
        informados1.max(),
        informados2.max(),
        informados3.max()
    )
)

plt.gca().yaxis.set_major_locator(MultipleLocator(100))

plt.legend()

plt.tight_layout()

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\informados.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\informados.pdf",
    bbox_inches='tight'
)

plt.show()
plt.close()

###########################################################################

plt.figure(figsize=(8, 5))

for i in range(n):

    plt.plot(
        range(tcon + 1),
        S[i, :tcon + 1],
        alpha=0.3
    )

plt.xlabel('Periodo')
plt.ylabel('Opinión')

plt.xlim(0, tcon)
plt.ylim(-1, 1)

plt.gca().yaxis.set_major_locator(
    MultipleLocator(0.1)
)

plt.tight_layout()

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\grafica_opiniones.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\grafica_opiniones.pdf",
    bbox_inches='tight'
)

plt.show()
plt.close()
################################################################################

plt.figure(figsize=(8, 5))

plt.hist(
    s,
    bins=50,
    edgecolor='black'
)

plt.xlabel('Opinión')
plt.ylabel('Frecuencia')

plt.gca().xaxis.set_major_locator(
    MultipleLocator(0.1)
)

plt.xlim(-1, 1)

plt.tight_layout()

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\grafica_hist.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\grafica_hist.pdf",
    bbox_inches='tight'
)

plt.show()
plt.close()

bNN_inicial = calcular_bNN(A_hist[0], S[:, 0]) 
bNN_final = calcular_bNN(A_hist[tcon], S[:, tcon]) # Eliminar NaN mask_inicial = ~np.isnan(bNN_inicial) mask_final = ~np.isnan(bNN_final)

mask_inicial = ~np.isnan(bNN_inicial) 
mask_final = ~np.isnan(bNN_final)

############################################################################

plt.figure(figsize=(7, 6))

plt.scatter(
    S[mask_inicial, 0],
    bNN_inicial[mask_inicial],
    s=25,
    alpha=0.5
)

plt.plot(
    [-1, 1],
    [-1, 1],
    '--',
    linewidth=1
)

plt.xlabel(r'Opinión del nodo $b_i$')
plt.ylabel(r'$bNN_i$')

plt.title('Estado inicial')

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\bNNvsS.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\bNNvsS.pdf",
    bbox_inches='tight'
)

plt.show()
plt.close()

############################################################################    

plt.figure(figsize=(7, 6))

plt.scatter(
    S[mask_final, tcon],
    bNN_final[mask_final],
    s=25,
    alpha=0.5
)

plt.plot(
    [-1, 1],
    [-1, 1],
    '--',
    linewidth=1
)

plt.xlabel(r'Opinión del nodo $b_i$')
plt.ylabel(r'$bNN_i$')

plt.title('Estado final')

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\bNNvsSfinal.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    r"C:\Users\Ivan1\OneDrive\Desktop\Tesis\Ejemplos para github\bNNvsSfinal.pdf",
    bbox_inches='tight'
)

plt.show()
plt.close()

#############################################################################






##############################################################
sigma=-1
epsilon=0.01
sigmasadmisibles=set()

todoslosconuntos=[]
while sigma<=1:
    nodosencadasigma=[]
    for u,v in G_fin.edges:
        if np.abs(s[v]-sigma)<epsilon:
            if np.abs(s[u]-sigma)<epsilon:
                sigmasadmisibles.add(sigma)
                nodosencadasigma.append((u,v))
    todoslosconuntos.append(nodosencadasigma)
    sigma +=0.01

todoslosconjuntos = [i for i in todoslosconuntos if i != []]

#print("Conunto de sigmas admisibles \n", sorted(sigmasadmisibles))
#print("Conjuntos aristas que aceptan a un sigma",todoslosconjuntos)
print("-"*50)

conjuntos_unicos = []
for conjunto in todoslosconjuntos:
    if conjunto not in conjuntos_unicos:
        conjuntos_unicos.append(conjunto)

conjuntos_ordenados = sorted(conjuntos_unicos, key=len, reverse=True)

conjuntosunicosord = []

for conjunto in conjuntos_ordenados:

    # Nodos del conjunto actual
    nodos_actual = set()

    for u, v in conjunto:
        nodos_actual.add(u)
        nodos_actual.add(v)

    # Comprobar si comparte nodos con algún conjunto ya guardado
    repetido = False

    for conjunto_guardado in conjuntosunicosord:

        nodos_guardado = set()

        for u, v in conjunto_guardado:
            nodos_guardado.add(u)
            nodos_guardado.add(v)

        if nodos_actual & nodos_guardado:
            repetido = True
            break

    # Como están ordenados de mayor a menor,
    # si no se repite ningún nodo, conservamos el conjunto
    if not repetido:
        conjuntosunicosord.append(conjunto)    


for i in conjuntosunicosord:
    mas_grande=i

    nodosdelacabinamasgrande=set()
    for u,v in mas_grande:
        nodosdelacabinamasgrande.add(u)
        nodosdelacabinamasgrande.add(v)
    
    opinionesdelosnodosenlacamara=[]
    for i in nodosdelacabinamasgrande:
        opinionesdelosnodosenlacamara.append(s[i])
    
    print("El promedio de los nodos dentro de la cabina es= ", np.sum(opinionesdelosnodosenlacamara)/len(opinionesdelosnodosenlacamara))
    print("Cantidad de nodos en la cabina= ", len(nodosdelacabinamasgrande))
    
    G_camara = nx.Graph()
    G_camara.add_edges_from(mas_grande)
    
    pos = nx.spring_layout(G_camara, seed=42, k=2)
    
    cmap = LinearSegmentedColormap.from_list(
        'opinion',
        ['red', 'black', 'blue']
    )
    
    norm = Normalize(vmin=-1, vmax=1)
    
    colores = [cmap(norm(s[i])) for i in G_camara.nodes()]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    nx.draw(
        G_camara,
        pos,
        ax=ax,
        with_labels=True,
        node_color=colores,
        node_size=100,
        edge_color='gray'
    )
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Opinión')
    
    plt.tight_layout()
    plt.show()







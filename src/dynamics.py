import numpy as np

from scipy.stats import skew, kurtosis


def calcular_bNN(A_t, s_t):
    """
    Calcula la opinión promedio de los vecinos
    que cada nodo sigue.
    """

    bNN = np.full(len(s_t), np.nan)

    for i in range(len(s_t)):

        vecinos = np.where(A_t[i] > 0)[0]

        if len(vecinos) > 0:
            bNN[i] = np.mean(s_t[vecinos])

    return bNN


def calcular_by(A_t, s_t):
    """
    Calcula la proyección sobre la primera dimensión
    después de una rotación de 45 grados.
    """

    bNN = calcular_bNN(A_t, s_t)

    mask = ~np.isnan(bNN)

    b = s_t[mask]
    bNN = bNN[mask]

    by = (b + bNN) / np.sqrt(2)

    return by


def calcular_BChom(A_t, s_t):
    """
    Calcula el coeficiente de bimodalidad
    utilizado como medida global de homofilia.
    """

    by = calcular_by(A_t, s_t)

    n = len(by)

    if n < 4:
        return np.nan

    g = skew(by)
    k = kurtosis(by, fisher=True)

    correccion = (
        3 * (n - 1)**2
        / ((n - 2) * (n - 3))
    )

    BC = (g**2 + 1) / (k + correccion)

    return BC

def inicializar_dinamica(
    A,
    b,
    t,
    maxim,
    segundo_maxim,
    tercer_maxim,
    T,
    seed=123
):
    """
    Inicializa opiniones, publicaciones y variables
    necesarias para la dinámica.
    """

    n = len(b)

    np.random.seed(seed)

    # Historial de opiniones
    S = np.zeros((n, T + 1))
    S[:, 0] = b

    # Vectores de difusión
    r1 = np.zeros(n)
    r2 = np.zeros(n)
    r3 = np.zeros(n)

    r1[maxim] = 1
    r2[segundo_maxim] = 1
    r3[tercer_maxim] = 1

    # Valores de las publicaciones
    sigma1 = np.random.uniform(
        np.maximum(
            b[maxim] - t[maxim],
            -1
        ),
        np.minimum(
            b[maxim] + t[maxim],
            1
        )
    )

    sigma2 = np.random.uniform(
        np.maximum(
            b[segundo_maxim] - t[segundo_maxim],
            -1
        ),
        np.minimum(
            b[segundo_maxim] + t[segundo_maxim],
            1
        )
    )

    sigma3 = np.random.uniform(
        np.maximum(
            b[tercer_maxim] - t[tercer_maxim],
            -1
        ),
        np.minimum(
            b[tercer_maxim] + t[tercer_maxim],
            1
        )
    )

    # Publicaciones para todos los nodos
    p1 = np.full(n, sigma1)
    p2 = np.full(n, sigma2)
    p3 = np.full(n, sigma3)

    # Número de informados
    informados1 = np.zeros(T + 1)
    informados2 = np.zeros(T + 1)
    informados3 = np.zeros(T + 1)

    informados1[0] = np.sum(r1)
    informados2[0] = np.sum(r2)
    informados3[0] = np.sum(r3)

    # Historial de redes
    A_hist = [A.copy()]

    # Enlaces
    enlaces = np.zeros(T + 1)
    enlaces[0] = np.sum(A)

    # Conexiones y desconexiones
    conexiones = 0
    desconexiones = 0

    Lconexiones = []
    Ldesconexiones = []

    # Matriz auxiliar
    U = np.ones((n, n))

    # BC_hom inicial
    bc = calcular_BChom(A, b)
    BChom = [bc]

    return {
        "S": S,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "informados1": informados1,
        "informados2": informados2,
        "informados3": informados3,
        "A_hist": A_hist,
        "enlaces": enlaces,
        "conexiones": conexiones,
        "desconexiones": desconexiones,
        "Lconexiones": Lconexiones,
        "Ldesconexiones": Ldesconexiones,
        "U": U,
        "BChom": BChom
    }

def ejecutar_dinamica(A, b, t, T, resultados):
    """
    Ejecuta la dinámica de difusión, opinión y
    evolución de la red.
    """

    n = len(b)

    # Recuperar variables iniciales
    S = resultados["S"]

    r1 = resultados["r1"]
    r2 = resultados["r2"]
    r3 = resultados["r3"]

    p1 = resultados["p1"]
    p2 = resultados["p2"]
    p3 = resultados["p3"]

    informados1 = resultados["informados1"]
    informados2 = resultados["informados2"]
    informados3 = resultados["informados3"]

    A_hist = resultados["A_hist"]

    enlaces = resultados["enlaces"]

    conexiones = resultados["conexiones"]
    desconexiones = resultados["desconexiones"]

    Lconexiones = resultados["Lconexiones"]
    Ldesconexiones = resultados["Ldesconexiones"]

    U = resultados["U"]

    BChom = resultados["BChom"]

    # Periodo de convergencia
    tcon = T

    # --------------------------------------------------
    # DINÁMICA TEMPORAL
    # --------------------------------------------------

    for a in range(T):

        # ==================================================
        # DIFUSIÓN DE LAS PUBLICACIONES
        # ==================================================

        expuestos1 = np.minimum(A @ r1 + r1, 1)
        expuestos2 = np.minimum(A @ r2 + r2, 1)
        expuestos3 = np.minimum(A @ r3 + r3, 1)

        # Aceptación según la tolerancia
        acepta1 = np.abs(p1 - b) < t
        acepta2 = np.abs(p2 - b) < t
        acepta3 = np.abs(p3 - b) < t

        # Actualización de informados
        r1 = np.where(
            (expuestos1 == 1) & acepta1,
            1,
            r1
        )

        r2 = np.where(
            (expuestos2 == 1) & acepta2,
            1,
            r2
        )

        r3 = np.where(
            (expuestos3 == 1) & acepta3,
            1,
            r3
        )

        informados1[a + 1] = np.sum(r1)
        informados2[a + 1] = np.sum(r2)
        informados3[a + 1] = np.sum(r3)

        # ==================================================
        # ACTUALIZACIÓN DE OPINIONES
        # ==================================================

        mask1 = (r1 == 1) & acepta1
        mask2 = (r2 == 1) & acepta2
        mask3 = (r3 == 1) & acepta3

        b_old = b.copy()
        b_new = b_old.copy()

        b_new[mask1] += (
            p1[mask1] - b_old[mask1]
        ) * np.abs(p1[mask1])

        b = np.clip(b_new, -1, 1)

        b_new[mask2] += (
            p2[mask2] - b_old[mask2]
        ) * np.abs(p2[mask2])

        b = np.clip(b_new, -1, 1)

        b_new[mask3] += (
            p3[mask3] - b_old[mask3]
        ) * np.abs(p3[mask3])

        b = np.clip(b_new, -1, 1)

        # ==================================================
        # EVOLUCIÓN DE LA RED
        # ==================================================

        A_new = A.copy()

        B = np.diag(b)

        des = A @ B - B @ A
        con = B @ U - U @ B

        # ==================================================
        # DESCONEXIONES
        # ==================================================

        for i in range(n):

            for j in range(n):

                if abs(des[i, j]) > t[i]:

                    A_new[i, j] = 0
                    desconexiones += 1

        # ==================================================
        # CONEXIONES
        # ==================================================

        for i in range(n):

            for j in range(n):

                if i == j:
                    continue

                if A_new[i, j] == 0:
                    continue

                candidatos = np.where(
                    A_new[i] - A_new[j] == -1
                )[0]

                for k in candidatos:

                    if i == k:
                        continue

                    if abs(con[i, k]) < t[i]:

                        A_new[i, k] = 1
                        conexiones += 1

        # Guardar cambios
        Lconexiones.append(conexiones)
        Ldesconexiones.append(desconexiones)

        A = A_new.copy()

        enlaces[a + 1] = np.sum(A)

        A_hist.append(A.copy())

        S[:, a + 1] = b

        # ==================================================
        # BC_hom
        # ==================================================

        bc = calcular_BChom(A, b)

        BChom.append(bc)

        # ==================================================
        # CRITERIO DE CONVERGENCIA
        # ==================================================

        if np.max(
            np.abs(
                S[:, a + 1] - S[:, a]
            )
        ) < 0.001:

            tcon = a + 1
            break

        tcon = a + 1

    # --------------------------------------------------
    # ACTUALIZAR RESULTADOS
    # --------------------------------------------------

    resultados["S"] = S

    resultados["r1"] = r1
    resultados["r2"] = r2
    resultados["r3"] = r3

    resultados["informados1"] = informados1
    resultados["informados2"] = informados2
    resultados["informados3"] = informados3

    resultados["A_hist"] = A_hist

    resultados["enlaces"] = enlaces

    resultados["conexiones"] = conexiones
    resultados["desconexiones"] = desconexiones

    resultados["Lconexiones"] = Lconexiones
    resultados["Ldesconexiones"] = Ldesconexiones

    resultados["BChom"] = BChom

    resultados["A_final"] = A
    resultados["b_final"] = b
    resultados["tcon"] = tcon

    return resultados
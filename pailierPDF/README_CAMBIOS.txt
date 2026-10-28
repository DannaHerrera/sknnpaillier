VERSIÓN: UN SOLO CLOUDB + KD-TREE POR HOJAS
==============================================

ARQUITECTURA FINAL DE ESTA REVISIÓN
-----------------------------------
Data Owner / zona confiable:
    KeyM.py
    SecureDataT.py
    KDTree.py (construcción en plaintext)

Cloud:
    CloudB.py     <- único cloud y procesador de consultas

Validación:
    TrustedR.py

Soporte:
    Config.py
    query.py
    Result.py
    Main.py

SE ELIMINAN
-----------
    QP.py
    SecureHelper.py

CloudB.py reemplaza a QP.py.

¿DÓNDE FUERON LAS FUNCIONES DE SecureHelper.py?
------------------------------------------------
No se eliminó la necesidad matemática de esas operaciones.

Paillier NO permite directamente:
    1) E(a) y E(b) -> E(a*b)
    2) comparar E(a) <= E(b)

Por ello las primitivas controladas se movieron a KeyM.py:

    KeyManagementModule.secure_multiply_masked()
    KeyManagementModule.secure_compare()

La PrivateKey continúa encapsulada en KeyM.

CloudB:
    - genera las máscaras;
    - opera homomórficamente;
    - solicita únicamente la operación controlada necesaria;
    - recibe un ciphertext o un booleano;
    - nunca recibe PrivateKey.

KD-TREE
-------
El KD-tree se construye ANTES del cifrado/outsourcing.

Cada hoja enviada a CloudB contiene:

    EncryptedKDLeaf(
        leaf_id,
        E(lower_bounds),
        E(upper_bounds),
        record_ids
    )

Ya no se usa durante la consulta:

    left
    right
    axis
    split

como mecanismo de recorrido jerárquico.

PROCESAMIENTO DE CONSULTA EN CloudB
-----------------------------------
FASE 1 - Encrypted KD-tree search
    secure_point_enclosure()

    Comprueba conceptualmente:
        lb_j <= q_j <= ub_j

FASE 2 - kNN retrieval
    secure_squared_distance()

    Calcula:
        E(sum_j (x_j - q_j)^2)

    únicamente para los registros de las hojas candidatas.

FASE 3 - Result verification
    encrypted_shortest_point()
    secure_region_min_distance()

    Para cada hoja restante:
        sp = punto de la región más cercano a q

        si:
            dist^2(q, sp) <= dist^2(q, kth)
        entonces:
            procesar hoja
        si no:
            podar hoja

EJECUCIÓN
---------
python Main.py \
    --source synthetic \
    --db-rows 100 \
    --queries 5 \
    --k 3 \
    --key-bits 512 \
    --leaf-capacity 16

OBSERVACIÓN DE SEGURIDAD
------------------------
Esta implementación tiene UN solo cloud, como se solicitó.

Sin embargo, las comparaciones seguras y la multiplicación de dos valores
cifrados no pueden implementarse únicamente con Paillier aditivo sin alguna
forma de operación confiable/interactiva.

En este prototipo esa responsabilidad queda en KeyM, que NO es un segundo
cloud: es el módulo confiable que mantiene la llave privada.

CloudB aprende los booleanos de SPE/verificación, por lo que el patrón de
acceso no es completamente oblivious.

Esto debe distinguirse de una implementación estricta de protocolos
multi-servidor orientados a ocultar también el access pattern.

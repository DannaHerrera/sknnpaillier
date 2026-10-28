from __future__ import annotations

import secrets
from typing import Sequence

from KDTree import EncryptedKDTree
from SecureDataT import SecureDataTransformationModule
from SecureHelper import TrustedPaillierHelper


class SecureKNNProcessingModule:
    """
    Query Processor para SKNN con Paillier.

    Modos de búsqueda:

    1) kd-tree
       - La BD se indexa previamente como KD-tree.
       - Distancias y scores permanecen cifrados.
       - El top-k candidato se mantiene cifrado.
       - El helper revela a QP únicamente decisiones booleanas
         necesarias para recorrer/podar el árbol.
       - Esto filtra el patrón de acceso, no los valores originales.

    2) bitonic
       - Calcula todas las distancias.
       - Ordena todos los scores mediante red bitónica.
       - Mantiene un patrón de acceso independiente de los datos.

    QP nunca recibe KeyManagementModule y no puede ejecutar decrypt().
    """

    def __init__(
        self,
        transformer: SecureDataTransformationModule,
        helper: TrustedPaillierHelper,
        distance_upper_bound: int,
        coordinate_offset: int = 0,
        coordinate_upper_bound: int = 0,
        ranking_mode: str = "kd-tree",
    ) -> None:

        if distance_upper_bound < 0:
            raise ValueError(
                "distance_upper_bound debe ser >= 0."
            )

        if coordinate_upper_bound < 0:
            raise ValueError(
                "coordinate_upper_bound debe ser >= 0."
            )

        if ranking_mode not in {"kd-tree", "bitonic"}:
            raise ValueError(
                "ranking_mode debe ser 'kd-tree' o 'bitonic'."
            )

        self.transformer = transformer
        self.helper = helper
        self.distance_upper_bound = int(
            distance_upper_bound
        )
        self.coordinate_offset = int(
            coordinate_offset
        )
        self.coordinate_upper_bound = int(
            coordinate_upper_bound
        )
        self.ranking_mode = ranking_mode

        self.encrypted_database: (
            Sequence[Sequence[int]] | None
        ) = None

        self.encrypted_kd_tree: EncryptedKDTree | None = None

        self.last_kd_nodes_visited = 0
        self.last_kd_pruned_branches = 0

    # ========================================================
    # RECIBIR BASE DE DATOS CIFRADA
    # ========================================================

    def set_database(
        self,
        encrypted_database: Sequence[
            Sequence[int]
        ],
    ) -> None:

        self.encrypted_database = encrypted_database

    # ========================================================
    # RECIBIR KD-TREE CIFRADO
    # ========================================================

    def set_kd_tree(
        self,
        encrypted_kd_tree: EncryptedKDTree,
    ) -> None:

        self.encrypted_kd_tree = encrypted_kd_tree

    # ========================================================
    # PROCESAR CONSULTA
    # ========================================================

    def process_query(
        self,
        encrypted_query: Sequence[int],
        k: int,
    ) -> list[int]:
        """
        Retorna únicamente k scores cifrados y ordenados:

            score = distancia² * (N + 1) + record_id

        QueryModule descifra esos scores al final.
        """

        if self.encrypted_database is None:
            raise RuntimeError(
                "QP no ha recibido la base de datos cifrada."
            )

        database_size = len(self.encrypted_database)

        if not 1 <= k <= database_size:
            raise ValueError(
                "k debe cumplir 1 <= k <= número de registros."
            )

        if self.ranking_mode == "kd-tree":
            return self.secure_kd_tree_top_k(
                encrypted_query,
                k,
            )

        encrypted_distances = (
            self.calculate_encrypted_distances(
                self.encrypted_database,
                encrypted_query,
            )
        )

        encrypted_scores = self.build_encrypted_scores(
            encrypted_distances
        )

        sorted_encrypted_scores = self.secure_rank(
            encrypted_scores
        )

        return list(sorted_encrypted_scores[:k])

    # ========================================================
    # SUMA HOMOMÓRFICA
    # ========================================================

    def encrypted_add(
        self,
        encrypted_a: int,
        encrypted_b: int,
    ) -> int:

        n2 = self.transformer.public_key.n2

        return (
            int(encrypted_a)
            * int(encrypted_b)
        ) % n2

    # ========================================================
    # MULTIPLICACIÓN POR ESCALAR
    # ========================================================

    def encrypted_scalar_multiply(
        self,
        ciphertext: int,
        scalar: int,
    ) -> int:

        public_key = self.transformer.public_key

        return pow(
            int(ciphertext),
            int(scalar) % public_key.n,
            public_key.n2,
        )

    # ========================================================
    # NEGACIÓN HOMOMÓRFICA
    # ========================================================

    def encrypted_negate(
        self,
        ciphertext: int,
    ) -> int:

        n = self.transformer.public_key.n

        return self.encrypted_scalar_multiply(
            ciphertext,
            n - 1,
        )

    # ========================================================
    # MULTIPLICACIÓN SEGURA
    # ========================================================

    def secure_multiply(
        self,
        encrypted_a: int,
        encrypted_b: int,
    ) -> int:
        """
        E(a), E(b) -> E(a*b) mediante máscaras y helper confiable.
        QP no descifra a ni b.
        """

        n = self.transformer.public_key.n

        mask_a = secrets.randbelow(n)
        mask_b = secrets.randbelow(n)

        encrypted_mask_a = self.transformer.encrypt(
            mask_a
        )

        encrypted_mask_b = self.transformer.encrypt(
            mask_b
        )

        encrypted_a_masked = self.encrypted_add(
            encrypted_a,
            encrypted_mask_a,
        )

        encrypted_b_masked = self.encrypted_add(
            encrypted_b,
            encrypted_mask_b,
        )

        encrypted_masked_product = (
            self.helper.multiply_masked_values(
                encrypted_a_masked,
                encrypted_b_masked,
            )
        )

        correction_a = self.encrypted_scalar_multiply(
            encrypted_a,
            n - mask_b,
        )

        result = self.encrypted_add(
            encrypted_masked_product,
            correction_a,
        )

        correction_b = self.encrypted_scalar_multiply(
            encrypted_b,
            n - mask_a,
        )

        result = self.encrypted_add(
            result,
            correction_b,
        )

        masks_product = (
            mask_a * mask_b
        ) % n

        encrypted_masks_product = self.transformer.encrypt(
            masks_product
        )

        encrypted_negative_masks = self.encrypted_negate(
            encrypted_masks_product
        )

        return self.encrypted_add(
            result,
            encrypted_negative_masks,
        )

    # ========================================================
    # DIFERENCIA CIFRADA
    # ========================================================

    def encrypted_difference(
        self,
        encrypted_a: int,
        encrypted_b: int,
    ) -> int:

        return self.encrypted_add(
            encrypted_a,
            self.encrypted_negate(encrypted_b),
        )

    # ========================================================
    # DISTANCIA EUCLIDIANA CUADRADA
    # ========================================================

    def secure_squared_distance(
        self,
        encrypted_record: Sequence[int],
        encrypted_query: Sequence[int],
    ) -> int:

        if len(encrypted_record) != len(encrypted_query):
            raise ValueError(
                "El registro y la consulta deben tener "
                "la misma dimensión."
            )

        encrypted_distance = self.transformer.encrypt(0)

        for encrypted_x, encrypted_q in zip(
            encrypted_record,
            encrypted_query,
        ):

            encrypted_delta = self.encrypted_difference(
                encrypted_x,
                encrypted_q,
            )

            encrypted_square = self.secure_multiply(
                encrypted_delta,
                encrypted_delta,
            )

            encrypted_distance = self.encrypted_add(
                encrypted_distance,
                encrypted_square,
            )

        return encrypted_distance

    # ========================================================
    # DISTANCIA AL PLANO DE CORTE DEL KD-TREE
    # ========================================================

    def secure_axis_squared_distance(
        self,
        encrypted_query_value: int,
        encrypted_split_value: int,
    ) -> int:

        encrypted_delta = self.encrypted_difference(
            encrypted_query_value,
            encrypted_split_value,
        )

        return self.secure_multiply(
            encrypted_delta,
            encrypted_delta,
        )

    # ========================================================
    # DISTANCIAS DE TODA LA BD - MODO BITONIC
    # ========================================================

    def calculate_encrypted_distances(
        self,
        encrypted_database: Sequence[
            Sequence[int]
        ],
        encrypted_query: Sequence[int],
    ) -> list[int]:

        return [
            self.secure_squared_distance(
                encrypted_record,
                encrypted_query,
            )
            for encrypted_record
            in encrypted_database
        ]

    # ========================================================
    # SCORE CIFRADO
    # ========================================================

    def build_encrypted_score(
        self,
        encrypted_distance: int,
        encrypted_id: int,
    ) -> int:

        if self.encrypted_database is None:
            raise RuntimeError(
                "QP no ha recibido la base de datos cifrada."
            )

        base = len(self.encrypted_database) + 1

        encrypted_scaled_distance = (
            self.encrypted_scalar_multiply(
                encrypted_distance,
                base,
            )
        )

        return self.encrypted_add(
            encrypted_scaled_distance,
            encrypted_id,
        )

    def build_encrypted_scores(
        self,
        encrypted_distances: Sequence[int],
    ) -> list[int]:

        encrypted_scores: list[int] = []

        for record_id, encrypted_distance in enumerate(
            encrypted_distances
        ):
            encrypted_scores.append(
                self.build_encrypted_score(
                    encrypted_distance,
                    self.transformer.encrypt(record_id),
                )
            )

        return encrypted_scores

    # ========================================================
    # COTA DEL SCORE
    # ========================================================

    def _real_score_upper_bound(
        self,
    ) -> int:

        if self.encrypted_database is None:
            raise RuntimeError(
                "QP no ha recibido la base de datos cifrada."
            )

        database_size = len(self.encrypted_database)
        base = database_size + 1

        return (
            self.distance_upper_bound * base
            + (database_size - 1)
        )

    # ========================================================
    # ENMASCARAMIENTO AFÍN PARA COMPARACIONES
    # ========================================================

    def _affine_mask_pair(
        self,
        encrypted_left: int,
        encrypted_right: int,
        comparison_upper_bound: int,
    ) -> tuple[int, int]:
        """
        Construye la misma transformación creciente:

            T(x) = scale*x + offset

        para ambos ciphertexts, evitando wrap-around módulo n.
        """

        n = self.transformer.public_key.n
        safe_limit = (n // 2) - 1

        bound = max(1, int(comparison_upper_bound))

        max_scale = safe_limit // bound

        if max_scale < 1:
            raise OverflowError(
                "La cota de comparación es demasiado grande "
                "para el módulo Paillier actual."
            )

        practical_max_scale = min(
            max_scale,
            (1 << 64) - 1,
        )

        if practical_max_scale <= 1:
            scale = 1
        else:
            scale = (
                secrets.randbelow(
                    practical_max_scale - 1
                )
                + 2
            )

        scaled_bound = scale * bound
        max_offset = safe_limit - scaled_bound

        if max_offset < 0:
            raise OverflowError(
                "No existe espacio para el enmascaramiento "
                "de comparación."
            )

        offset = (
            secrets.randbelow(max_offset + 1)
            if max_offset > 0
            else 0
        )

        encrypted_offset = self.transformer.encrypt(
            offset
        )

        encrypted_left_scaled = (
            self.encrypted_scalar_multiply(
                encrypted_left,
                scale,
            )
        )

        encrypted_right_scaled = (
            self.encrypted_scalar_multiply(
                encrypted_right,
                scale,
            )
        )

        encrypted_left_masked = self.encrypted_add(
            encrypted_left_scaled,
            encrypted_offset,
        )

        encrypted_right_masked = self.encrypted_add(
            encrypted_right_scaled,
            encrypted_offset,
        )

        return (
            encrypted_left_masked,
            encrypted_right_masked,
        )

    # ========================================================
    # COMPARACIÓN QUE REVELA SOLO EL BOOLEANO A QP
    # ========================================================

    def secure_compare_revealed(
        self,
        encrypted_left: int,
        encrypted_right: int,
        comparison_upper_bound: int,
        less_or_equal: bool = True,
    ) -> bool:
        """
        Utilizado únicamente por el KD-tree para decidir qué rama
        recorrer o podar. QP aprende el resultado booleano, pero no
        los valores comparados.
        """

        (
            encrypted_left_masked,
            encrypted_right_masked,
        ) = self._affine_mask_pair(
            encrypted_left,
            encrypted_right,
            comparison_upper_bound,
        )

        return self.helper.compare_masked_values(
            encrypted_left_masked,
            encrypted_right_masked,
            less_or_equal=less_or_equal,
        )

    # ========================================================
    # COMPARE-AND-SWAP CIFRADO
    # ========================================================

    def secure_compare_swap(
        self,
        encrypted_left: int,
        encrypted_right: int,
        comparison_upper_bound: int,
        ascending: bool,
    ) -> tuple[int, int]:

        (
            encrypted_left_masked,
            encrypted_right_masked,
        ) = self._affine_mask_pair(
            encrypted_left,
            encrypted_right,
            comparison_upper_bound,
        )

        return self.helper.compare_and_swap_masked(
            encrypted_left_original=encrypted_left,
            encrypted_right_original=encrypted_right,
            encrypted_left_masked=encrypted_left_masked,
            encrypted_right_masked=encrypted_right_masked,
            ascending=ascending,
        )

    # ========================================================
    # SIGUIENTE POTENCIA DE DOS
    # ========================================================

    @staticmethod
    def _next_power_of_two(
        value: int,
    ) -> int:

        if value <= 1:
            return 1

        return 1 << (value - 1).bit_length()

    # ========================================================
    # RANKING CIFRADO BITÓNICO
    # ========================================================

    def secure_rank(
        self,
        encrypted_scores: Sequence[int],
    ) -> list[int]:

        if not encrypted_scores:
            return []

        values = list(encrypted_scores)
        original_size = len(values)
        target_size = self._next_power_of_two(
            original_size
        )

        real_upper_bound = self._real_score_upper_bound()

        for dummy_index in range(
            target_size - original_size
        ):
            dummy_score = (
                real_upper_bound
                + 1
                + dummy_index
            )

            values.append(
                self.transformer.encrypt(dummy_score)
            )

        comparison_upper_bound = (
            real_upper_bound
            + (target_size - original_size)
        )

        size = 2

        while size <= target_size:
            stride = size // 2

            while stride > 0:
                for index in range(target_size):
                    partner = index ^ stride

                    if partner <= index:
                        continue

                    ascending = (
                        (index & size) == 0
                    )

                    (
                        values[index],
                        values[partner],
                    ) = self.secure_compare_swap(
                        values[index],
                        values[partner],
                        comparison_upper_bound,
                        ascending,
                    )

                stride //= 2

            size *= 2

        return values[:original_size]

    # ========================================================
    # DESPLAZAR COORDENADA A DOMINIO NO NEGATIVO
    # ========================================================

    def _shift_coordinate(
        self,
        encrypted_value: int,
    ) -> int:

        if self.coordinate_offset == 0:
            return encrypted_value

        return self.encrypted_add(
            encrypted_value,
            self.transformer.encrypt(
                self.coordinate_offset
            ),
        )

    # ========================================================
    # DECIDIR RAMA KD-TREE
    # ========================================================

    def _query_goes_left(
        self,
        encrypted_query_value: int,
        encrypted_split_value: int,
    ) -> bool:

        if self.coordinate_upper_bound == 0:
            return True

        encrypted_query_shifted = self._shift_coordinate(
            encrypted_query_value
        )

        encrypted_split_shifted = self._shift_coordinate(
            encrypted_split_value
        )

        return self.secure_compare_revealed(
            encrypted_query_shifted,
            encrypted_split_shifted,
            self.coordinate_upper_bound,
            less_or_equal=True,
        )

    # ========================================================
    # INSERTAR CANDIDATO EN TOP-K CIFRADO
    # ========================================================

    def _insert_encrypted_candidate(
        self,
        candidates: list[int],
        encrypted_score: int,
        k: int,
    ) -> list[int]:
        """
        Inserta un score sin descifrarlo. Se ordenan como máximo
        k+1 scores y se conservan los k menores.
        """

        values = list(candidates)
        values.append(encrypted_score)

        ranked = self.secure_rank(values)

        if len(ranked) > k:
            ranked = ranked[:k]

        return ranked

    # ========================================================
    # BÚSQUEDA EXACTA KD-TREE ASISTIDA
    # ========================================================

    def secure_kd_tree_top_k(
        self,
        encrypted_query: Sequence[int],
        k: int,
    ) -> list[int]:
        """
        Búsqueda kNN exacta sobre un KD-tree cuyos puntos e IDs
        están cifrados.

        Privacidad:
        - QP no obtiene coordenadas, distancias, IDs ni scores.
        - Distancias y ranking top-k siguen cifrados.
        - Para aprovechar la poda del KD-tree, QP sí obtiene los
          booleanos de rama/poda. Por ello el patrón de acceso no
          es oblivious.
        """

        if self.encrypted_database is None:
            raise RuntimeError(
                "QP no ha recibido la base de datos cifrada."
            )

        if self.encrypted_kd_tree is None:
            raise RuntimeError(
                "QP no ha recibido el KD-tree cifrado."
            )

        tree = self.encrypted_kd_tree

        if len(encrypted_query) != tree.dimensions:
            raise ValueError(
                "La consulta y el KD-tree tienen distinta dimensión."
            )

        if tree.database_size != len(self.encrypted_database):
            raise ValueError(
                "El KD-tree y la BD cifrada tienen distinto tamaño."
            )

        self.last_kd_nodes_visited = 0
        self.last_kd_pruned_branches = 0

        candidates: list[int] = []
        base = tree.database_size + 1
        score_upper_bound = self._real_score_upper_bound()

        def search(node_index: int | None) -> None:
            nonlocal candidates

            if node_index is None:
                return

            node = tree.nodes[node_index]
            self.last_kd_nodes_visited += 1

            axis = node.axis

            # ------------------------------------------------
            # 1) Determinar rama cercana con comparación segura.
            # ------------------------------------------------

            goes_left = self._query_goes_left(
                encrypted_query[axis],
                node.encrypted_record[axis],
            )

            if goes_left:
                near_child = node.left
                far_child = node.right
            else:
                near_child = node.right
                far_child = node.left

            # ------------------------------------------------
            # 2) Evaluar distancia del nodo actual cifrada.
            # ------------------------------------------------

            encrypted_distance = self.secure_squared_distance(
                node.encrypted_record,
                encrypted_query,
            )

            encrypted_score = self.build_encrypted_score(
                encrypted_distance,
                node.encrypted_id,
            )

            candidates = self._insert_encrypted_candidate(
                candidates,
                encrypted_score,
                k,
            )

            # ------------------------------------------------
            # 3) Explorar primero la rama cercana.
            # ------------------------------------------------

            search(near_child)

            # ------------------------------------------------
            # 4) Si aún no hay k candidatos, no puede podarse.
            # ------------------------------------------------

            if far_child is None:
                return

            if len(candidates) < k:
                search(far_child)
                return

            # ------------------------------------------------
            # 5) Poda KD-tree.
            #
            # Si distancia_plano² <= peor_distancia_actual,
            # la rama lejana todavía puede contener un vecino.
            #
            # No se extrae peor_distancia. Se compara:
            #
            #   plane² * (N+1) <= worst_score
            #
            # donde worst_score está cifrado.
            # ------------------------------------------------

            encrypted_plane_distance = (
                self.secure_axis_squared_distance(
                    encrypted_query[axis],
                    node.encrypted_record[axis],
                )
            )

            encrypted_plane_scaled = (
                self.encrypted_scalar_multiply(
                    encrypted_plane_distance,
                    base,
                )
            )

            encrypted_worst_score = candidates[-1]

            must_visit_far = self.secure_compare_revealed(
                encrypted_plane_scaled,
                encrypted_worst_score,
                score_upper_bound,
                less_or_equal=True,
            )

            if must_visit_far:
                search(far_child)
            else:
                self.last_kd_pruned_branches += 1

        search(tree.root)

        # candidates ya está cifrado y ordenado ascendentemente.
        return list(candidates[:k])

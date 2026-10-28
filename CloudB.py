from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Sequence

import numpy as np

from KDTree import (
    EncryptedKDLeaf,
    EncryptedKDTree,
)
from KeyM import KeyManagementModule
from SecureDataT import (
    SecureDataTransformationModule,
)


@dataclass
class _Candidate:
    record_id: int
    encrypted_distance: int
    encrypted_score: int


class CloudBQueryProcessingModule:
    """
    ÚNICO CLOUD del sistema.

    CloudB procesa las consultas cifradas y ejecuta:

      1. Encrypted KD-tree search:
         SPE-like sobre las hojas.

      2. kNN retrieval:
         ESSED-like sobre los registros de las hojas candidatas.

      3. Result verification:
         shortest point + distancia mínima de región.

    SecureHelper.py ya no existe.

    Cuando una operación no puede realizarse únicamente con las
    propiedades homomórficas de Paillier, CloudB solicita una
    operación CONTROLADA a KeyM. CloudB no recibe PrivateKey.
    """

    def __init__(
        self,
        transformer: SecureDataTransformationModule,
        key_manager: KeyManagementModule,
    ) -> None:
        self.transformer = (
            transformer
        )

        self.key_manager = (
            key_manager
        )

        self.encrypted_database = None

        self.encrypted_kd_tree: (
            EncryptedKDTree | None
        ) = None

        self.last_query_stats: dict[
            str,
            int,
        ] = {}

    # ==========================================================
    # RECEPCIÓN DE BD E ÍNDICE
    # ==========================================================

    def set_database(
        self,
        encrypted_database: Sequence[
            Sequence[int]
        ],
    ) -> None:
        self.encrypted_database = (
            encrypted_database
        )

    def set_kd_tree(
        self,
        encrypted_kd_tree: EncryptedKDTree,
    ) -> None:
        self.encrypted_kd_tree = (
            encrypted_kd_tree
        )

    # ==========================================================
    # OPERACIONES HOMOMÓRFICAS PAILLIER
    # ==========================================================

    def encrypted_add(
        self,
        encrypted_a: int,
        encrypted_b: int,
    ) -> int:
        n2 = (
            self.transformer.public_key.n2
        )

        return (
            int(encrypted_a)
            * int(encrypted_b)
        ) % n2

    def encrypted_scalar_multiply(
        self,
        ciphertext: int,
        scalar: int,
    ) -> int:
        public_key = (
            self.transformer.public_key
        )

        return pow(
            int(ciphertext),
            int(scalar)
            % public_key.n,
            public_key.n2,
        )

    def encrypted_negate(
        self,
        ciphertext: int,
    ) -> int:
        n = (
            self.transformer.public_key.n
        )

        return (
            self.encrypted_scalar_multiply(
                ciphertext,
                n - 1,
            )
        )

    def encrypted_difference(
        self,
        encrypted_a: int,
        encrypted_b: int,
    ) -> int:
        return self.encrypted_add(
            encrypted_a,
            self.encrypted_negate(
                encrypted_b
            ),
        )

    # ==========================================================
    # MULTIPLICACIÓN SEGURA
    # ==========================================================

    def secure_multiply(
        self,
        encrypted_a: int,
        encrypted_b: int,
    ) -> int:
        """
        CloudB genera máscaras aleatorias.

        KeyM solo ve los valores enmascarados y devuelve el producto
        nuevamente cifrado. Después CloudB elimina los términos de
        las máscaras homomórficamente.
        """
        n = (
            self.transformer.public_key.n
        )

        mask_a = secrets.randbelow(
            n
        )

        mask_b = secrets.randbelow(
            n
        )

        encrypted_a_masked = (
            self.encrypted_add(
                encrypted_a,
                self.transformer.encrypt(
                    mask_a
                ),
            )
        )

        encrypted_b_masked = (
            self.encrypted_add(
                encrypted_b,
                self.transformer.encrypt(
                    mask_b
                ),
            )
        )

        encrypted_masked_product = (
            self.key_manager.secure_multiply_masked(
                encrypted_a_masked,
                encrypted_b_masked,
                self.transformer.encrypt,
            )
        )

        # - a*r_b
        result = self.encrypted_add(
            encrypted_masked_product,
            self.encrypted_scalar_multiply(
                encrypted_a,
                n - mask_b,
            ),
        )

        # - b*r_a
        result = self.encrypted_add(
            result,
            self.encrypted_scalar_multiply(
                encrypted_b,
                n - mask_a,
            ),
        )

        # - r_a*r_b
        encrypted_masks_product = (
            self.transformer.encrypt(
                (
                    mask_a * mask_b
                ) % n
            )
        )

        result = self.encrypted_add(
            result,
            self.encrypted_negate(
                encrypted_masks_product
            ),
        )

        return result

    # ==========================================================
    # ESSED-LIKE
    # ==========================================================

    def secure_squared_distance(
        self,
        encrypted_record: Sequence[int],
        encrypted_query: Sequence[int],
    ) -> int:
        if (
            len(encrypted_record)
            != len(encrypted_query)
        ):
            raise ValueError(
                "Registro y consulta deben tener "
                "la misma dimensión."
            )

        encrypted_distance = (
            self.transformer.encrypt(0)
        )

        for (
            encrypted_x,
            encrypted_q,
        ) in zip(
            encrypted_record,
            encrypted_query,
        ):
            encrypted_delta = (
                self.encrypted_difference(
                    encrypted_x,
                    encrypted_q,
                )
            )

            encrypted_square = (
                self.secure_multiply(
                    encrypted_delta,
                    encrypted_delta,
                )
            )

            encrypted_distance = (
                self.encrypted_add(
                    encrypted_distance,
                    encrypted_square,
                )
            )

        return encrypted_distance

    # ==========================================================
    # COMPARACIONES CONTROLADAS
    # ==========================================================

    def _secure_le(
        self,
        encrypted_left: int,
        encrypted_right: int,
        *,
        signed: bool,
    ) -> bool:
        return (
            self.key_manager.secure_compare(
                encrypted_left,
                encrypted_right,
                signed=signed,
                less_or_equal=True,
            )
        )

    def _secure_lt(
        self,
        encrypted_left: int,
        encrypted_right: int,
        *,
        signed: bool,
    ) -> bool:
        return (
            self.key_manager.secure_compare(
                encrypted_left,
                encrypted_right,
                signed=signed,
                less_or_equal=False,
            )
        )

    # ==========================================================
    # SPE-LIKE: q DENTRO DE UNA HOJA
    # ==========================================================

    def secure_point_enclosure(
        self,
        encrypted_query: Sequence[int],
        leaf: EncryptedKDLeaf,
    ) -> bool:
        for (
            encrypted_q,
            encrypted_lb,
            encrypted_ub,
        ) in zip(
            encrypted_query,
            leaf.encrypted_lower_bounds,
            leaf.encrypted_upper_bounds,
        ):
            if not self._secure_le(
                encrypted_lb,
                encrypted_q,
                signed=True,
            ):
                return False

            if not self._secure_le(
                encrypted_q,
                encrypted_ub,
                signed=True,
            ):
                return False

        return True

    # ==========================================================
    # SHORTEST POINT DE q A UNA REGIÓN
    # ==========================================================

    def encrypted_shortest_point(
        self,
        encrypted_query: Sequence[int],
        leaf: EncryptedKDLeaf,
    ) -> tuple[int, ...]:
        """
        sp_j =
          ub_j  si ub_j < q_j
          lb_j  si q_j < lb_j
          q_j   en otro caso
        """
        shortest: list[int] = []

        for (
            encrypted_q,
            encrypted_lb,
            encrypted_ub,
        ) in zip(
            encrypted_query,
            leaf.encrypted_lower_bounds,
            leaf.encrypted_upper_bounds,
        ):
            if self._secure_lt(
                encrypted_ub,
                encrypted_q,
                signed=True,
            ):
                shortest.append(
                    encrypted_ub
                )

            elif self._secure_lt(
                encrypted_q,
                encrypted_lb,
                signed=True,
            ):
                shortest.append(
                    encrypted_lb
                )

            else:
                shortest.append(
                    encrypted_q
                )

        return tuple(
            shortest
        )

    def secure_region_min_distance(
        self,
        encrypted_query: Sequence[int],
        leaf: EncryptedKDLeaf,
    ) -> int:
        encrypted_sp = (
            self.encrypted_shortest_point(
                encrypted_query,
                leaf,
            )
        )

        return (
            self.secure_squared_distance(
                encrypted_sp,
                encrypted_query,
            )
        )

    # ==========================================================
    # SCORE CIFRADO
    # ==========================================================

    def build_encrypted_score(
        self,
        encrypted_distance: int,
        record_id: int,
    ) -> int:
        if self.encrypted_kd_tree is None:
            raise RuntimeError(
                "KD-tree no configurado."
            )

        base = (
            self.encrypted_kd_tree.database_size
            + 1
        )

        encrypted_scaled_distance = (
            self.encrypted_scalar_multiply(
                encrypted_distance,
                base,
            )
        )

        encrypted_id = (
            self.transformer.encrypt(
                record_id
            )
        )

        return self.encrypted_add(
            encrypted_scaled_distance,
            encrypted_id,
        )

    # ==========================================================
    # TOP-K
    # ==========================================================

    def _candidate_le(
        self,
        left: _Candidate,
        right: _Candidate,
    ) -> bool:
        return self._secure_le(
            left.encrypted_score,
            right.encrypted_score,
            signed=False,
        )

    def _insert_candidate(
        self,
        candidates: list[_Candidate],
        candidate: _Candidate,
        k: int,
    ) -> None:
        candidates.append(
            candidate
        )

        index = (
            len(candidates) - 1
        )

        while index > 0:
            if self._candidate_le(
                candidates[index - 1],
                candidates[index],
            ):
                break

            (
                candidates[index - 1],
                candidates[index],
            ) = (
                candidates[index],
                candidates[index - 1],
            )

            index -= 1

        if len(candidates) > k:
            candidates.pop()

    def _evaluate_leaf(
        self,
        leaf: EncryptedKDLeaf,
        encrypted_query: Sequence[int],
        candidates: list[_Candidate],
        k: int,
        evaluated_ids: set[int],
    ) -> int:
        if self.encrypted_database is None:
            raise RuntimeError(
                "BD cifrada no configurada."
            )

        evaluations = 0

        for record_id in leaf.record_ids:
            if record_id in evaluated_ids:
                continue

            encrypted_record = (
                self.encrypted_database[
                    record_id
                ]
            )

            encrypted_distance = (
                self.secure_squared_distance(
                    encrypted_record,
                    encrypted_query,
                )
            )

            encrypted_score = (
                self.build_encrypted_score(
                    encrypted_distance,
                    record_id,
                )
            )

            self._insert_candidate(
                candidates,
                _Candidate(
                    record_id=record_id,
                    encrypted_distance=(
                        encrypted_distance
                    ),
                    encrypted_score=(
                        encrypted_score
                    ),
                ),
                k,
            )

            evaluated_ids.add(
                record_id
            )

            evaluations += 1

        return evaluations

    # ==========================================================
    # FASE 1: BUSCAR HOJA(S) QUE CONTIENEN q
    # ==========================================================

    def _find_candidate_leaves(
        self,
        encrypted_query: Sequence[int],
    ) -> list[EncryptedKDLeaf]:
        if self.encrypted_kd_tree is None:
            raise RuntimeError(
                "KD-tree no configurado."
            )

        return [
            leaf
            for leaf
            in self.encrypted_kd_tree.leaves
            if self.secure_point_enclosure(
                encrypted_query,
                leaf,
            )
        ]

    def _nearest_leaf_when_outside_root(
        self,
        encrypted_query: Sequence[int],
    ) -> EncryptedKDLeaf:
        """
        Fallback para consultas fuera de los límites globales
        observados durante la construcción del índice.
        """
        if self.encrypted_kd_tree is None:
            raise RuntimeError(
                "KD-tree no configurado."
            )

        best_leaf = None
        best_distance = None

        for leaf in (
            self.encrypted_kd_tree.leaves
        ):
            encrypted_distance = (
                self.secure_region_min_distance(
                    encrypted_query,
                    leaf,
                )
            )

            if best_leaf is None:
                best_leaf = leaf
                best_distance = (
                    encrypted_distance
                )
                continue

            if self._secure_lt(
                encrypted_distance,
                best_distance,
                signed=False,
            ):
                best_leaf = leaf
                best_distance = (
                    encrypted_distance
                )

        if best_leaf is None:
            raise RuntimeError(
                "El KD-tree no contiene hojas."
            )

        return best_leaf

    # ==========================================================
    # PROCESAMIENTO COMPLETO
    # ==========================================================

    def process_query(
        self,
        encrypted_query: Sequence[int],
        k: int,
    ) -> tuple:
        if self.encrypted_database is None:
            raise RuntimeError(
                "CloudB no recibió la BD cifrada."
            )

        if self.encrypted_kd_tree is None:
            raise RuntimeError(
                "CloudB no recibió el KD-tree."
            )

        if not (
            1
            <= k
            <= len(self.encrypted_database)
        ):
            raise ValueError(
                "k fuera de rango."
            )

        if (
            len(encrypted_query)
            != self.encrypted_kd_tree.dimensions
        ):
            raise ValueError(
                "Dimensión de consulta incorrecta."
            )

        # ------------------------------------------------------
        # 1. ENCRYPTED KD-TREE SEARCH
        # ------------------------------------------------------

        initial_leaves = (
            self._find_candidate_leaves(
                encrypted_query
            )
        )

        if not initial_leaves:
            initial_leaves = [
                self._nearest_leaf_when_outside_root(
                    encrypted_query
                )
            ]

        candidates: list[
            _Candidate
        ] = []

        evaluated_ids: set[int] = set()
        processed_leaf_ids: set[int] = set()

        distance_evaluations = 0

        for leaf in initial_leaves:
            distance_evaluations += (
                self._evaluate_leaf(
                    leaf,
                    encrypted_query,
                    candidates,
                    k,
                    evaluated_ids,
                )
            )

            processed_leaf_ids.add(
                leaf.leaf_id
            )

        # ------------------------------------------------------
        # 2 + 3. kNN RETRIEVAL + RESULT VERIFICATION
        # ------------------------------------------------------

        verified_leaves = 0
        additional_leaves = 0

        for leaf in (
            self.encrypted_kd_tree.leaves
        ):
            if (
                leaf.leaf_id
                in processed_leaf_ids
            ):
                continue

            verified_leaves += 1

            # Si aún no tenemos k candidatos,
            # necesariamente necesitamos más datos.
            should_process = (
                len(candidates) < k
            )

            if not should_process:
                encrypted_min_region_distance = (
                    self.secure_region_min_distance(
                        encrypted_query,
                        leaf,
                    )
                )

                encrypted_kth_distance = (
                    candidates[
                        -1
                    ].encrypted_distance
                )

                should_process = (
                    self._secure_le(
                        encrypted_min_region_distance,
                        encrypted_kth_distance,
                        signed=False,
                    )
                )

            if should_process:
                distance_evaluations += (
                    self._evaluate_leaf(
                        leaf,
                        encrypted_query,
                        candidates,
                        k,
                        evaluated_ids,
                    )
                )

                processed_leaf_ids.add(
                    leaf.leaf_id
                )

                additional_leaves += 1

        selected_ids = [
            candidate.record_id
            for candidate
            in candidates[:k]
        ]

        # CloudB entrega exclusivamente resultados cifrados.
        # Los identificadores y el patron de acceso son visibles.
        encrypted_neighbors = [
            tuple(self.encrypted_database[record_id])
            for record_id in selected_ids
        ]
        encrypted_distances = [
            candidate.encrypted_distance
            for candidate in candidates[:k]
        ]
        self.last_query_stats = {
            "total_leaves": len(self.encrypted_kd_tree.leaves),
            "initial_leaves": len(initial_leaves),
            "verified_leaves": verified_leaves,
            "additional_leaves": additional_leaves,
            "distance_evaluations": distance_evaluations,
            "database_size": self.encrypted_kd_tree.database_size,
        }
        return selected_ids, encrypted_neighbors, encrypted_distances

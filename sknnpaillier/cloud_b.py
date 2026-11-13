"""Evaluate encrypted nearest-neighbor queries via hierarchical KD-tree pruning.
Nonlinear operations use a trusted key manager in the same process.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Sequence

from sknnpaillier.kd_tree import EncryptedKDNode, EncryptedKDTree
from sknnpaillier.key_m import KeyManagementModule
from sknnpaillier.log import get_logger
from sknnpaillier.secure_data_t import SecureDataTransformationModule


@dataclass
class _Candidate:
    """Store an internal candidate and its encrypted ranking values.

    Attributes:
        record_id (int): Visible database row ID.
        encrypted_distance (int): Encrypted squared distance.
        encrypted_score (int): Encrypted distance/ID tie-breaking score.
    """
    record_id: int
    encrypted_distance: int
    encrypted_score: int


class CloudBQueryProcessingModule:

    """Evaluate nearest-neighbor queries over encrypted records.

    Record IDs and access patterns are visible. Distances and ranking
    scores must not wrap modulo n; no overflow exception is raised. The
    key manager reference does not enforce a separate trust boundary.

    Attributes:
        transformer (SecureDataTransformationModule): Encryption collaborator.
        key_manager (KeyManagementModule): Trusted nonlinear-operation provider.
        encrypted_database (Sequence[Sequence[int]] | None): Encrypted rows.
        encrypted_kd_tree (EncryptedKDTree | None): Encrypted hierarchical index.
        last_query_stats (dict[str, int]): Counters from the last completed query.
    """
    def __init__(
        self,
        transformer: SecureDataTransformationModule,
        key_manager: KeyManagementModule,
    ) -> None:
        """Configure collaborators and initialize empty query state.

        Args:
            transformer (SecureDataTransformationModule): Encryptor for this database.
            key_manager (KeyManagementModule): Manager for the same public key.

        Returns:
            None: Initializes the processor without a database or index.
        """
        self.transformer = (
            transformer
        )

        self.key_manager = (
            key_manager
        )

        self.encrypted_database: Sequence[Sequence[int]] | None = None

        self.encrypted_kd_tree: (
            EncryptedKDTree | None
        ) = None

        self.last_query_stats: dict[
            str,
            int,
        ] = {}
        self._total_leaves = 0

    # RECEPCIÓN DE BD E ÍNDICE

    def set_database(
        self,
        encrypted_database: Sequence[
            Sequence[int]
        ],
    ) -> None:
        """Attach encrypted records without copying or validating them.

        The caller must ensure that shape, key, and record IDs match the index.

        Args:
            encrypted_database (Sequence[Sequence[int]]): Ciphertext rows indexed by ID.

        Returns:
            None: Stores the supplied sequence.
        """
        self.encrypted_database = (
            encrypted_database
        )

    def set_kd_tree(
        self,
        encrypted_kd_tree: EncryptedKDTree,
    ) -> None:
        """Attach the encrypted leaf index.

        Consistency with database dimensions and record IDs is not validated.

        Args:
            encrypted_kd_tree (EncryptedKDTree): Index matching the attached database.

        Returns:
            None: Stores the supplied index.
        """
        self.encrypted_kd_tree = encrypted_kd_tree

        # Contar hojas una sola vez, al recibir el índice.
        def count_leaves(node: EncryptedKDNode) -> int:
            if node.record_ids is not None:
                return 1
            return sum(
                count_leaves(child)
                for child in (node.left, node.right)
                if child is not None
            )

        self._total_leaves = count_leaves(encrypted_kd_tree.root)

    # OPERACIONES HOMOMÓRFICAS PAILLIER

    def encrypted_add(
        self,
        encrypted_a: int,
        encrypted_b: int,
    ) -> int:
        """Add plaintext residues by multiplying ciphertexts.

        Args:
            encrypted_a (int): Ciphertext of the first operand.
            encrypted_b (int): Ciphertext of the second operand.

        Returns:
            int: Encryption of a + b modulo n.
        """
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
        """Multiply an encrypted residue by a plaintext scalar.

        Args:
            ciphertext (int): Encrypted operand.
            scalar (int): Multiplier, reduced modulo n.

        Returns:
            int: Encryption of the modular scalar product.
        """
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
        """Negate an encrypted plaintext residue.

        Args:
            ciphertext (int): Encrypted operand.

        Returns:
            int: Encryption of the additive inverse modulo n.
        """
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
        """Subtract plaintext residues using homomorphic negation.

        Args:
            encrypted_a (int): Ciphertext of the first operand.
            encrypted_b (int): Ciphertext of the second operand.

        Returns:
            int: Encryption of a - b modulo n.
        """
        return self.encrypted_add(
            encrypted_a,
            self.encrypted_negate(
                encrypted_b
            ),
        )

    # MULTIPLICACION SEGURA

    def secure_multiply(
        self,
        encrypted_a: int,
        encrypted_b: int,
    ) -> int:

        """Multiply plaintext residues using random masks and the key manager.

        Args:
            encrypted_a (int): Ciphertext of the first operand.
            encrypted_b (int): Ciphertext of the second operand.

        Returns:
            int: Encryption of a * b modulo n.
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

    # ESSED-LIKE

    def secure_squared_distance(
        self,
        encrypted_record: Sequence[int],
        encrypted_query: Sequence[int],
    ) -> int:
        """Compute an encrypted squared Euclidean distance.

        The intended nonnegative distance must be below n. Modular wrapping
        is silent and can invalidate subsequent comparisons.

        Args:
            encrypted_record (Sequence[int]): Encrypted record coordinates.
            encrypted_query (Sequence[int]): Encrypted query of the same dimension.

        Returns:
            int: Encrypted sum of squared coordinate differences modulo n.

        Raises:
            ValueError: If record and query lengths differ.
        """
        if (
            len(encrypted_record)
            != len(encrypted_query)
        ):
            raise ValueError(
                "Record and query must have "
                "the same dimension."
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

    # COMPARACIONES CONTROLADAS

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


    def secure_point_enclosure(
        self,
        encrypted_query: Sequence[int],
        leaf: EncryptedKDNode,
    ) -> bool:
        """Check whether a query lies within a leaf region.

        Signed comparisons decode coordinates in the trusted manager.
        Matching query and bound lengths are a caller requirement; the region
        helpers use zip and do not independently validate every length.

        Args:
            encrypted_query (Sequence[int]): Encrypted coordinates matching the leaf.
            leaf (EncryptedKDNode): Region with encrypted inclusive bounds.

        Returns:
            bool: Whether every compared coordinate lies within its bounds.
        """
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

    # SHORTEST POINT DE q A UNA REGIÓN

    def encrypted_shortest_point(
        self,
        encrypted_query: Sequence[int],
        leaf: EncryptedKDNode,
    ) -> tuple[int, ...]:
        """Clamp each query coordinate to its region interval.

        Signed comparisons decode coordinates in the trusted manager.
        Matching query and bound lengths are a caller requirement; the region
        helpers use zip and do not independently validate every length.

        Args:
            encrypted_query (Sequence[int]): Encrypted coordinates matching the leaf.
            leaf (EncryptedKDNode): Region with encrypted inclusive bounds.

        Returns:
            tuple[int, ...]: Encrypted closest point in the region.
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
        leaf: EncryptedKDNode,
    ) -> int:
        """Compute the encrypted minimum squared distance to a region.

        Signed comparisons decode coordinates in the trusted manager.
        Matching query and bound lengths are a caller requirement; the region
        helpers use zip and do not independently validate every length.

        Args:
            encrypted_query (Sequence[int]): Encrypted coordinates matching the leaf.
            leaf (EncryptedKDNode): Region with encrypted inclusive bounds.

        Returns:
            int: Encrypted squared distance from the query to the region.

        Raises:
            ValueError: If the resulting closest point and query lengths differ.
        """
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
        """Encode distance ordering and deterministic ID tie-breaking.

        The complete score must be below n. Overflow is not detected.

        Args:
            encrypted_distance (int): Encrypted nonnegative squared distance.
            record_id (int): Database row ID.

        Returns:
            int: Encryption of distance * (database_size + 1) + record_id modulo n.

        Raises:
            RuntimeError: If no KD-tree is configured.
        """
        if self.encrypted_kd_tree is None:
            raise RuntimeError(
                "KD-tree not configured."
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

    # TOP-K

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
        leaf: EncryptedKDNode,
        encrypted_query: Sequence[int],
        candidates: list[_Candidate],
        k: int,
        evaluated_ids: set[int],
    ) -> int:
        if self.encrypted_database is None:
            raise RuntimeError(
                "Encrypted database not configured."
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

    # BÚSQUEDA JERÁRQUICA CON PODA DE SUBÁRBOLES

    def process_query(
        self,
        encrypted_query: Sequence[int],
        k: int,
    ) -> tuple[list[int], list[tuple[int, ...]], list[int]]:
        """Find exact top-k using encrypted region lower bounds.

        Explore the most promising child first. Prune a subtree only when
        its minimum possible distance is STRICTLY greater than the current
        kth distance. Equality must be explored to retain record-ID tie
        ordering. All comparisons depend on the trusted key manager.
        """
        if self.encrypted_database is None:
            raise RuntimeError("CloudB did not receive the encrypted database.")
        if self.encrypted_kd_tree is None:
            raise RuntimeError("CloudB did not receive the encrypted KD-tree.")
        if not 1 <= k <= len(self.encrypted_database):
            raise ValueError("k is out of range.")
        if len(encrypted_query) != self.encrypted_kd_tree.dimensions:
            raise ValueError("Dimension of query is incorrect.")

        t_inicio = time.perf_counter()
        candidates: list[_Candidate] = []
        evaluated_ids: set[int] = set()
        stats = {
            "visited_nodes": 0,
            "visited_leaves": 0,
            "pruned_subtrees": 0,
            "region_distance_evaluations": 0,
            "distance_evaluations": 0,
        }

        def region_distance(node: EncryptedKDNode) -> int:
            stats["region_distance_evaluations"] += 1
            return self.secure_region_min_distance(encrypted_query, node)

        def visit(node: EncryptedKDNode, lower_bound: int | None = None) -> None:
            # Compare with best kth *after* exploring previous siblings.
            if len(candidates) == k:
                if lower_bound is None:
                    lower_bound = region_distance(node)
                # Prune strictly greater distances, not equal distances:
                # equal distances can still improve the ID tie-break.
                if self._secure_lt(
                    candidates[-1].encrypted_distance,
                    lower_bound,
                    signed=False,
                ):
                    stats["pruned_subtrees"] += 1
                    return

            stats["visited_nodes"] += 1
            if node.record_ids is not None:
                stats["visited_leaves"] += 1
                stats["distance_evaluations"] += self._evaluate_leaf(
                    node, encrypted_query, candidates, k, evaluated_ids
                )
                return

            children = [child for child in (node.left, node.right)
                        if child is not None]
            if not children:
                raise ValueError("KD-tree internal node has no children.")

            # Order at most two children by their encrypted region distances.
            child_bounds = [(child, region_distance(child)) for child in children]
            if len(child_bounds) == 2 and self._secure_lt(
                child_bounds[1][1], child_bounds[0][1], signed=False
            ):
                child_bounds.reverse()

            for child, bound in child_bounds:
                visit(child, bound)

        visit(self.encrypted_kd_tree.root)

        if len(candidates) != k:
            raise RuntimeError("Index did not provide k distinct records.")

        selected_ids = [c.record_id for c in candidates]
        encrypted_neighbors = [
            tuple(self.encrypted_database[record_id])
            for record_id in selected_ids
        ]
        encrypted_distances = [c.encrypted_distance for c in candidates]

        # Preserve the keys expected by experimento4.py.
        self.last_query_stats = {
            "total_leaves": self._total_leaves,
            "initial_leaves": stats["visited_leaves"],
            "verified_leaves": 0,
            "additional_leaves": 0,
            **stats,
            "database_size": self.encrypted_kd_tree.database_size,
        }

        t_fin = time.perf_counter()
        logger = get_logger("sknn.cloudb")
        logger.debug({
            "event": "hierarchical_query",
            "k": k,
            "time_s": round(t_fin - t_inicio, 6),
            **self.last_query_stats,
        })
        logger.info({
            "event": "query_total",
            "k": k,
            "evaluated_distances": stats["distance_evaluations"],
            "time_s": round(t_fin - t_inicio, 6),
        })
        return selected_ids, encrypted_neighbors, encrypted_distances

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Config import Config
from SecureDataT import (
    SecureDataTransformationModule,
)


class DataOwnerModule:
    """
    Data Owner.

    Responsabilidades:
        1. Cargar el dataset.
        2. Preparar y validar los datos.
        3. Obtener la base de datos DB.
        4. Mantener inicialmente DB en texto plano.
        5. Enviar DB a SecureDataT.
        6. Iniciar la construcción y cifrado del KD-tree.
        7. Externalizar E(DB) y el índice cifrado a CloudB.
    """

    def __init__(
        self,
        config: Config,
        secure_data_transformer:
            SecureDataTransformationModule,
    ) -> None:

        self.config = config

        self.secure_data_transformer = (
            secure_data_transformer
        )

        self.full_dataset: (
            np.ndarray | None
        ) = None

        self.database: (
            np.ndarray | None
        ) = None

        self.metadata: (
            pd.DataFrame | None
        ) = None

        self.dataset_permutation: (
            np.ndarray | None
        ) = None

    # ==========================================================
    # DATASET SINTÉTICO
    # ==========================================================

    def _load_synthetic(
        self,
    ) -> np.ndarray:

        total_rows = (
            self.config.db_rows
            + self.config.queries
        )

        rng = np.random.default_rng(
            self.config.random_seed
        )

        # Shuttle utiliza 9 atributos.
        attributes = 9

        return rng.integers(
            0,
            100,
            size=(
                total_rows,
                attributes,
            ),
            dtype=np.int64,
        )
    # DATASET SHUTTLE
    def _load_shuttle(
        self,
    ) -> np.ndarray:

        try:
            from ucimlrepo import (
                fetch_ucirepo,
            )

        except ImportError as error:

            raise RuntimeError(
                "Para usar Shuttle instale "
                "ucimlrepo con:\n"
                "pip install ucimlrepo"
            ) from error

        dataset = fetch_ucirepo(
            id=148
        )

        features = (
            dataset.data.features
        )

        if (
            features is None
            or features.empty
        ):
            raise RuntimeError(
                "El dataset Shuttle "
                "no contiene atributos."
            )

        return features.to_numpy(
            dtype=np.int64
        )

    # DATASET CSV

    def _load_csv(
        self,
        path: Path,
    ) -> np.ndarray:

        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: "
                f"{path}"
            )

        dataframe = pd.read_csv(
            path
        )
        

        if (
            "x" in dataframe.columns
            and "y" in dataframe.columns
        ):

            print(
                "CSV espacial detectado."
            )

            print(
                "Columnas usadas por SKNN: "
                "x, y"
            )


            metadata_columns = [
                column
                for column in [
                    "LINEARID",
                    "FULLNAME",
                    "RTTYP",
                    "MTFCC",
                ]
                if column
                in dataframe.columns
            ]

            if metadata_columns:

                self.metadata = (
                    dataframe[
                        metadata_columns
                    ].copy()
                )

            # ----------------------------------------------
            # El algoritmo SKNN recibe solamente:
            #
            #     [x, y]
            # ----------------------------------------------

            spatial_data = (
                dataframe[
                    [
                        "x",
                        "y",
                    ]
                ].copy()
            )

            spatial_data = (
                spatial_data.apply(
                    pd.to_numeric,
                    errors="raise",
                )
            )

            return (
                spatial_data.to_numpy(
                    dtype=np.int64
                )
            )

        # ------------------------------------------------------
        # CASO CSV GENÉRICO
        # ------------------------------------------------------

        if (
            self.config.drop_last_column
        ):

            dataframe = (
                dataframe.iloc[
                    :,
                    :-1,
                ]
            )

        dataframe = (
            dataframe.apply(
                pd.to_numeric,
                errors="raise",
            )
        )

        return dataframe.to_numpy(
            dtype=np.int64
        )

    # ==========================================================
    # CARGAR Y PREPARAR DATASET
    # ==========================================================

    def load_dataset(
        self,
    ) -> np.ndarray:
        """
        Carga el dataset dependiendo de Config.source.

        Para CSV espaciales TIGER:
            1. Lee las columnas x, y.
            2. Conserva los metadatos aparte.
            3. Mezcla todo el dataset con una
               permutación reproducible.
            4. Aplica exactamente la misma permutación
               a los metadatos.

        db_rows filas constituyen una muestra distribuida del dataset
        completo y no únicamente las primeras filas
        del shapefile original.
        """

        if (
            self.config.source
            == "synthetic"
        ):

            data = (
                self._load_synthetic()
            )

        elif (
            self.config.source
            == "shuttle"
        ):

            data = (
                self._load_shuttle()
            )

        elif (
            self.config.source
            == "csv"
        ):

            if (
                self.config.csv_path
                is None
            ):
                raise ValueError(
                    "Debe especificar "
                    "--csv-path cuando "
                    "--source=csv."
                )

            data = (
                self._load_csv(
                    self.config.csv_path
                )
            )

        else:

            raise ValueError(
                "Fuente de datos "
                f"no soportada: "
                f"{self.config.source}"
            )


        # VALIDA CANTIDAD MÍNIMA

        required_rows = (
            self.config.db_rows
            + self.config.queries
        )

        if (
            len(data)
            < required_rows
        ):
            raise ValueError(
                f"Se necesitan al menos "
                f"{required_rows} registros, "
                f"pero solo existen "
                f"{len(data)}."
            )

        # EL ESQUEMA ACTUAL TRABAJA CON ENTEROS

        data = (
            np.rint(
                data
            )
            .astype(
                np.int64
            )
        )

        if (
            self.config.source
            == "csv"
        ):

            rng = (
                np.random.default_rng(
                    self.config.random_seed
                )
            )

            self.dataset_permutation = (
                rng.permutation(
                    len(data)
                )
            )

            data = (
                data[
                    self.dataset_permutation
                ]
            )

            if (
                self.metadata
                is not None
            ):

                self.metadata = (
                    self.metadata
                    .iloc[
                        self.dataset_permutation
                    ]
                    .reset_index(
                        drop=True
                    )
                )

            print()
            print(
                "Dataset CSV mezclado "
                "de forma reproducible."
            )

            print(
                "Semilla:",
                self.config.random_seed,
            )

        # ------------------------------------------------------
        # GUARDAR DATASET COMPLETO
        # ------------------------------------------------------

        self.full_dataset = (
            data
        )

        # ------------------------------------------------------
        # INFORMACIÓN DEL DATASET
        # ------------------------------------------------------

        print()
        print(
            "Dataset cargado:"
        )

        print(
            f"  Registros     : "
            f"{len(data)}"
        )

        print(
            f"  Dimensiones   : "
            f"{data.shape[1]}"
        )

        if (
            data.shape[1] == 2
        ):

            print(
                f"  X             : "
                f"{data[:, 0].min()} "
                f"- "
                f"{data[:, 0].max()}"
            )

            print(
                f"  Y             : "
                f"{data[:, 1].min()} "
                f"- "
                f"{data[:, 1].max()}"
            )

        return (
            data
        )

    # PREPARAR BASE DE DATOS

    def prepare_database(
        self,
    ) -> np.ndarray:
        """
        Selecciona los primeros db_rows registros
        del dataset ya preparado.

        Para CSV/TIGER, el dataset ya fue mezclado
        previamente con una permutación reproducible.
        """

        if (
            self.full_dataset
            is None
        ):
            self.load_dataset()

        assert (
            self.full_dataset
            is not None
        )

        self.database = (
            self.full_dataset[
                :self.config.db_rows
            ].copy()
        )

        print()
        print(
            "Base de datos preparada:"
        )

        print(
            f"  Registros     : "
            f"{len(self.database)}"
        )

        print(
            f"  Dimensiones   : "
            f"{self.database.shape[1]}"
        )

        return (
            self.database
        )

    # OUTSOURCING

    def outsource_database(
        self,
    ) -> None:
        """
        Envía DB a SecureDataT.

        SecureDataT:
            - construye el KD-tree en claro;
            - obtiene las hojas;
            - cifra DB;
            - cifra lb y ub;
            - envía E(DB) y EKDTree a CloudB.
        """

        if (
            self.database
            is None
        ):
            self.prepare_database()

        assert (
            self.database
            is not None
        )

        self.secure_data_transformer\
            .send_database_to_cloud(
                database=(
                    self.database
                ),
                leaf_capacity=(
                    self.config.leaf_capacity
                ),
            )

    # ACCESO PARA VALIDACIÓN EXPERIMENTAL

    def get_database(
        self,
    ) -> np.ndarray:

        if (
            self.database
            is None
        ):
            raise RuntimeError(
                "La base de datos aún "
                "no ha sido preparada."
            )

        return (
            self.database.copy()
        )

    def get_full_dataset(
        self,
    ) -> np.ndarray:

        if (
            self.full_dataset
            is None
        ):
            raise RuntimeError(
                "El dataset aún "
                "no ha sido cargado."
            )

        return (
            self.full_dataset.copy()
        )

    def get_metadata(
        self,
    ) -> pd.DataFrame:

        if (
            self.metadata
            is None
        ):
            raise RuntimeError(
                "No existen metadatos "
                "para este dataset."
            )

        return (
            self.metadata.copy()
        )

    def get_dataset_permutation(
        self,
    ) -> np.ndarray:

        if (
            self.dataset_permutation
            is None
        ):
            raise RuntimeError(
                "No existe una permutación "
                "para este dataset."
            )

        return (
            self.dataset_permutation.copy()
        )
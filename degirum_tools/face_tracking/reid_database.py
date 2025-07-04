#
# reid_database.py: ReID database handling class
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements ReID_Database class to manage object embeddings and attributes in LanceDB database
#


import hashlib
import threading
import lancedb
import numpy as np
from typing import Any, Tuple, Dict, List, Optional


class ReID_Database:
    """
    Class to hold the database of object embeddings.
    """

    tbl_embeddings = "embeddings"  # name of the table for object embeddings
    tbl_attributes = "attributes"  # name of the table for object attributes

    key_embedding = "vector"  # key for the embedding vector (must be "vector" to define vector embedding field)
    key_embedding_hash = (
        "embedding_hash"  # key for the embedding hash (used for deduplication)
    )
    key_object_id = "object_id"  # key for the object ID
    key_attributes = "attributes"  # key for the object attributes

    def __init__(self, db_path: str, threshold: float = 0.4):
        """
        Constructor.

        Args:
            db_path (str): Path to the database file.
            threshold (float): Threshold for the embedding similarity metric.
        """
        self._lock = threading.Lock()
        self._db = lancedb.connect(db_path)
        self._tables: Dict[str, lancedb.table.Table] = {}
        self._threshold = threshold

    def list_objects(self) -> dict:
        """
        List all object IDs in the database.

        Returns:
            dict: map of object ID to attributes
        """
        with self._lock:
            table, _ = self._open_table(ReID_Database.tbl_attributes)
            if table is None:
                return {}

            objects = table.search().to_list()
            return {
                obj[ReID_Database.key_object_id]: obj[ReID_Database.key_attributes]
                for obj in objects
            }

    def add_object(self, object_id: str, attributes: Any):
        """
        Add or change object attributes in the object attributes table.

        Args:
            object_id (str): The unique object ID.
            attributes (Any): The attributes of the object to add/change
        """

        with self._lock:
            data = [
                {
                    ReID_Database.key_object_id: object_id,
                    ReID_Database.key_attributes: attributes,
                }
            ]
            table, newly_created = self._open_table(ReID_Database.tbl_attributes, data)
            if table is not None and not newly_created:
                # if the table already exists, check if the object ID exists
                result = (
                    table.search()
                    .where(f"{ReID_Database.key_object_id} == '{object_id}'")
                    .to_list()
                )
                if result:
                    # update existing attributes
                    table.update(
                        where=f"{ReID_Database.key_object_id} == '{object_id}'",
                        values=data[0],
                    )
                else:
                    # object ID does not exist, add new attributes
                    table.add(data)

    def count_embeddings(self):
        """
        Count all object embeddings in the database.

        Returns:
            dict: A dictionary where the key is the object ID and the value is the tuple containing count of embeddings for that object and its attributes.
        """
        from collections import Counter

        with self._lock:
            embeddings_table, _ = self._open_table(ReID_Database.tbl_embeddings)
            attributes_table, _ = self._open_table(ReID_Database.tbl_attributes)
            if embeddings_table is None or attributes_table is None:
                return {}

            object_ids = (
                embeddings_table.search()
                .select(["object_id"])
                .to_arrow()["object_id"]
                .to_pylist()
            )

            object_counts = dict(Counter(object_ids))

            objects = {
                obj[ReID_Database.key_object_id]: obj[ReID_Database.key_attributes]
                for obj in attributes_table.search().to_list()
            }
            return {id: (count, objects[id]) for id, count in object_counts.items()}

    def add_embeddings(
        self,
        object_id: str,
        embeddings: List[np.ndarray],
        *,
        dedup: bool = True,
    ):
        """
        Add an embedding for given object ID to the database.

        Args:
            object_id (str): The object ID.
            embeddings (List[np.ndarray]): The list of embedding vectors.
            dedup (bool): Whether to deduplicate embeddings. If True, only unique embeddings will be added.
        """

        with self._lock:
            data = [
                {
                    ReID_Database.key_object_id: object_id,
                    ReID_Database.key_embedding: embedding,
                    ReID_Database.key_embedding_hash: hashlib.sha256(
                        embedding.tobytes()
                    ).hexdigest(),
                }
                for embedding in embeddings
            ]
            table, newly_created = self._open_table(ReID_Database.tbl_embeddings, data)
            if table is not None and not newly_created:
                if dedup:  # check for duplicates
                    existing_hashes = set(
                        table.search()
                        .select([ReID_Database.key_embedding_hash])
                        .to_arrow()[ReID_Database.key_embedding_hash]
                        .to_pylist()
                    )
                    data_hashes = {d[ReID_Database.key_embedding_hash] for d in data}
                    duplicate_hashes = existing_hashes.intersection(data_hashes)

                    # Remove rows from the table where the embedding hash is in duplicate_hashes
                    if duplicate_hashes:
                        hashes_str = "', '".join(duplicate_hashes)
                        table.delete(
                            where=f"{ReID_Database.key_embedding_hash} IN ('{hashes_str}')"
                        )

                if data:
                    table.add(data)

    def get_attributes_by_id(self, object_id: np.ndarray) -> Optional[Any]:
        """
        Get object attributes by object ID

        Args:
            object_id (np.ndarray): Object ID string.

        Returns:
            Optional[Any]: The attributes of the object or None if not found.
        """

        with self._lock:
            attributes = None
            attributes_table, _ = self._open_table(ReID_Database.tbl_attributes)
            if attributes_table is not None:
                # query the attributes table for the object attributes
                attribute_result = (
                    attributes_table.search()
                    .where(f"{ReID_Database.key_object_id} == '{object_id}'")
                    .to_list()
                )
                if attribute_result:
                    attributes = attribute_result[0][ReID_Database.key_attributes]

            return attributes

    def get_attributes_by_embedding(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], Optional[Any]]:
        """
        Get the object ID and its attributes by its embedding.

        Args:
            embedding (np.ndarray): The embedding vector.

        Returns:
            tuple: The tuple containing object ID and attributes of the object; (None, None) if not found.
        """

        no_result = (None, None)

        with self._lock:
            embeddings_table, _ = self._open_table(ReID_Database.tbl_embeddings)
            if embeddings_table is None:
                return no_result

            # query the embedding table for the closest embedding
            try:
                embedding_result = (
                    embeddings_table.search(
                        embedding, vector_column_name=ReID_Database.key_embedding
                    )
                    .metric("cosine")
                    .distance_range(0.0, self._threshold)
                    .limit(1)
                    .to_list()
                )
            except Exception:
                return no_result

            if not embedding_result:
                return no_result

            object_id = embedding_result[0][ReID_Database.key_object_id]
            attributes = None

            attributes_table, _ = self._open_table(ReID_Database.tbl_attributes)
            if attributes_table is not None:
                # query the attributes table for the object attributes
                attribute_result = (
                    attributes_table.search()
                    .where(f"{ReID_Database.key_object_id} == '{object_id}'")
                    .to_list()
                )
                if attribute_result:
                    attributes = attribute_result[0][ReID_Database.key_attributes]

            return object_id, attributes

    def _open_table(
        self, table_name, data: Optional[list] = None
    ) -> Tuple[Optional[lancedb.table.Table], bool]:
        """
        Open the table in the database if it is not opened yet.
        If the table does not exist, create it.

        Args:
            table_name (str): Name of the table to open or create.
            data (Optional[list]): Data to create the table with or add to existing table.

        Returns:
            tuple: The tuple containing opened table (or None if the table was not created) and a boolean indicating if table was newly created.
        """

        newly_created = False
        table = self._tables.get(table_name)
        if table is None:
            # table is not opened yet, check if it exists
            if table_name in self._db.table_names():
                # table exists, open it
                table = self._db.open_table(table_name)
                self._tables[table_name] = table
            else:
                # table does not exist, create it
                if data is not None:
                    table = self._db.create_table(table_name, data=data)
                    self._tables[table_name] = table
                    newly_created = True

        return table, newly_created

"""
The module contains a definition of the class of connections in a neuron
"""


class Connection:
    """
    Connection in a neuron
    """

    def __init__(self, node, weight):
        self.node = node
        self.weight = weight


class Connections:
    """
    Connections in a neuron
    """

    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def edit_connection(self, old_node, new_node):
        connection = next(conn for conn in self.connections if conn.node.node_id == old_node.node_id)
        connection.node = new_node

    def is_valid_connection(self, connection) -> bool:
        if connection in self.connections:
            return False
        return True
